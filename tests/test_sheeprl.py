#!/usr/bin/env python3
import importlib
import os
import shutil
import sys
import warnings
from unittest import mock

import pytest
from diambra.arena.utils.engine_mock import load_mocker

# Add the scripts directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sheeprl"))
sys.path.append(ROOT_DIR)

import evaluate
import train

STANDARD_ARGS = [
    os.path.join(ROOT_DIR, "__main__.py"),
    "env.capture_video=False",
    "metric.log_level=0",
    "checkpoint.every=10000000",
    "buffer.memmap=False",
]


def _test_agent(mocker, agent, kwargs):
    load_mocker(mocker)

    agent = importlib.import_module(f"agent-{agent}")
    os.environ["DIAMBRA_ENVS"] = "127.0.0.1:32781"
    agent.main(**kwargs)


def _test_train_eval(
    mocker, n_envs, args, evaluation=False, root_dir=None, run_name=None
):
    load_mocker(mocker)

    try:
        # Environment setup
        initial_port = 32781
        envs = f"127.0.0.1:{initial_port}"
        for i in range(1, n_envs):
            envs += f" 127.0.0.1:{initial_port + i}"
        os.environ["DIAMBRA_ENVS"] = envs

        # SheepRL config folder setup
        os.environ["SHEEPRL_SEARCH_PATH"] = (
            "file://sheeprl/configs;pkg://sheeprl.configs"
        )

        # Execution of the train script
        with mock.patch.object(sys, "argv", STANDARD_ARGS + args):
            train.train()

        if evaluation:
            # Take checkpoint
            ckpt_root = os.path.join("logs", "runs", root_dir, run_name)
            ckpt_dir = sorted([d for d in os.listdir(ckpt_root) if "version" in d])[-1]
            ckpt_path = os.path.join(ckpt_root, ckpt_dir, "checkpoint")
            ckpt_file_name = os.listdir(ckpt_path)[-1]
            ckpt_path = os.path.join(ckpt_path, ckpt_file_name)
            # Execution of the evaluate script
            with mock.patch.object(
                sys, "argv", STANDARD_ARGS[:1] + [f"checkpoint_path={ckpt_path}"]
            ):
                evaluate.run()

        # Delete log folder
        try:
            shutil.rmtree("./logs", False, None)
        except (OSError, WindowsError):
            warnings.warn("Unable to delete folder {}.".format("./logs"))
        return 0
    except Exception as e:
        print(e)
        return 1


def test_sheeprl_train_base(mocker):
    assert (
        _test_train_eval(
            mocker,
            2,
            ["exp=custom_exp", "checkpoint.save_last=False"],
        )
        == 0
    )


def test_sheeprl_train_parallel_envs(mocker):
    assert (
        _test_train_eval(
            mocker,
            6,
            ["exp=custom_parallel_env_exp", "checkpoint.save_last=False"],
        )
        == 0
    )


def test_sheeprl_train_fabric(mocker):
    assert (
        _test_train_eval(
            mocker,
            2,
            [
                "exp=custom_fabric_exp",
                "fabric.accelerator=cpu",
                "fabric.devices=1",
                "checkpoint.save_last=False",
            ],
        )
        == 0
    )


def test_sheeprl_train_metrics(mocker):
    assert (
        _test_train_eval(
            mocker,
            2,
            [
                "exp=custom_metric_exp",
                "fabric.accelerator=cpu",
                "fabric.devices=1",
                "checkpoint.save_last=False",
            ],
        )
        == 0
    )


def test_sheeprl_evaluation(mocker):
    assert (
        _test_train_eval(
            mocker,
            3,
            [
                "exp=custom_exp",
                "checkpoint.save_last=True",
                "root_dir=pytest_ppo",
                "run_name=eval",
            ],
            evaluation=True,
            root_dir="pytest_ppo",
            run_name="eval",
        )
        == 0
    )


def test_sheeprl_ppo_agent(mocker):
    cfg_path = os.path.join(
        ROOT_DIR, "/fake-logs/runs/ppo/doapp/fake-experiment/version_0/config.yaml"
    )
    checkpoint_path = os.path.join(
        ROOT_DIR,
        "/fake-logs/runs/ppo/doapp/fake-experiment/version_0/checkpoint/ckpt_1024_0.ckpt",
    )
    assert _test_agent(
        mocker,
        "ppo",
        {"cfg_path": cfg_path, "checkpoint_path": checkpoint_path, "test": True},
    )


def test_sheeprl_dreamer_v3_agent(mocker):
    cfg_path = os.path.join(
        ROOT_DIR,
        "/fake-logs/runs/dreamer_v3/doapp/fake-experiment/version_0/config.yaml",
    )
    checkpoint_path = os.path.join(
        ROOT_DIR,
        "/fake-logs/runs/dreamer_v3/doapp/fake-experiment/version_0/checkpoint/ckpt_1024_0.ckpt",
    )
    assert _test_agent(
        mocker,
        "dreamer_v3",
        {"cfg_path": cfg_path, "checkpoint_path": checkpoint_path, "test": True},
    )

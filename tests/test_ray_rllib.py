#!/usr/bin/env python3
import sys
import pytest
from os.path import expanduser
import os
from diambra.arena.utils.engine_mock import DiambraEngineMock

# Add the scripts directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ray_rllib"))
sys.path.append(root_dir)

import basic, saving_loading_evaluating, parallel_envs, dict_obs_space, agent

# Example Usage:
# pytest
# (optional)
#    module.py (Run specific module)
#    -s (show output)
#    -k "expression" (filter tests using case-insensitive with parts of the test name and/or parameters values combined with boolean operators, e.g. "wrappers and doapp")

def func(script, mocker, *args):

    diambra_engine_mock = DiambraEngineMock()

    mocker.patch("diambra.arena.engine.interface.DiambraEngine.__init__", diambra_engine_mock._mock__init__)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine._env_init", diambra_engine_mock._mock_env_init)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine._reset", diambra_engine_mock._mock_reset)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine._step_1p", diambra_engine_mock._mock_step_1p)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine._step_2p", diambra_engine_mock._mock_step_2p)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine.close", diambra_engine_mock._mock_close)

    try:
        os.environ["DIAMBRA_ENVS"] = "0.0.0.0:50051"
        return script.main(*args)
    except Exception as e:
        print(e)
        return 1

trained_model_folder = os.path.join(root_dir, "results/doapp_sr6_84x5_mdas_c/")
env_spaces_descriptor_path = os.path.join(trained_model_folder, "diambra_ray_env_spaces")
#[parallel_envs, ()] # Not possible to test parallel_envs script as it requires multiple envs and the mocker does not work with child processes / threads
scripts = [[basic, ()], [saving_loading_evaluating, ()], [dict_obs_space, ()], [agent, (trained_model_folder, env_spaces_descriptor_path)]]

@pytest.mark.parametrize("script", scripts)
def test_ray_rllib_scripts(script, mocker):

    assert func(script[0], mocker, *script[1]) == 0

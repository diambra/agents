#!/usr/bin/env python3
import pytest
import sys
from os.path import expanduser
import os
from diambra.arena.utils.engine_mock import DiambraEngineMock

# Add the scripts directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stable_baselines3"))
sys.path.append(root_dir)

import basic, saving_loading_evaluating, parallel_envs, dict_obs_space, training, agent

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

cfg_file = os.path.join(root_dir, "cfg_files/sfiii3n/sr6_128x4_das_nc.yaml")
scripts = [[basic, ()], [saving_loading_evaluating, ()], [parallel_envs, ()], [dict_obs_space, ()], [training, (cfg_file,)], [agent, (cfg_file, "model")]]

@pytest.mark.parametrize("script", scripts)
def test_sb3_scripts(script, mocker):

    assert func(script[0], mocker, *script[1]) == 0

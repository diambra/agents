#!/usr/bin/env python3
import pytest
import sys
import os
from diambra.arena.utils.engine_mock import load_mocker

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
    load_mocker(mocker)

    try:
        os.environ["DIAMBRA_ENVS"] = "0.0.0.0:50051"
        return script.main(*args)
    except Exception as e:
        print(e)
        return 1

cfg_file1 = os.path.join(root_dir, "cfg_files/sfiii3n/sr6_128x4_das_nc.yaml")
cfg_file2 = os.path.join(root_dir, "cfg_files/doapp/sr6_128x4_das_nc.yaml")
scripts = [[basic, ()], [saving_loading_evaluating, ()], [parallel_envs, ()], [dict_obs_space, ()],
           [training, (cfg_file1,)], [training, (cfg_file2,)], [agent, (cfg_file1, "model", True)], [agent, (cfg_file2, "model", True)]]

@pytest.mark.parametrize("script", scripts)
def test_sb3_scripts(script, mocker):
    assert func(script[0], mocker, *script[1]) == 0

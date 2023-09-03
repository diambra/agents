#!/usr/bin/env python3
import pytest
import sys
import os
from diambra.arena.utils.engine_mock import load_mocker

# Add the scripts directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stable_baselines"))
sys.path.append(root_dir)

import training, agent

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

cfg_file1 = os.path.join(root_dir, "cfg_files/doapp/sr6_128x4_das_nc.yaml")
cfg_file2 = os.path.join(root_dir, "cfg_files/sfiii3n/sr6_128x4_das_nc.yaml")
cfg_file3 = os.path.join(root_dir, "cfg_files/tektagt/sr6_128x4_das_nc.yaml")
cfg_file4 = os.path.join(root_dir, "cfg_files/umk3/sr6_128x4_das_nc.yaml")
cfg_file5 = os.path.join(root_dir, "cfg_files/samsh5sp/sr6_128x4_das_nc.yaml")
cfg_file6 = os.path.join(root_dir, "cfg_files/kof98umh/sr6_128x4_das_nc.yaml")
scripts = [[training, (cfg_file1,)], [training, (cfg_file2,)], [training, (cfg_file3,)], [training, (cfg_file4,)],
           [training, (cfg_file5,)], [training, (cfg_file6,)], [agent, (cfg_file1, "model")]]

@pytest.mark.parametrize("script", scripts)
def test_sb_scripts(script, mocker):
    assert func(script[0], mocker, *script[1]) == 0

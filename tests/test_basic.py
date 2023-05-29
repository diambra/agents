#!/usr/bin/env python3
import pytest
import sys
import random
from os.path import expanduser
import os
from diambra.arena.utils.engine_mock import DiambraEngineMock

# Add the scripts directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from basic.no_action import agent as NoActionAgent
from basic.random_1 import agent as RandomAgent1
from basic.random_2 import agent as RandomAgent2

# Example Usage:
# pytest
# (optional)
#    module.py (Run specific module)
#    -s (show output)
#    -k "expression" (filter tests using case-insensitive with parts of the test name and/or parameters values combined with boolean operators, e.g. "wrappers and doapp")

def func(agent, mocker):

    diambra_engine_mock = DiambraEngineMock()

    mocker.patch("diambra.arena.engine.interface.DiambraEngine.__init__", diambra_engine_mock._mock__init__)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine._env_init", diambra_engine_mock._mock_env_init)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine._reset", diambra_engine_mock._mock_reset)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine._step_1p", diambra_engine_mock._mock_step_1p)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine._step_2p", diambra_engine_mock._mock_step_2p)
    mocker.patch("diambra.arena.engine.interface.DiambraEngine.close", diambra_engine_mock._mock_close)

    try:
        return agent.main()
    except Exception as e:
        print(e)
        return 1

agents = [NoActionAgent, RandomAgent1, RandomAgent2]

@pytest.mark.parametrize("agent", agents)
def test_basic_agents(agent, mocker):

    assert func(agent, mocker) == 0

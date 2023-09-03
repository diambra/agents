#!/usr/bin/env python3
import pytest
import sys
import os
from diambra.arena.utils.engine_mock import load_mocker

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

    load_mocker(mocker, override_perfect_probability=0.0)

    try:
        return agent.main(test=True)
    except Exception as e:
        print(e)
        return 1

agents = [NoActionAgent, RandomAgent1, RandomAgent2]

@pytest.mark.parametrize("agent", agents)
def test_basic_agents(agent, mocker):
    assert func(agent, mocker) == 0

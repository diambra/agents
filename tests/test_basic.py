#!/usr/bin/env python3
import pytest
import sys
import random
from os.path import expanduser
import os
from diambra.arena.utils.engine_mock import DiambraEngineMock, EngineMockParams

# Add the scripts directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from basic.no_action import agent as NoActionAgent
from basic.random_1 import agent as RandomAgent1
from basic.random_2 import agent as RandomAgent2

# Example Usage:
# pytest
# (optional)
#    module.py (Run specific module)
#    -s (show output)
#    -k 'expression' (filter tests using case-insensitive with parts of the test name and/or parameters values combined with boolean operators, e.g. 'wrappers and doapp')

def func(agent, mocker):

    round_winning_probability = 0.5
    perfect_probability=0.2

    rounds_per_stage = random.choice([2, 3])
    stages_per_game = random.choice([2, 3])
    number_of_chars = random.choice([11,33])
    number_of_chars_per_round = random.choice([1, 2])
    min_health = random.choice([-1, 0])
    max_health = random.choice([100, 208])
    frame_shape = random.choice([[128, 128, 3], [480, 512, 3]])
    n_actions = random.choice([[9, 7, 12], [9, 6, 6], [9, 6, 12]])

    diambra_engine_mock_params = EngineMockParams(round_winning_probability=round_winning_probability,
                                                  perfect_probability=perfect_probability, rounds_per_stage=rounds_per_stage,
                                                  stages_per_game=stages_per_game, number_of_chars=number_of_chars,
                                                  number_of_chars_per_round=number_of_chars_per_round,
                                                  min_health=min_health, max_health=max_health, frame_shape=frame_shape,
                                                  n_actions=n_actions)
    diambra_engine_mock = DiambraEngineMock(diambra_engine_mock_params)

    mocker.patch('diambra.arena.engine.interface.DiambraEngine.__init__', diambra_engine_mock._mock__init__)
    mocker.patch('diambra.arena.engine.interface.DiambraEngine._env_init', diambra_engine_mock._mock_env_init)
    mocker.patch('diambra.arena.engine.interface.DiambraEngine._reset', diambra_engine_mock._mock_reset)
    mocker.patch('diambra.arena.engine.interface.DiambraEngine._step_1p', diambra_engine_mock._mock_step_1p)
    mocker.patch('diambra.arena.engine.interface.DiambraEngine._step_2p', diambra_engine_mock._mock_step_2p)
    mocker.patch('diambra.arena.engine.interface.DiambraEngine.close', diambra_engine_mock._mock_close)

    try:
        return agent.main()
    except Exception as e:
        print(e)
        return 1

agents = [NoActionAgent, RandomAgent1, RandomAgent2]

@pytest.mark.parametrize("agent", agents)
def test_no_action_agent(agent, mocker):

    assert func(agent, mocker) == 0

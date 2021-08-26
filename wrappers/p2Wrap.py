from sbUtils import P2ToP1AddObsMove
import gym
import numpy as np

# Gym Env wrapper for two players mode to be used in integrated Self Play
class integratedSelfPlay(gym.Wrapper):
    def __init__(self, env):

        gym.Wrapper.__init__(self, env)

        # Modify action space
        assert self.action_space["P1"] == self.action_space["P2"],\
               "P1 and P2 action spaces are supposed to be identical: {} {}"\
                   .format(self.action_space["P1"], self.action_space["P2"])
        self.action_space = self.action_space["P1"]

    # Step the environment
    def step(self, action):

        return self.env.step(action)

    # Reset the environment
    def reset(self):

        return self.env.reset()

# Gym Env wrapper for two players mode with RL algo on P2
class selfPlayVsRL(gym.Wrapper):
    def __init__(self, env, p2Policy):

        gym.Wrapper.__init__(self, env)

        # Modify action space
        self.action_space = self.action_space["P1"]

        # P2 action logic
        self.p2Policy = p2Policy

    # Save last Observation
    def updateLastObs(self, obs):
        self.lastObs = obs

    # Update p2Policy RL policy weights
    def updateP2PolicyWeights(self, weightsPath):
        self.p2Policy.updateWeights(weightsPath)

    # Step the environment
    def step(self, action):

        # Observation modification and P2 actions selected by the model
        self.lastObs[:,:,-1] = P2ToP1AddObsMove(self.lastObs[:,:,-1])
        p2PolicyActions, _ = self.p2Policy.act(self.lastObs)

        obs, reward, done, info = self.env.step(np.hstack((action, p2PolicyActions)))
        self.updateLastObs(obs)

        return obs, reward, done, info

    # Reset the environment
    def reset(self):

        obs = self.env.reset()
        self.updateLastObs(obs)

        return obs

# Gym Env wrapper for two players mode with HUM+Gamepad on P2
class vsHum(gym.Wrapper):
    def __init__(self, env, p2Policy):

        gym.Wrapper.__init__(self, env)

        # Modify action space
        self.action_space = self.action_space["P1"]

        # P2 action logic
        self.p2Policy = p2Policy

        # If p2 action logic is gamepad, add it to self.gamepads (for char selection)
        # Check action space is prescribed as "multiDiscrete"
        self.p2Policy.initialize(self.env.actionList())
        if self.actionsSpace[1] != "multiDiscrete":
            raise Exception("Action Space for P2 must be \"multiDiscrete\" when using gamePad")
        if not self.attackButCombinations[1]:
            raise Exception("Use attack buttons combinations for P2 must be \"True\" when using gamePad")

    # Step the environment
    def step(self, action):

        # P2 actions selected by the Gamepad
        p2PolicyActions, _ = self.p2Policy.act()

        return self.env.step(np.hstack((action, p2PolicyActions)))

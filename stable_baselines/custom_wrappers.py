import gym
import numpy as np
import gymnasium

# Split actions in move and attack
class SplitActionsInMoveAndAttack(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Add to observations additional info
        :param env: (Gym Environment) the environment to wrap
        """
        gym.ObservationWrapper.__init__(self, env)

        assert self.unwrapped.env_settings.n_players == 1, "ERROR: \"SplitActionsInMoveAndAttack\" only works on single agent environment"
        assert isinstance(self.action_space, gym.spaces.Discrete), "ERROR: \"SplitActionsInMoveAndAttack\" only works on environments with a Discrete action space"

        new_observation_space = {}
        for k, v in self.observation_space.spaces.items():
            if k == "action":
                assert isinstance(v, gymnasium.spaces.MultiBinary), "ERROR: \"SplitActionsInMoveAndAttack\" only works on normalized environment via the dedicated wrapper"
                self.move_size = 9
                self.attack_size = self.action_space.n + 1 - self.move_size
                self.action_stack_size = int(v.n / self.action_space.n)
                self.move_vec_size = self.move_size * self.action_stack_size
                self.attack_vec_size = self.attack_size * self.action_stack_size
                new_observation_space["action_move"] = gymnasium.spaces.MultiBinary(self.move_vec_size)
                new_observation_space["action_attack"] = gymnasium.spaces.MultiBinary(self.attack_vec_size)
            else:
                new_observation_space[k] = v

        self.observation_space = gymnasium.spaces.Dict(new_observation_space)

    # Process observation
    def observation(self, obs):
        new_observation = {}
        for k, v in obs.items():
            if k == "action":
                move_actions_vector = np.zeros((self.move_vec_size), dtype=np.uint8)
                attack_actions_vector = np.zeros((self.attack_vec_size), dtype=np.uint8)
                actions_idx = np.where(obs["action"] == 1)[0]
                for action_idx in actions_idx:
                    stack_idx = int(action_idx / self.action_space.n)
                    relative_idx = action_idx - stack_idx * self.action_space.n
                    if relative_idx == 0:
                        move_actions_vector[stack_idx * self.move_size] = 1
                        attack_actions_vector[stack_idx * self.attack_size] = 1
                    elif relative_idx > self.move_size - 1:
                        attack_actions_vector[stack_idx * self.attack_size + relative_idx - self.move_size + 1] = 1
                        move_actions_vector[stack_idx * self.move_size] = 1
                    else:
                        move_actions_vector[stack_idx * self.move_size + relative_idx] = 1
                        attack_actions_vector[stack_idx * self.attack_size] = 1
                new_observation["action_move"] = move_actions_vector
                new_observation["action_attack"] = attack_actions_vector
            else:
                new_observation[k] = v

        return new_observation

# Convert additional obs to fifth observation channel for stable baselines
class RamStatesToChannel(gym.ObservationWrapper):
    def __init__(self, env, ram_states):
        """
        Add to observations additional info
        :param env: (Gym Environment) the environment to wrap
        :param ram_states: (list) ordered parameters for additional Obs
        """
        gym.ObservationWrapper.__init__(self, env)
        shp = self.env.observation_space["frame"].shape
        self.actions_in_ram_states = False
        if "action_move" in ram_states and "action_attack" in ram_states:
            self.actions_in_ram_states = True
            self.ram_states = [key for key in ram_states if (key != "action_move" and key != "action_attack")]

        self.box_high_bound = self.env.observation_space["frame"].high.max()
        self.box_low_bound = self.env.observation_space["frame"].low.min()
        assert (self.box_high_bound == 1.0 or self.box_high_bound == 255),\
               "Observation space max bound must be either 1.0 or 255 to use Additional Obs"
        assert (self.box_low_bound == 0.0 or self.box_low_bound == -1.0),\
               "Observation space min bound must be either 0.0 or -1.0 to use Additional Obs"

        self.old_obs_space = self.observation_space
        self.observation_space = gym.spaces.Box(low=self.box_low_bound, high=self.box_high_bound,
                                                shape=(shp[0], shp[1], shp[2] + 1), dtype=np.float32)
        self.shp = self.observation_space.shape

    # Process observation
    def observation(self, obs):
        # Adding a channel to the standard image, it will be in last position and
        # it will store additional obs
        shp = obs["frame"].shape
        obs_new = np.zeros((shp[0], shp[1], shp[2]+1), dtype=self.observation_space.dtype)

        # Storing standard image in the first channel leaving the last one for
        # additional obs
        obs_new[:, :, 0:shp[2]] = obs["frame"]

        # Adding new info to the additional channel, on a very
        # long line and then reshaping into the obs dim
        new_data = np.zeros((shp[0] * shp[1]))

        # Adding RAM states
        counter = 0
        if self.actions_in_ram_states is True:
            val = np.concatenate((obs["action_move"], obs["action_attack"]))
            for elem in val:
                counter = counter + 1
                new_data[counter] = elem

        for key in self.ram_states:
            if isinstance(obs[key], np.ndarray):
                for elem in obs[key]:
                    counter = counter + 1
                    new_data[counter] = elem
            else:
                counter = counter + 1
                new_data[counter] = obs[key]

        new_data[0] = counter
        new_data = np.reshape(new_data, (shp[0], -1))
        new_data = new_data * self.box_high_bound
        obs_new[:, :, shp[2]] = new_data

        return obs_new
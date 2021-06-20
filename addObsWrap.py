import gym
from gym import spaces
import numpy as np

# Observation modification (adding one channel to store additional info)
def processObs(obs, shp, dtype, boxHighBound, playerSide, keyToAdd, keysToDict):

    # Adding a channel to the standard image, it will be in last position and it will store additional obs
    obsNew = np.zeros((shp[0], shp[1], shp[2]), dtype=dtype)

    # Storing standard image in the first channel leaving the last one for additional obs
    obsNew[:,:,0:shp[2]-1] = obs["frame"]

    # Creating the additional channel where to store new info
    obsNewAdd = np.zeros((shp[0], shp[1], 1), dtype=dtype)

    # Adding new info to the additional channel, on a very
    # long line and then reshaping into the obs dim
    newData = np.zeros((shp[0] * shp[1]))

    # Adding new info for 1P
    counter = 0
    for key in keyToAdd:

        tmpList = keysToDict[key]
        if tmpList[0] == "Px":
            val = obs["P1"]

            for idx in range(len(tmpList)-1):

                if tmpList[idx+1] == "actionsBuf":
                    val = np.concatenate((val["actionsBuf"]["move"], val["actionsBuf"]["attack"]))
                elif "Char" in tmpList[idx+1]:
                    val = val[tmpList[idx+1]]
                else:
                    val = [val[tmpList[idx+1]]]
        else:
            val = [obs[tmpList[0]]]

        for elem in val:
            counter = counter + 1
            newData[counter] = elem

    newData[0] = counter

    # Adding new info for P2 in 2P games
    if playerSide == "P1P2":
        halfPosIdx = int((shp[0] * shp[1]) / 2)
        counter = halfPosIdx

        for key in keyToAdd:

            for addInfo in additionalInfo[key+"P2"]:

                counter = counter + 1
                newData[counter] = addInfo

        newData[halfPosIdx] = counter - halfPosIdx

    newData = np.reshape(newData, (shp[0], -1))

    newData = newData * boxHighBound

    obsNew[:,:,shp[2]-1] = newData

    return obsNew

# Convert additional obs to fifth observation channel for stable baselines
class AdditionalObsToChannel(gym.ObservationWrapper):
    def __init__(self, env, keyToAdd):
        """
        Add to observations additional info
        :param env: (Gym Environment) the environment to wrap
        :param keyToAdd: (list) ordered parameters for additional Obs
        """
        gym.ObservationWrapper.__init__(self, env)
        shp = self.env.observation_space["frame"].shape
        self.keyToAdd = keyToAdd

        self.boxHighBound = self.env.observation_space["frame"].high.max()
        self.boxLowBound = self.env.observation_space["frame"].low.min()
        assert (self.boxHighBound == 1.0 or self.boxHighBound == 255),\
                "Observation space max bound must be either 1.0 or 255 to use Additional Obs"
        assert (self.boxLowBound == 0.0 or self.boxLowBound == -1.0),\
                "Observation space min bound must be either 0.0 or -1.0 to use Additional Obs"

        # Loop among all keys
        self.keysToDict = {}
        for key in keyToAdd:
            elemToAdd = []
            # Loop among all spaces
            for k in env.observation_space.spaces:
                # Skip frame and consider only a single player
                if k == "frame" or k == "P2":
                    continue
                if type(env.observation_space[k]) == gym.spaces.dict.Dict:
                    for l in env.observation_space.spaces[k].spaces:
                        if type(env.observation_space[k][l]) == gym.spaces.dict.Dict:
                            if key == l:
                                elemToAdd.append("Px")
                                elemToAdd.append(l)
                                self.keysToDict[key] = elemToAdd
                        else:
                            if key == l:
                                elemToAdd.append("Px")
                                elemToAdd.append(l)
                                self.keysToDict[key] = elemToAdd
                else:
                    if key == k:
                        elemToAdd.append(k)
                        self.keysToDict[key] = elemToAdd


        self.observation_space = spaces.Box(low=self.boxLowBound, high=self.boxHighBound,
                                            shape=(shp[0], shp[1], shp[2] + 1),
                                            dtype=np.float32)
        self.shp = self.observation_space.shape

    # Process observation
    def observation(self, obs):

        return processObs(obs, self.shp, self.observation_space.dtype, self.boxHighBound,
                          self.env.playerSide, self.keyToAdd, self.keysToDict)

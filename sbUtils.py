from stable_baselines.common.callbacks import BaseCallback
import cv2, sys, os, time
import numpy as np

# Visualize Obs content
def showObs(observation, keyToAdd, keyToAddCount, actBufLen, nActions, waitKey, viz, charList, hardCore, idxList):

    if not hardCore:
        shp = observation.shape
        for idx in idxList:
            addPar = observation[:,:,shp[2]-1]
            addPar = np.reshape(addPar, (-1))

            counter = 0 + idx*int( (shp[0]*shp[1]) / 2 )

            print("Additional Par P{} =".format(idx+1), addPar[counter])

            counter += 1

            for idK in range(len(keyToAdd)):

                var = addPar[counter:counter+keyToAddCount[idK][idx]] if keyToAddCount[idK][idx] > 1 else addPar[counter]
                counter += keyToAddCount[idK][idx]

                if "actions" in keyToAdd[idK]:

                    moveActions   = var[0:actBufLen*nActions[idx][0]]
                    attackActions = var[actBufLen*nActions[idx][0]:actBufLen*(nActions[idx][0]+nActions[idx][1])]
                    moveActions   = np.reshape(moveActions, (actBufLen,-1))
                    attackActions = np.reshape(attackActions, (actBufLen,-1))
                    print("Move actions P{} =\n".format(idx+1), moveActions)
                    print("Attack actions P{} =\n ".format(idx+1), attackActions)
                elif "ownChar" in keyToAdd[idK] or "oppChar" in keyToAdd[idK]:
                    print("{}P{} =".format(keyToAdd[idK], idx+1), charList[list(var).index(1.0)])
                else:
                    print("{}P{} =".format(keyToAdd[idK], idx+1), var)

        if viz:
            obs = np.array(observation[:,:,0:shp[2]-1]).astype(np.float32)
    else:
        if viz:
            obs = np.array(observation).astype(np.float32)

    if viz:
        for idx in range(obs.shape[2]):
            cv2.imshow("image"+str(idx), obs[:,:,idx])

        cv2.waitKey(waitKey)

# Util to copy P2 additional OBS into P1 position on last (add info dedicated) channel
def P2ToP1AddObsMove(observation):
    shp = observation.shape
    startIdx = int((shp[0]*shp[1])/2)
    observation = np.reshape(observation, (-1))
    numAddParP2 = int(observation[startIdx])
    addParP2 = observation[startIdx:startIdx+numAddParP2 + 1]
    observation[0:numAddParP2 + 1] = addParP2
    observation = np.reshape(observation, (shp[0], -1))
    return observation

# Linear scheduler for RL agent parameters
def linear_schedule(initial_value, final_value = 0.0):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0 ), "linear_schedule work only with positive decreasing values"

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return final_value + progress * (initial_value - final_value)

    return func

# AutoSave Callback
class AutoSave(BaseCallback):
    """
    Callback for saving a model, it is saved every ``check_freq`` steps

    :param check_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, numEnv: int, save_path: str, verbose=1):
        super(AutoSave, self).__init__(verbose)
        self.check_freq = int(check_freq/numEnv)
        self.numEnv = numEnv
        self.save_path_base = save_path + 'autoSave_'

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print("Saving latest model to {}".format(self.save_path_base))
            # Save the agent
            self.model.save(self.save_path_base+str(self.n_calls*self.numEnv))

        return True

# Update p2Brain model Callback
class UpdateRLPolicyWeights(BaseCallback):
    def __init__(self, check_freq: int, numEnv: int, save_path: str,
                 prevAgentsSampling={"probability": 0.0, "list":[]}, verbose=1):
        super(UpdateRLPolicyWeights, self).__init__(verbose)
        self.check_freq = int(check_freq/numEnv)
        self.numEnv = numEnv
        self.save_path = save_path + 'lastModel'
        self.samplingProbability = prevAgentsSampling["probability"]
        self.prevAgentsList = prevAgentsSampling["list"]

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Selects if using previous agent or the last saved one
            if np.random.rand() < self.samplingProbability:
                # Sample an old model from the list
                if self.verbose > 0:
                    print("Using an older model")

                # Sample one of the older models
                idx = int(np.random.rand() * len(self.prevAgentsList))
                weightsPathsSampled = self.prevAgentsList[idx]

                # Load new weights
                self.training_env.env_method("updateP2PolicyWeights", weightsPath=weightsPathsSampled)
            else:
                # Use the last saved model
                if self.verbose > 0:
                    print("Using last saved model")

                if self.verbose > 0:
                    print("Saving latest model to {}".format(self.save_path))

                # Save the agent
                self.model.save(self.save_path)

                # Load new weights
                self.training_env.env_method("updateP2PolicyWeights", weightsPath=self.save_path)

        return True

# Abort training when run out of recorded trajectories for imitation learning
class ImitationLearningExhaustedExamples(BaseCallback):
    """
    Callback for aborting training when run out of Imitation Learning examples
    """
    def __init__(self):
        super(ImitationLearningExhaustedExamples, self).__init__()

    def _on_step(self) -> bool:

        return np.any(self.env.get_attr("exhausted"))

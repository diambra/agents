from stable_baselines.common.callbacks import BaseCallback
import numpy as np

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
    def __init__(self, check_freq: int, numEnv: int, verbose=1):
        super(UpdateRLPolicyWeights, self).__init__(verbose)
        self.check_freq = int(check_freq/numEnv)
        self.numEnv = numEnv

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Load new weights
            wPath = "/home/alexpalms/Work/ArtificialTwin/Diambra/diambraengine/" +\
                    "stableBaselines/diambraInterface/AIvsCOM/doapp_ppo2_Model_CustCnnSmall_bL_d_noComb/9M"
            self.training_env.env_method("updateP2BrainWeights", weightsPath=wPath)

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

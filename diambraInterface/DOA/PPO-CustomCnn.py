#!/usr/bin/env python
# coding: utf-8

gameFolder = "DOA++-MAME"

import sys, os
import time
timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

sys.path.append(os.path.join(os.path.abspath(''), '../'))
sys.path.append(os.path.join(os.path.abspath(''), '../../../games',gameFolder))

modelFolder = "ppo2_Model_CustCnn/"

os.makedirs(modelFolder, exist_ok=True)

from makeDiambraEnv import *

from Environment import setup_print_actions_dict
actionsPrintDict = setup_print_actions_dict()

import tensorflow as tf

from customPolicies.utils import linear_schedule, AutoSave
from customPolicies.customCnnPolicy import *

from stable_baselines import PPO2

diambraKwargs = {}
diambraKwargs["roms_path"] = "../../../roms/MAMEToolkit/roms/"
diambraKwargs["binary_path"] = "../../../../customMAME/"
diambraKwargs["player"] = "P1"
diambraKwargs["frame_ratio"] = 3
diambraKwargs["render"] = True
diambraKwargs["sound"] = True
#diambraKwargs["throttle"] = True
diambraKwargs["character"] = "Kasumi"

wrapperKwargs = {}
wrapperKwargs["hwc_obs_resize"] = [256, 256, 1]
wrapperKwargs["normalize_rewards"] = True
wrapperKwargs["clip_rewards"] = False
wrapperKwargs["frame_stack"] = 6
wrapperKwargs["scale"] = True
wrapperKwargs["scale_mod"] = 0

#keyToAdd = None
keyToAdd = []
keyToAdd.append("actionsBuf")
#keyToAdd.append("player")
keyToAdd.append("healthP1")
keyToAdd.append("healthP2")
keyToAdd.append("positionP1")
keyToAdd.append("positionP2")
#keyToAdd.append("winsP1")
#keyToAdd.append("winsP2")

numEnv=1

env = make_diambra_env(diambraMame, env_prefix="Eval", num_env=numEnv, seed=timeDepSeed,
                       continue_game=0, diambra_kwargs=diambraKwargs,
                       wrapper_kwargs=wrapperKwargs, key_to_add=keyToAdd)

# Load the trained agent
model = PPO2.load(modelFolder+"21_5M", env=env)

# Start
os.system("clear")
observation = env.reset()

cumulativeEpRew = 0.0
cumulativeEpRewAll = []
cumulativeTotRew = 0.0

maxNumEp = 10
currNumEp = 0

deterministicFlag = True

while currNumEp < maxNumEp:

    #action = model.predict(observation, deterministic=deterministicFlag)
    action_prob = model.action_probability(observation)
    print("Actions prob. = ", np.round(action_prob[0],2))

    if deterministicFlag:
       action = [[np.argmax(action_prob)]]
    else:
       raise "Not implemented"

    print(actionsPrintDict[action[0][0]])

    observation, reward, done, info = env.step(action[0])

    cumulativeEpRew += reward

    if np.any(done):
        currNumEp += 1
        print("Ep. # = ", currNumEp)
        print("Ep. Cumulative Rew # = ", cumulativeEpRew)
        sys.stdout.flush()
        cumulativeEpRewAll.append(cumulativeEpRew)
        cumulativeTotRew += cumulativeEpRew
        cumulativeEpRew = 0.0

print("Mean cumulative reward = ", cumulativeTotRew/maxNumEp)
print("Mean cumulative reward = ", np.mean(cumulativeEpRewAll))
print("Std cumulative reward = ", np.std(cumulativeEpRewAll))

env.close()

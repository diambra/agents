#!/usr/bin/env python
# coding: utf-8

gameFolder = "DOA++-MAME"

import sys, os
import time
timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

sys.path.append(os.path.join(os.path.abspath(''), '../'))
sys.path.append(os.path.join(os.path.abspath(''), '../../../games',gameFolder))

from makeDiambraEnv import *

from Environment import setup_print_actions_dict
actionsPrintDict = setup_print_actions_dict()

diambraKwargs = {}
diambraKwargs["roms_path"] = "../../../roms/MAMEToolkit/roms/"
diambraKwargs["binary_path"] = "../../../../customMAME/"
diambraKwargs["player"] = "P1"
diambraKwargs["frame_ratio"] = 3
diambraKwargs["render"] = True
diambraKwargs["sound"] = True
#diambraKwargs["throttle"] = True
diambraKwargs["character"] = "Kasumi"

numEnv=1

env = make_diambra_env(diambraMame, env_prefix="Eval", num_env=numEnv, seed=timeDepSeed,
                       continue_game=0, showFinal=True, diambra_kwargs=diambraKwargs)

# Start
os.system("clear")
observation = env.reset()

cumulativeEpRew = 0.0
cumulativeEpRewAll = []
cumulativeTotRew = 0.0

maxNumEp = 10
currNumEp = 0

while currNumEp < maxNumEp:

    action = [env.action_space.sample()]

    print("(Random)", actionsPrintDict[action[0]])

    observation, reward, done, info = env.step(action)

    cumulativeEpRew += reward

    if np.any(done):
        currNumEp += 1
        print("Ep. # = ", currNumEp)
        print("Ep. Cumulative Rew # = ", cumulativeEpRew)
        cumulativeEpRewAll.append(cumulativeEpRew)
        cumulativeTotRew += cumulativeEpRew
        cumulativeEpRew = 0.0

print("Mean cumulative reward = ", cumulativeTotRew/maxNumEp)
print("Mean cumulative reward = ", np.mean(cumulativeEpRewAll))
print("Std cumulative reward = ", np.std(cumulativeEpRewAll))

env.close()

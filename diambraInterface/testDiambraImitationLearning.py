import sys, os
from os.path import expanduser
from os import listdir
import time
import pickle, bz2
import numpy as np
import argparse

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',        type=str, default="none",     help='Path where recorded files are stored (none)')
    parser.add_argument('--nProc',       type=int, default=1,          help='Number of processors [(1), 2]')
    parser.add_argument('--actionSpace', type=str, default="discrete", help='Action space [(discrete), multiDiscrete]')
    opt = parser.parse_args()
    print(opt)

    homeDir = expanduser("~")
    base_path = os.path.dirname(__file__)

    sys.path.append(base_path)
    sys.path.append(os.path.join(base_path, '../../utils'))

    from diambraImitationLearning import *

    # Show files in folder
    if opt.path == "none":
        trajRecFolder = os.path.join(homeDir, "DIAMBRA/trajRecordings/doapp")
    else:
        trajRecFolder = opt.path

    trajectoriesFiles = [os.path.join(trajRecFolder, f) for f in listdir(trajRecFolder) if os.path.isfile(os.path.join(trajRecFolder, f))]
    print(trajectoriesFiles)

    diambraILKwargs = {}
    diambraILKwargs["hwcDim"] = [128,128,5]
    diambraILKwargs["actionSpace"] = opt.actionSpace
    diambraILKwargs["nActions"] = [9, 8]
    diambraILKwargs["trajFilesList"] = trajectoriesFiles# P1
    diambraILKwargs["totalCpus"] = opt.nProc

    env = makeDiambraImitationLearningEnv(diambraImitationLearning, diambraILKwargs)

    observation = env.reset()[0]
    env.render(mode="human")

    env.env_method("trajSummary")

    nChars = env.get_attr("nChars")[0]
    charNames = env.get_attr("charNames")[0]
    n_actions = env.get_attr("nActions")[0]
    actBufLen = env.get_attr("actBufLen")[0]
    playerId = env.get_attr("playerId")[0]

    limAct = [None, None]
    for idx in range(2):
        limAct[idx] = [actBufLen * n_actions[0],
                       actBufLen * n_actions[0] + actBufLen * n_actions[1]]

    # Visualize Obs content
    def observationViz(observation, limAct):

       shp = observation.shape
       additionalPar = int(observation[0,0,shp[2]-1])

       # 1P
       nScalarAddPar = additionalPar - nChars - actBufLen*(n_actions[0]+n_actions[1])

       print("Additional Par = ", additionalPar)
       print("N scalar actions = ", nScalarAddPar)

       addPar = observation[:,:,shp[2]-1]
       addPar = np.reshape(addPar, (-1))
       addPar = addPar[1:additionalPar+1]
       actions = addPar[0:additionalPar-nScalarAddPar-nChars]

       moveActionsP1   = actions[0:limAct[0][0]]
       attackActionsP1 = actions[limAct[0][0]:limAct[0][1]]
       moveActionsP1   = np.reshape(moveActionsP1, (actBufLen,-1))
       attackActionsP1 = np.reshape(attackActionsP1, (actBufLen,-1))
       print("Move actions P1 =\n", moveActionsP1)
       print("Attack actions P1 =\n ", attackActionsP1)

       others = addPar[additionalPar-nScalarAddPar-nChars:]
       print("ownHealth = ", others[0])
       print("oppHealth = ", others[1])
       print("ownPosition = ", others[2])
       print("oppPosition = ", others[3])
       print("stage = ", others[4])
       print("Playing Char  = ", charNames[list(others[nScalarAddPar:
                                                       nScalarAddPar + nChars]).index(1.0)])

       obs = np.array(observation).astype(np.float32)

    cumulativeEpRew = 0.0
    cumulativeEpRewAll = []

    maxNumEp = 10
    currNumEp = 0

    procIdx = 0

    while currNumEp < maxNumEp:

        dummyActions = [0 for i in range(diambraILKwargs["totalCpus"])]
        observation, reward, done, info = env.step(dummyActions)
        env.render(mode="human")

        observation = observation[procIdx]
        reward = reward[procIdx]
        done = done[procIdx]
        action = info[procIdx]["action"]
        print("Reward = ", reward)
        if done:
            observation = info[procIdx]["terminal_observation"]

        # Visualize observations content
        observationViz(observation, limAct) # Keep space bar pressed to continue env execution

        cumulativeEpRew += reward

        if done:
            currNumEp += 1
            print("Ep. # = ", currNumEp)
            print("Ep. Cumulative Rew # = ", cumulativeEpRew)

            cumulativeEpRewAll.append(cumulativeEpRew)
            cumulativeEpRew = 0.0

        if np.any(env.get_attr("exhausted")):
            break

    if diambraILKwargs["totalCpus"] == 1:
        print("All ep. rewards =", cumulativeEpRewAll)
        print("Mean cumulative reward =", np.mean(cumulativeEpRewAll))
        print("Std cumulative reward =", np.std(cumulativeEpRewAll))

    print("ALL GOOD!")
except Exception as e:
    print(e)
    print("ALL BAD")

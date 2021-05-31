import sys, os
from os.path import expanduser
import time
import numpy as np
import argparse

def reject_outliers(data):
    m = 2
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return filtered

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gameId',      type=str, default="doapp",    help='Game ID [(doapp), sfiii3n, tektagt, umk3]')
    parser.add_argument('--player',      type=str, default="Random",   help='Player [(Random), P1, P2, P1P2]')
    parser.add_argument('--frameRatio',  type=int, default=1,          help='Frame ratio')
    parser.add_argument('--nEpisodes',   type=int, default=1,          help='Number of episodes')
    parser.add_argument('--actionSpace', type=str, default="discrete", help='(discrete)/multidiscrete')
    parser.add_argument('--attButComb',  type=int, default=0,          help='If to use attack button combinations (0=False)/1=True')
    parser.add_argument('--targetSpeed', type=int, default=100,        help='Number of episodes')
    opt = parser.parse_args()
    print(opt)

    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    base_path = os.path.dirname(__file__)

    sys.path.append(base_path)

    from diambraMameGym import diambraMame
    from makeDiambraEnv import makeDiambraEnv

    # Common settings
    diambraKwargs = {}
    diambraKwargs["romsPath"] = os.path.join(base_path, "../../roms/mame/")
    diambraKwargs["binaryPath"] = os.path.join(base_path, "../../customMAME/")
    diambraKwargs["frameRatio"] = opt.frameRatio
    diambraKwargs["throttle"] = False
    diambraKwargs["render"] = False
    diambraKwargs["sound"] = diambraKwargs["throttle"]

    diambraKwargs["player"] = opt.player

    #keyToAdd = None
    keyToAdd = []
    keyToAdd.append("actionsBuf")

    if opt.gameId != "tektagt": # DOA, SFIII, UMK3
        keyToAdd.append("ownHealth")
        keyToAdd.append("oppHealth")
    else: # TEKTAG
        keyToAdd.append("ownHealth_1")
        keyToAdd.append("ownHealth_2")
        keyToAdd.append("oppHealth_1")
        keyToAdd.append("oppHealth_2")
        keyToAdd.append("ownActiveChar")
        keyToAdd.append("oppActiveChar")

    keyToAdd.append("ownPosition")
    keyToAdd.append("oppPosition")
    #keyToAdd.append("ownWins")
    #keyToAdd.append("oppWins")
    keyToAdd.append("stage")
    keyToAdd.append("character")

    if opt.gameId != "tektagt":
        diambraKwargs["characters"] = ["Random", "Random"]
    else:
        diambraKwargs["characters"] = [["Random", "Random"], ["Random", "Random"]]
    diambraKwargs["charOutfits"] = [2, 2]

    # DIAMBRA gym kwargs
    diambraGymKwargs = {}
    diambraGymKwargs["P2brain"] = None
    diambraGymKwargs["continueGame"] = 0.0
    diambraGymKwargs["showFinal"] = False
    diambraGymKwargs["actionSpace"] = [opt.actionSpace, opt.actionSpace]
    diambraGymKwargs["attackButCombinations"] = [opt.attButComb, opt.attButComb]
    diambraGymKwargs["actBufLen"] = 12

    # Recording kwargs
    trajRecKwargs = None

    wrapperKwargs = {}
    wrapperKwargs["hwcObsResize"] = [256, 256, 1]
    wrapperKwargs["normalizeRewards"] = True
    wrapperKwargs["clipRewards"] = False
    wrapperKwargs["frameStack"] = 4
    wrapperKwargs["dilation"] = 1
    wrapperKwargs["scale"] = True
    wrapperKwargs["scaleMod"] = 0

    envId = opt.gameId + "_Test"
    env = makeDiambraEnv(diambraMame, envPrefix=envId, seed=timeDepSeed,
                         diambraKwargs=diambraKwargs, diambraGymKwargs=diambraGymKwargs,
                         wrapperKwargs=wrapperKwargs, keyToAdd=keyToAdd,
                         trajRecKwargs=trajRecKwargs)

    observation = env.reset()

    shp = observation.shape

    print("Number of characters =", len(env.charNames))

    additionalParP1 = int(observation[0,0,shp[2]-1])
    print("Additional Parameters P1 =", additionalParP1)

    nScalarAddParP1 = additionalParP1 - len(env.charNames)\
                    - env.actBufLen*(env.nActions[0][0]+env.nActions[0][1]) # 1P
    print("Number of scalar Parameters P1 =", nScalarAddParP1)


    if diambraKwargs["player"] == "P1P2":
        additionalParP2 = int(observation[int(shp[0]/2),0,shp[2]-1])
        print("Additional Parameters P2 =", additionalParP2)

        nScalarAddParP2 = additionalParP2 - len(env.charNames)\
                        - env.actBufLen*(env.nActions[1][0]+env.nActions[1][1])# 2P
        print("Number of scalar Parameters P2 =", nScalarAddParP2)

    limAct = [None, None]
    for idx in range(2):
        limAct[idx] = [env.actBufLen * env.nActions[idx][0],
                       env.actBufLen * env.nActions[idx][0] + env.actBufLen * env.nActions[idx][1]]

    maxNumEp = opt.nEpisodes
    currNumEp = 0

    tic = time.time()
    fpsVal = []

    while currNumEp < maxNumEp:

        toc = time.time()
        fps = 1/(toc - tic)
        tic = toc
        #print("FPS = {}".format(fps))
        fpsVal.append(fps)

        # 1P
        action = env.actionSpaces[0].sample()

        # 2P
        action2 = env.actionSpaces[1].sample()
        if diambraKwargs["player"] == "P1P2":
            action = np.append(action, action2)

        observation, reward, done, info = env.step(action)

        if np.any(done):
            currNumEp += 1
            print("Ep. # = ", currNumEp)

            observation = env.reset()

    env.close()

    fpsVal2 = reject_outliers(fpsVal)
    avgFps = np.mean(fpsVal2)
    print("Average speed = {} FPS, STD {} FPS", avgFps, np.std(fpsVal2))

    if abs(avgFps - opt.targetSpeed) > opt.targetSpeed*0.025:
        raise RuntimeError("Fps different than expected: {} VS {}".format(avgFps, opt.targetSpeed))

    print("ALL GOOD!")
except Exception as e:
    print(e)
    print("ALL BAD")

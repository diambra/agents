import sys, os
from os.path import expanduser
import time
import numpy as np
import argparse

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gameId',       type=str,   default="doapp",    help='Game ID [(doapp), sfiii3n, tektagt, umk3]')
    parser.add_argument('--player',       type=str,   default="Random",   help='Player [(Random), P1, P2, P1P2]')
    parser.add_argument('--character1',   type=str,   default="Random",   help='Character P1 (Random)')
    parser.add_argument('--character2',   type=str,   default="Random",   help='Character P2 (Random)')
    parser.add_argument('--character1_2', type=str,   default="Random",   help='Character P1_2 (Random)')
    parser.add_argument('--character2_2', type=str,   default="Random",   help='Character P2_2 (Random)')
    parser.add_argument('--frameRatio',   type=int,   default=3,          help='Frame ratio')
    parser.add_argument('--nEpisodes',    type=int,   default=1,          help='Number of episodes')
    parser.add_argument('--continueGame', type=float, default=-1.0,       help='ContinueGame flag (-inf,+1.0]')
    parser.add_argument('--actionSpace',  type=str,   default="discrete", help='(discrete)/multidiscrete')
    parser.add_argument('--attButComb',   type=int,   default=0,          help='If to use attack button combinations (0=False)/1=True')
    parser.add_argument('--noAction',     type=int,   default=0,          help='If to use no action policy (0=False)')
    parser.add_argument('--recordTraj',   type=int,   default=0,          help='If to record trajectories (0=False)')
    opt = parser.parse_args()
    print(opt)

    homeDir = expanduser("~")

    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    base_path = os.path.dirname(__file__)

    sys.path.append(base_path)
    sys.path.append(os.path.join(base_path, '../../utils'))
    sys.path.append(os.path.join(base_path, '../../pythonGamePadInterface'))

    from diambraGamepad import diambraGamepad
    from policies import gamepadPolicy, RLPolicy # To train AI against another AI or HUM

    from diambraMameGym import diambraMame
    from makeDiambraEnv import makeDiambraEnv

    # Common settings
    diambraKwargs = {}
    diambraKwargs["romsPath"] = os.path.join(base_path, "../../roms/mame/")
    diambraKwargs["binaryPath"] = os.path.join(base_path, "../../customMAME/")
    diambraKwargs["frameRatio"] = opt.frameRatio
    diambraKwargs["throttle"] = False
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
        diambraKwargs["characters"] = [opt.character1, opt.character2]
    else:
        diambraKwargs["characters"] = [[opt.character1, opt.character1_2], [opt.character2, opt.character2_2]]
    diambraKwargs["charOutfits"] = [2, 2]

    # GamePad policy initialization
    gamePad_policy = gamepadPolicy(diambraGamepad)

    # DIAMBRA gym kwargs
    diambraGymKwargs = {}
    diambraGymKwargs["P2brain"] = None#gamePad_policy
    diambraGymKwargs["continueGame"] = opt.continueGame
    diambraGymKwargs["showFinal"] = False
    diambraGymKwargs["actionSpace"] = [opt.actionSpace, opt.actionSpace]
    diambraGymKwargs["attackButCombinations"] = [opt.attButComb, opt.attButComb]
    diambraGymKwargs["actBufLen"] = 12

    # Recording kwargs
    trajRecKwargs = {}
    trajRecKwargs["userName"] = "Alex"
    trajRecKwargs["filePath"] = os.path.join( homeDir, "DIAMBRA/trajRecordings", opt.gameId)
    trajRecKwargs["ignoreP2"] = 0
    trajRecKwargs["commitHash"] = "0000000"

    if opt.recordTraj == 0:
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

    print("Obs space =", env.observation_space)
    print("Obs space type =", env.observation_space.dtype)
    print("Obs space high bound =", env.observation_space.high)
    print("Obs space low bound =", env.observation_space.low)

    # Printing action spaces
    for idx in range(2):

        if diambraKwargs["player"] != "P1P2" and idx == 1:
            continue

        print("Action space P{} = ".format(idx+1), env.actionSpaces[idx])
        print("Action space type P{} = ".format(idx+1), env.actionSpaces[idx].dtype)
        if diambraGymKwargs["actionSpace"][idx] == "multiDiscrete":
            print("Action space n = ", env.actionSpaces[idx].nvec)
        else:
            print("Action space n = ", env.actionSpaces[idx].n)

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

    cumulativeEpRew = 0.0
    cumulativeEpRewAll = []

    maxNumEp = opt.nEpisodes
    currNumEp = 0

    while currNumEp < maxNumEp:

        # 1P
        action = env.actionSpaces[0].sample()
        if opt.noAction == 1:
            if diambraGymKwargs["actionSpace"][idx] == "multiDiscrete":
                action = [0, 0]
            else:
                action = 0

        # 2P
        action2 = env.actionSpaces[1].sample()
        if diambraKwargs["player"] == "P1P2":
            action = np.append(action, action2)

        #action = int(input("Action"))
        print("Action:", action)
        observation, reward, done, info = env.step(action)

        addPar = observation[:,:,shp[2]-1]
        addPar = np.reshape(addPar, (-1))

        # P1
        addParP1 = addPar[1:additionalParP1+1]
        actionsP1 = addParP1[0:additionalParP1-nScalarAddParP1-env.numberOfCharacters]

        moveActionsP1   = actionsP1[0:limAct[0][0]]
        attackActionsP1 = actionsP1[limAct[0][0]:limAct[0][1]]
        moveActionsP1   = np.reshape(moveActionsP1, (env.actBufLen,-1))
        attackActionsP1 = np.reshape(attackActionsP1, (env.actBufLen,-1))
        print("Move actions P1 =\n", moveActionsP1)
        print("Attack actions P1 =\n ", attackActionsP1)

        othersP1 = addParP1[additionalParP1-nScalarAddParP1-env.numberOfCharacters:]
        if opt.gameId != "tektagt":
            print("ownHealthP1 = ", othersP1[0])
            print("oppHealthP1 = ", othersP1[1])
            print("ownPositionP1 = ", othersP1[2])
            print("oppPositionP1 = ", othersP1[3])
            print("stageP1 = ", othersP1[4])
        else:
            print("ownHealth_1P1 = ", othersP1[0])
            print("ownHealth_2P1 = ", othersP1[1])
            print("oppHealth_1P1 = ", othersP1[2])
            print("oppHealth_2P1 = ", othersP1[3])
            print("ownActiveCharP1 = ", othersP1[4])
            print("oppActiveCharP1 = ", othersP1[5])
            print("ownPositionP1 = ", othersP1[6])
            print("oppPositionP1 = ", othersP1[7])
            print("stageP1 = ", othersP1[8])
        print("Playing Char P1 = ", env.charNames[list(othersP1[nScalarAddParP1:
                                                                nScalarAddParP1 + env.numberOfCharacters]).index(1.0)])

        # 2P
        if diambraKwargs["player"] == "P1P2":
            addParP2 = addPar[int((shp[0]*shp[1])/2)+1:int((shp[0]*shp[1])/2)+additionalParP2+1]
            actionsP2 = addParP2[0:additionalParP2-nScalarAddParP2-env.numberOfCharacters]

            moveActionsP2   = actionsP2[0:limAct[1][0]]
            attackActionsP2 = actionsP2[limAct[1][0]:limAct[1][1]]
            moveActionsP2   = np.reshape(moveActionsP2, (env.actBufLen,-1))
            attackActionsP2 = np.reshape(attackActionsP2, (env.actBufLen,-1))
            print("Move actions P2 =\n", moveActionsP2)
            print("Attack actions P2 =\n", attackActionsP2)

            othersP2 = addParP2[additionalParP2-nScalarAddParP2-env.numberOfCharacters:]
            if opt.gameId != "tektagt":
                print("ownHealthP2 = ", othersP2[0])
                print("oppHealthP2 = ", othersP2[1])
                print("ownPositionP2 = ", othersP2[2])
                print("oppPositionP2 = ", othersP2[3])
                print("stageP2 = ", othersP2[4])
            else:
                print("ownHealth_1P2 = ", othersP2[0])
                print("ownHealth_2P2 = ", othersP2[1])
                print("oppHealth_1P2 = ", othersP2[2])
                print("oppHealth_2P2 = ", othersP2[3])
                print("ownActiveCharP2 = ", othersP2[4])
                print("oppActiveCharP2 = ", othersP2[5])
                print("ownPositionP2 = ", othersP2[6])
                print("oppPositionP2 = ", othersP2[7])
                print("stageP2 = ", othersP2[8])
            print("Playing Char P2 = ", env.charNames[list(othersP2[nScalarAddParP2:
                                                                    nScalarAddParP2 + env.numberOfCharacters]).index(1.0)])

        print("Frames shape:", observation.shape)
        print("Reward:", reward)
        print("Actions Buffer P1 = ", info["actionsBufP1"])
        if diambraKwargs["player"] == "P1P2":
            print("Actions Buffer P2 = ", info["actionsBufP2"])
        print("Fighting = ", info["fighting"])
        print("Rewards = ", info["rewards"])
        if opt.gameId != "tektagt":
            print("HealthP1 = ", info["healthP1"])
            print("HealthP2 = ", info["healthP2"])
        else:
            print("HealthP1_1 = ", info["healthP1_1"])
            print("HealthP1_2 = ", info["healthP1_2"])
            print("HealthP2_1 = ", info["healthP2_1"])
            print("HealthP2_2 = ", info["healthP2_2"])
            print("ActiveCharP1 = ", info["activeCharP1"])
            print("ActiveCharP2 = ", info["activeCharP2"])

        print("PositionP1 = ", info["positionP1"])
        print("PositionP2 = ", info["positionP2"])
        print("WinP1 = ", info["winsP1"])
        print("WinP2 = ", info["winsP2"])
        print("Stage = ", info["stage"])
        print("Round done = ", info["roundDone"])
        print("Stage done = ", info["stageDone"])
        print("Game done = ", info["gameDone"])
        print("Episode done = ", info["episodeDone"])

        cumulativeEpRew += reward

        if np.any(done):
            currNumEp += 1
            print("Ep. # = ", currNumEp)
            print("Ep. Cumulative Rew # = ", cumulativeEpRew)
            cumulativeEpRewAll.append(cumulativeEpRew)
            cumulativeEpRew = 0.0

            observation = env.reset()

    print("Mean cumulative reward = ", np.mean(cumulativeEpRewAll))
    print("Std cumulative reward = ", np.std(cumulativeEpRewAll))

    env.close()

    if len(cumulativeEpRewAll) != maxNumEp:
        raise RuntimeError("Not run all episodes")

    if opt.continueGame <= 0.0:
        maxContinue = int(-opt.continueGame)
    else:
        maxContinue = 0

    if opt.gameId == "tektagt":
        maxContinue = (maxContinue + 1) * 0.7 - 1

    if opt.noAction == 1 and np.mean(cumulativeEpRewAll) > -(maxContinue+1)*3.999:
        raise RuntimeError("NoAction policy and average reward different than {} ({})".format(-(maxContinue+1)*4, np.mean(cumulativeEpRewAll)))

    print("ALL GOOD!")
except Exception as e:
    print(e)
    print("ALL BAD")

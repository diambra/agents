import cv2, sys, os, time
from os.path import expanduser
import numpy as np
import argparse
base_path = os.path.dirname(__file__)
sys.path.append(os.path.join(base_path, '../.'))
sys.path.append(os.path.join(base_path, '../../gym/'))

from utils import discreteToMultiDiscreteAction
from makeStableBaselinesEnv import makeStableBaselinesEnv

# Visualize Obs content
def showObs(observation, gameId, actBufLen, waitKey, viz, charList, limAct, hardCore, idxList):

    if not hardCore:
        shp = observation.shape
        nChars = len(charList)

        for idx in idxList:
            additionalPar = int(observation[0+idx*int(shp[0]/2),0,shp[2]-1])

            nScalarAddPar = additionalPar - 2*nChars - limAct[idx][1]

            print("Additional Par P{} =".format(idx+1), additionalPar)
            print("N scalar actions P{} =".format(idx+1), nScalarAddPar)

            addPar = observation[:,:,shp[2]-1]
            addPar = np.reshape(addPar, (-1))
            addPar = addPar[1+idx*int((shp[0]*shp[1])/2):additionalPar+1+idx*int((shp[0]*shp[1])/2)]
            actions = addPar[0:additionalPar-nScalarAddPar-2*nChars]

            moveActions   = actions[0:limAct[idx][0]]
            attackActions = actions[limAct[idx][0]:limAct[idx][1]]
            moveActions   = np.reshape(moveActions, (actBufLen,-1))
            attackActions = np.reshape(attackActions, (actBufLen,-1))
            print("Move actions P{} =\n".format(idx+1), moveActions)
            print("Attack actions P{} =\n ".format(idx+1), attackActions)

            others = addPar[additionalPar-nScalarAddPar-2*nChars:]
            if gameId != "tektagt":
                print("ownHealthP{} =".format(idx+1), others[0])
                print("oppHealthP{} =".format(idx+1), others[1])
                print("ownPositionP{} =".format(idx+1), others[2])
                print("oppPositionP{} =".format(idx+1), others[3])
                if nScalarAddPar == 5:
                    print("stage =", others[4])
            else:
                print("ownHealth1P{} =".format(idx+1), others[0])
                print("ownHealth2P{} =".format(idx+1), others[1])
                print("oppHealth1P{} =".format(idx+1), others[2])
                print("oppHealth2P{} =".format(idx+1), others[3])
                print("ownActiveCharP{} =".format(idx+1), others[4])
                print("oppActiveCharP{} =".format(idx+1), others[5])
                print("ownPositionP{} =".format(idx+1), others[6])
                print("oppPositionP{} =".format(idx+1), others[7])
                if nScalarAddPar == 9:
                    print("stage =".format(idx+1), others[8])
            print("ownCharP{} =".format(idx+1), charList[list(others[nScalarAddPar:
                                                                     nScalarAddPar + nChars]).index(1.0)])
            print("oppCharP{} =".format(idx+1), charList[list(others[nScalarAddPar + nChars:
                                                                     nScalarAddPar + 2*nChars]).index(1.0)])

        if viz:
            obs = np.array(observation[:,:,0:shp[2]-1]).astype(np.float32)
    else:
        if viz:
            obs = np.array(observation).astype(np.float32)

    if viz:
        for idx in range(obs.shape[2]):
            cv2.imshow("image"+str(idx), obs[:,:,idx])

        cv2.waitKey(waitKey)

if __name__ == '__main__':
    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

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
        parser.add_argument('--hardCore',     type=int,   default=0,          help='Hard core mode (0=False)')
        opt = parser.parse_args()
        print(opt)

        homeDir = expanduser("~")

        # Common settings
        diambraKwargs = {}
        diambraKwargs["romsPath"] = os.path.join(base_path, "../../roms/mame/")
        diambraKwargs["binaryPath"] = os.path.join(base_path, "../../customMAME/")
        diambraKwargs["frameRatio"] = opt.frameRatio
        diambraKwargs["throttle"] = False
        diambraKwargs["sound"] = diambraKwargs["throttle"]

        diambraKwargs["player"] = opt.player

        if opt.gameId != "tektagt":
            diambraKwargs["characters"] = [opt.character1, opt.character2]
        else:
            diambraKwargs["characters"] = [[opt.character1, opt.character1_2], [opt.character2, opt.character2_2]]
        diambraKwargs["charOutfits"] = [2, 2]

        # DIAMBRA gym kwargs
        diambraGymKwargs = {}
        diambraGymKwargs["actionSpace"] = [opt.actionSpace, opt.actionSpace]
        diambraGymKwargs["attackButCombinations"] = [opt.attButComb, opt.attButComb]
        diambraGymKwargs["actBufLen"] = 12
        if diambraKwargs["player"] != "P1P2":
            diambraGymKwargs["showFinal"] = False
            diambraGymKwargs["continueGame"] = opt.continueGame
            diambraGymKwargs["actionSpace"] = diambraGymKwargs["actionSpace"][0]
            diambraGymKwargs["attackButCombinations"] = diambraGymKwargs["attackButCombinations"][0]

        idxList = [0, 1]
        if diambraKwargs["player"] != "P1P2":
            idxList = [0]

        # Recording kwargs
        trajRecKwargs = {}
        trajRecKwargs["userName"] = "Alex"
        trajRecKwargs["filePath"] = os.path.join( homeDir, "DIAMBRA/trajRecordings", opt.gameId)
        trajRecKwargs["ignoreP2"] = 0
        trajRecKwargs["commitHash"] = "0000000"

        if opt.recordTraj == 0:
            trajRecKwargs = None

        # Env wrappers kwargs
        wrapperKwargs = {}
        wrapperKwargs["noOpMax"] = 0
        wrapperKwargs["hwcObsResize"] = [128, 128, 1]
        wrapperKwargs["normalizeRewards"] = True
        wrapperKwargs["clipRewards"] = False
        wrapperKwargs["frameStack"] = 4
        wrapperKwargs["dilation"] = 1
        wrapperKwargs["scale"] = True
        wrapperKwargs["scaleMod"] = 0

        # Additional obs key list
        keyToAdd = []
        keyToAdd.append("actionsBuf") # env.actBufLen*(env.n_actions[0]+env.n_actions[1])

        if gameId != "tektagt":
            keyToAdd.append("ownHealth")   # 1
            keyToAdd.append("oppHealth")   # 1
        else:
            keyToAdd.append("ownHealth1") # 1
            keyToAdd.append("ownHealth2") # 1
            keyToAdd.append("oppHealth1") # 1
            keyToAdd.append("oppHealth2") # 1
            keyToAdd.append("ownActiveChar") # 1
            keyToAdd.append("oppActiveChar") # 1

        keyToAdd.append("ownPosition")     # 1
        keyToAdd.append("oppPosition")     # 1
        if diambraKwargs["player"] != "P1P2":
            keyToAdd.append("stage")           # 1
        keyToAdd.append("ownChar")       # len(env.charNames)
        keyToAdd.append("oppChar")       # len(env.charNames)

        envId = opt.gameId + "_randomTestSBWrappers"
        hardCore = False if opt.hardCore == 0 else True
        numOfEnvs = 1
        env = makeStableBaselinesEnv(envId, numOfEnvs, timeDepSeed, diambraKwargs, diambraGymKwargs,
                                     wrapperKwargs, trajRecKwargs, keyToAdd=keyToAdd, noVec=True, hardCore=hardCore)

        print("Observation Space:", env.observation_space)
        print("Action Space:", env.action_space)
        if not hardCore:
            print("Keys to Dict:", env.keysToDict)

        limAct = [None, None]
        for idx in range(2):
            if diambraKwargs["player"] != "P1P2":
                limAct[idx] = [env.actBufLen * env.nActions[0],
                               env.actBufLen * env.nActions[0] + env.actBufLen * env.nActions[1]]
            else:
                limAct[idx] = [env.actBufLen * env.nActions[idx][0],
                               env.actBufLen * env.nActions[idx][0] + env.actBufLen * env.nActions[idx][1]]

        actionsPrintDict = env.printActionsDict()
        observation = env.reset()

        cumulativeEpRew = 0.0
        cumulativeEpRewAll = []

        maxNumEp = opt.nEpisodes
        currNumEp = 0

        while currNumEp < maxNumEp:

            actions = [None, None]
            if diambraKwargs["player"] != "P1P2":
                actions = env.action_space.sample()

                if opt.noAction == 1:
                    if diambraGymKwargs["actionSpace"] == "multiDiscrete":
                        for iEl, _ in enumerate(actions):
                            actions[iEl] = 0
                    else:
                        actions = 0

                if diambraGymKwargs["actionSpace"] == "discrete":
                    moveAction, attAction = discreteToMultiDiscreteAction(actions, env.nActions[0])
                else:
                    moveAction, attAction = actions[0], actions[1]

                print("(P1) {} {}".format(actionsPrintDict[0][moveAction],
                                          actionsPrintDict[1][attAction]))

            else:
                for idx in range(2):
                    actions[idx] = env.action_space["P{}".format(idx+1)].sample()

                    if opt.noAction == 1 and idx == 0:
                        if diambraGymKwargs["actionSpace"][idx] == "multiDiscrete":
                            for iEl, _ in enumerate(actions[idx]):
                                actions[idx][iEl] = 0
                        else:
                            actions[idx] = 0

                    if diambraGymKwargs["actionSpace"][idx] == "discrete":
                        moveAction, attAction = discreteToMultiDiscreteAction(actions[idx], env.nActions[idx][0])
                    else:
                        moveAction, attAction = actions[idx][0], actions[idx][1]

                    print("(P{}) {} {}".format(idx+1, actionsPrintDict[0][moveAction],
                                                      actionsPrintDict[1][attAction]))

            if diambraKwargs["player"] == "P1P2" or diambraGymKwargs["actionSpace"] != "discrete":
                actions = np.append(actions[0], actions[1])

            observation, reward, done, info = env.step(actions)

            cumulativeEpRew += reward
            print("action =", actions)
            print("reward =", reward)
            print("done =", done)
            for k, v in info.items():
                print("info[\"{}\"] = {}".format(k, v))
            showObs(observation, gameId, env.actBufLen, 1, False, env.charNames, limAct, hardCore, idxList)
            print("--")
            print("Current Cumulative Reward =", cumulativeEpRew)

            print("----------")


            if np.any(done):
                print("Resetting Env")
                currNumEp += 1
                print("Ep. # = ", currNumEp)
                print("Ep. Cumulative Rew # = ", cumulativeEpRew)
                cumulativeEpRewAll.append(cumulativeEpRew)
                cumulativeEpRew = 0.0

                observation = env.reset()
                showObs(observation, gameId, env.actBufLen, 1, False, env.charNames, limAct, hardCore, idxList)

        print("Cumulative reward = ", cumulativeEpRewAll)
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

import sys, os, time
import numpy as np
import argparse
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))
sys.path.append(os.path.join(base_path, '../../games_cpp/gym/'))

from gymUtils import discreteToMultiDiscreteAction
from sbUtils import showObs
from makeStableBaselinesEnv import makeStableBaselinesEnv

if __name__ == '__main__':
    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--gameId',         type=str,   default="doapp",    help='Game ID')
        parser.add_argument('--player',         type=str,   default="Random",   help='Player [(Random), P1, P2, P1P2]')
        parser.add_argument('--character1',     type=str,   default="Random",   help='Character P1 (Random)')
        parser.add_argument('--character2',     type=str,   default="Random",   help='Character P2 (Random)')
        parser.add_argument('--character1_2',   type=str,   default="Random",   help='Character P1_2 (Random)')
        parser.add_argument('--character2_2',   type=str,   default="Random",   help='Character P2_2 (Random)')
        parser.add_argument('--frameRatio',     type=int,   default=6,          help='Frame ratio')
        parser.add_argument('--nEpisodes',      type=int,   default=1,          help='Number of episodes')
        parser.add_argument('--continueGame',   type=float, default=0.0,       help='ContinueGame flag (-inf,+1.0]')
        parser.add_argument('--actionSpace',    type=str,   default="discrete", help='(discrete)/multidiscrete')
        parser.add_argument('--attButComb',     type=int,   default=0,          help='If to use attack button combinations (0=False)/1=True')
        parser.add_argument('--noAction',       type=int,   default=0,          help='If to use no action policy (0=False)')
        parser.add_argument('--hardCore',       type=int,   default=0,          help='Hard core mode (0=False)')
        parser.add_argument('--interactiveViz', type=int,   default=0,          help='Interactive Visualization (0=False)')
        opt = parser.parse_args()
        print(opt)

        vizFlag = bool(opt.interactiveViz)
        waitKey = 1;
        if vizFlag:
            waitKey = 0

        # Common settings
        diambraKwargs = {}
        diambraKwargs["gameId"]   = opt.gameId
        diambraKwargs["romsPath"] = os.path.join(base_path, "../../roms/mame/")

        diambraKwargs["mamePath"] = os.path.join(base_path, "../../customMAME/")
        diambraKwargs["libPath"] = os.path.join(base_path, "../../games_cpp/build/diambraEnvLib/libdiambraEnv.so")

        diambraKwargs["continueGame"] = opt.continueGame

        diambraKwargs["mameDiambraStepRatio"] = opt.frameRatio
        diambraKwargs["lockFps"] = False

        diambraKwargs["player"] = opt.player

        diambraKwargs["characters"] = [[opt.character1, opt.character1_2], [opt.character2, opt.character2_2]]
        diambraKwargs["charOutfits"] = [2, 2]

        # DIAMBRA gym kwargs
        diambraGymKwargs = {}
        diambraGymKwargs["actionSpace"] = [opt.actionSpace, opt.actionSpace]
        diambraGymKwargs["attackButCombinations"] = [opt.attButComb, opt.attButComb]
        if diambraKwargs["player"] != "P1P2":
            diambraGymKwargs["actionSpace"] = diambraGymKwargs["actionSpace"][0]
            diambraGymKwargs["attackButCombinations"] = diambraGymKwargs["attackButCombinations"][0]

        idxList = [0, 1]
        if diambraKwargs["player"] != "P1P2":
            idxList = [0]

        # Env wrappers kwargs
        wrapperKwargs = {}
        wrapperKwargs["noOpMax"] = 0
        wrapperKwargs["hwcObsResize"] = [128, 128, 1]
        wrapperKwargs["normalizeRewards"] = True
        wrapperKwargs["clipRewards"] = False
        wrapperKwargs["frameStack"] = 4
        wrapperKwargs["dilation"] = 1
        wrapperKwargs["actionsStack"] = 12
        wrapperKwargs["scale"] = True
        wrapperKwargs["scaleMod"] = 0

        # Additional obs key list
        keyToAdd = []
        keyToAdd.append("actions")

        if opt.gameId != "tektagt":
            keyToAdd.append("ownHealth")
            keyToAdd.append("oppHealth")
        else:
            keyToAdd.append("ownHealth1")
            keyToAdd.append("ownHealth2")
            keyToAdd.append("oppHealth1")
            keyToAdd.append("oppHealth2")
            keyToAdd.append("ownActiveChar")
            keyToAdd.append("oppActiveChar")

        keyToAdd.append("ownPosition")
        keyToAdd.append("oppPosition")
        if diambraKwargs["player"] != "P1P2":
            keyToAdd.append("stage")

        if opt.gameId != "tektagt":
            keyToAdd.append("ownChar")
            keyToAdd.append("oppChar")
        else:
            keyToAdd.append("ownChar1")
            keyToAdd.append("ownChar2")
            keyToAdd.append("oppChar1")
            keyToAdd.append("oppChar2")

        envId = opt.gameId + "_randomTestSBWrappers"
        hardCore = False if opt.hardCore == 0 else True
        numOfEnvs = 1
        env = makeStableBaselinesEnv(envId, numOfEnvs, timeDepSeed, diambraKwargs,
                                     diambraGymKwargs, wrapperKwargs,
                                     keyToAdd=keyToAdd, noVec=True, hardCore=hardCore)

        print("Observation Space:", env.observation_space)
        print("Action Space:", env.action_space)
        if not hardCore:
            print("Keys to Dict:")
            for k,v in env.keysToDict.items():
                print(k, v)

        nActions = env.nActions

        actionsPrintDict = env.printActionsDict

        observation = env.reset()

        showObs(observation, keyToAdd, env.keyToAddCount, wrapperKwargs["actionsStack"], nActions,
                waitKey, vizFlag, env.charNames, hardCore, idxList)

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
                    moveAction, attAction = discreteToMultiDiscreteAction(actions, env.nActions[0][0])
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
            showObs(observation, keyToAdd, env.keyToAddCount, wrapperKwargs["actionsStack"], nActions,
                    waitKey, vizFlag, env.charNames, hardCore, idxList)
            print("--")
            print("Current Cumulative Reward =", cumulativeEpRew)

            print("----------")

            if done:
                print("Resetting Env")
                currNumEp += 1
                print("Ep. # = ", currNumEp)
                print("Ep. Cumulative Rew # = ", cumulativeEpRew)
                cumulativeEpRewAll.append(cumulativeEpRew)
                cumulativeEpRew = 0.0

                observation = env.reset()
                showObs(observation, keyToAdd, env.keyToAddCount, wrapperKwargs["actionsStack"], nActions,
                        waitKey, vizFlag, env.charNames, hardCore, idxList)

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

import sys, os
import random
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
    opt = parser.parse_args()
    print(opt)

    base_path = os.path.dirname(__file__)

    sys.path.append(base_path)
    sys.path.append(os.path.join(base_path, '../../utils'))
    sys.path.append(os.path.join(base_path, '../../pythonGamePadInterface'))

    from diambraMameGym import diambraMame
    from diambraGamepad import diambraGamepad
    from policies import gamepadPolicy, RLPolicy # To train AI against another AI or HUM

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

    # Initialize P2 policy = GamePad
    gamePad_policy = gamepadPolicy(diambraGamepad)

    # DIAMBRA gym kwargs
    diambraGymKwargs = {}
    diambraGymKwargs["P2brain"] = None#gamePad_policy
    diambraGymKwargs["continueGame"] = opt.continueGame
    diambraGymKwargs["showFinal"] = False
    diambraGymKwargs["actionSpace"] = [opt.actionSpace, opt.actionSpace]
    diambraGymKwargs["attackButCombinations"] = [opt.attButComb, opt.attButComb]
    diambraGymKwargs["actBufLen"]             = 12

    envId = opt.gameId + "_Test"
    env = diambraMame(envId, diambraKwargs, **diambraGymKwargs)

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


    actionsPrintDict = env.printActionsDict()

    observation = env.reset()

    cumulativeEpRew = 0.0
    cumulativeEpRewAll = []

    maxNumEp = opt.nEpisodes
    currNumEp = 0

    while currNumEp < maxNumEp:

        actions = [None, None]
        for idx in range(2):

            actions[idx] = env.actionSpaces[idx].sample()

            if opt.noAction == 1 and idx == 0:
                if diambraGymKwargs["actionSpace"][idx] == "multiDiscrete":
                    for iEl, _ in enumerate(actions[idx]):
                        actions[idx][iEl] = 0
                else:
                    actions[idx] = 0

            if diambraGymKwargs["actionSpace"][idx] == "discrete":
                moveAction, attAction = env.discreteToMultiDiscreteAction(actions[idx])
            else:
                moveAction, attAction = actions[idx][0], actions[idx][1]

            if diambraKwargs["player"] != "P1P2" and idx == 1:
                continue

            print("(P{}) {} {}".format(idx+1, actionsPrintDict[0][moveAction],
                                              actionsPrintDict[1][attAction]))

        if diambraKwargs["player"] != "P1P2" and diambraGymKwargs["actionSpace"][0] == "discrete":
            actions = actions[0]
        else:
            actions = np.append(actions[0], actions[1])

        observation, reward, done, info = env.step(actions)

        cumulativeEpRew += reward

        print("Frames shape:", observation.shape)
        print("Reward:", reward)
        print("Current Cumulative Reward:", cumulativeEpRew)
        print("Actions Buffer P1 = ", info["actionsBufP1"])
        if diambraKwargs["player"] == "P1P2":
            print("Actions Buffer P2 = ", info["actionsBufP2"])
        print("Rewards:", info["rewards"])
        print("Fighting = ", info["fighting"])

        if diambraKwargs["player"] == "P1P2":
            print("Char P1 = ", env.charNames[env.playingCharacters[0]])
            print("Char P2 = ", env.charNames[env.playingCharacters[1]])
        else:
            print("Char = ", env.charNames[env.playingCharacters[env.playerId]])

        if opt.gameId == "tektagt":
            print("FightingP1 = ", info["fightingP1"])
            print("FightingP2 = ", info["fightingP2"])
            print("healthP1_1 = ", info["healthP1_1"])
            print("healthP1_2 = ", info["healthP1_2"])
            print("healthP2_1 = ", info["healthP2_1"])
            print("healthP2_2 = ", info["healthP2_2"])
            print("activeCharP1 = ", info["activeCharP1"])
            print("activeCharP2 = ", info["activeCharP2"])
        else:
            print("healthP1 = ", info["healthP1"])
            print("healthP2 = ", info["healthP2"])

        print("PositionP1 = ", info["positionP1"])
        print("PositionP2 = ", info["positionP2"])
        print("WinP1 = ", info["winsP1"])
        print("WinP2 = ", info["winsP2"])
        print("Stage = ", info["stage"])
        print("Round done = ", info["roundDone"])
        print("Stage done = ", info["stageDone"])
        print("Game done = ", info["gameDone"])
        print("Episode done = ", info["episodeDone"])
        print("Done = ", done)

        if done:
            print("Resetting Env")
            currNumEp += 1
            observation = env.reset()
            cumulativeEpRewAll.append(cumulativeEpRew)
            cumulativeEpRew = 0.0

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

    if opt.noAction == 1 and np.mean(cumulativeEpRewAll) > -2*(maxContinue+1)*env.maxHealth+0.001:
        raise RuntimeError("NoAction policy and average reward different than {} ({})".format(-2*(maxContinue+1)*env.maxHealth, np.mean(cumulativeEpRewAll)))

    print("ALL GOOD!")
except Exception as e:
    print(e)
    print("ALL BAD")

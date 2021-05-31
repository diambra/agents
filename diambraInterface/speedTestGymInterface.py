import sys, os, time
import random
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

    base_path = os.path.dirname(__file__)

    sys.path.append(base_path)

    from diambraMameGym import diambraMame

    diambraKwargs = {}
    diambraKwargs["romsPath"] = os.path.join(base_path, "../../roms/mame/")
    diambraKwargs["binaryPath"] = os.path.join(base_path, "../../customMAME/")
    diambraKwargs["frameRatio"] = opt.frameRatio
    diambraKwargs["throttle"] = False
    diambraKwargs["render"] = False
    diambraKwargs["sound"] = diambraKwargs["throttle"]

    diambraKwargs["player"] = opt.player

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
    diambraGymKwargs["actBufLen"]             = 12

    envId = opt.gameId + "_Test"
    env = diambraMame(envId, diambraKwargs, **diambraGymKwargs)

    observation = env.reset()

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

        actions = [None, None]
        for idx in range(2):

            actions[idx] = env.actionSpaces[idx].sample()

        if diambraKwargs["player"] != "P1P2" and diambraGymKwargs["actionSpace"][0] == "discrete":
            actions = actions[0]
        else:
            actions = np.append(actions[0], actions[1])

        observation, reward, done, info = env.step(actions)

        if done:
            print("Resetting Env")
            currNumEp += 1
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

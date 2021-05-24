import sys, os
import time
import argparse

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gameId', type=str, default="doapp", help='Game ID [(doapp), sfiii3n, tektagt, umk3]')
    opt = parser.parse_args()
    print(opt)

    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    base_path = os.path.dirname(__file__)

    sys.path.append(base_path)
    sys.path.append(os.path.join(base_path, '../../games'))

    tensorBoardFolder = "./{}stableBaselinesTestTensorboard/".format(opt.gameId)
    modelFolder = "./{}stableBaselinesTestModel/".format(opt.gameId)

    os.makedirs(modelFolder, exist_ok=True)

    from diambraMameGym import diambraMame
    from makeDiambraEnvSB import makeDiambraEnv

    import tensorflow as tf

    from customPolicies.utils import linear_schedule, AutoSave
    from customPolicies.customCnnPolicy import *

    from stable_baselines import PPO2

    # Diambra environment kwargs
    diambraKwargs = {}
    diambraKwargs["romsPath"] = os.path.join(base_path, "../../roms/mame/")
    diambraKwargs["binaryPath"] = os.path.join(base_path, "../../customMAME/")
    diambraKwargs["frameRatio"] = 6
    diambraKwargs["render"]      = True

    diambraKwargs["player"] = "Random" # P1 / P2

    # Game dependent kwawrgs
    if opt.gameId == "doapp":
        diambraKwargs["difficulty"]  = 3
        diambraKwargs["characters"] =["Random", "Random"]
    elif opt.gameId == "sfiii3n":
        diambraKwargs["difficulty"]  = 6
        diambraKwargs["characters"] =["Random", "Random"]
    elif opt.gameId == "tektagt":
        diambraKwargs["difficulty"]  = 6
        diambraKwargs["characters"] =[["Random", "Random"], ["Random", "Random"]]
    elif opt.gameId == "umk3":
        diambraKwargs["difficulty"]  = 3
        diambraKwargs["characters"] =["Random", "Random"]
    else:
        raise Exception("Game not implemented: {}".format(opt.gameId))

    diambraKwargs["charOutfits"] =[2, 2]

    # DIAMBRA gym kwargs
    diambraGymKwargs = {}
    diambraGymKwargs["P2brain"] = None
    diambraGymKwargs["continueGame"] = 0.0 # If < 0.0 means number of continues
    diambraGymKwargs["showFinal"] = False
    diambraGymKwargs["gamePads"] = [None, None]
    diambraGymKwargs["actionSpace"] = ["discrete", "discrete"]
    diambraGymKwargs["attackButCombinations"] = [False, False]

    # Gym Wrappers kwargs
    wrapperKwargs = {}
    wrapperKwargs["hwcObsResize"] = [128, 128, 1]
    wrapperKwargs["normalizeRewards"] = True
    wrapperKwargs["clipRewards"] = False
    wrapperKwargs["frameStack"] = 4
    wrapperKwargs["dilation"] = 1
    wrapperKwargs["scale"] = True
    wrapperKwargs["scaleMod"] = 0

    # Additional observations
    keyToAdd = []
    keyToAdd.append("actionsBuf") # env.actBufLen*(env.n_actions[0]+env.n_actions[1])

    if opt.gameId != "tektagt":
        keyToAdd.append("ownHealth")   # 1
        keyToAdd.append("oppHealth")   # 1
    else:
        keyToAdd.append("ownHealth_1") # 1
        keyToAdd.append("ownHealth_2") # 1
        keyToAdd.append("oppHealth_1") # 1
        keyToAdd.append("oppHealth_2") # 1

    keyToAdd.append("ownPosition")     # 1
    keyToAdd.append("oppPosition")     # 1
    keyToAdd.append("stage")           # 1
    keyToAdd.append("character")       # len(env.charNames)

    numEnv=2

    envId = opt.gameId + "_Train"
    env = makeDiambraEnv(diambraMame, envPrefix=envId, numEnv=numEnv, seed=timeDepSeed,
                         diambraKwargs=diambraKwargs, diambraGymKwargs=diambraGymKwargs,
                         wrapperKwargs=wrapperKwargs, keyToAdd=keyToAdd, useSubprocess=True)

    print("Obs_space = ", env.observation_space)
    print("Obs_space type = ", env.observation_space.dtype)
    print("Obs_space high = ", env.observation_space.high)
    print("Obs_space low = ", env.observation_space.low)

    print("Act_space = ", env.action_space)
    print("Act_space type = ", env.action_space.dtype)
    if diambraGymKwargs["actionSpace"][0] == "multiDiscrete":
        print("Act_space n = ", env.action_space.nvec)
    else:
        print("Act_space n = ", env.action_space.n)

    # Policy param
    nActions = env.get_attr("nActions")[0][0]
    actBufLen = env.get_attr("actBufLen")[0]

    policyKwargs={}
    policyKwargs["n_add_info"] = actBufLen*(nActions[0]+nActions[1]) + len(keyToAdd)-2 # No Char Info
    policyKwargs["layers"] = [64, 64]

    policyKwargs["cnn_extractor"] = local_nature_cnn_small

    print("nAddInfo =", policyKwargs["n_add_info"])

    # PPO param
    setGamma = 0.94
    setLearningRate = linear_schedule(2.5e-4, 2.5e-6)
    setClipRange = linear_schedule(0.15, 0.025)
    setClipRangeVf = setClipRange

    # Initialize the model
    model = PPO2(CustCnnPolicy, env, verbose=1,
                 gamma = setGamma, nminibatches=4, noptepochs=4, n_steps=128,
                 learning_rate=setLearningRate, cliprange=setClipRange, cliprange_vf=setClipRangeVf,
                 tensorboard_log=tensorBoardFolder, policy_kwargs=policyKwargs)

    print("Model discount factor = ", model.gamma)

    # Create the callback: autosave every USER DEF steps
    autoSaveCallback = AutoSave(check_freq=256, numEnv=numEnv, save_path=os.path.join(modelFolder,"0M_"))

    # Train the agent
    timeSteps = 512
    model.learn(total_timesteps=timeSteps, callback=autoSaveCallback)

    # Save the agent
    model.save(os.path.join(modelFolder, "512"))

    print("ALL GOOD!")
except Exception as e:
    print(e)
    print("ALL BAD")

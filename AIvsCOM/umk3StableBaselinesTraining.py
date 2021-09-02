import sys, os, time

if __name__ == '__main__':
    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_path, '../'))

    modelFolder = os.path.join(base_path, "umk3Model/")
    tensorBoardFolder = os.path.join(base_path, "umk3TB/")

    os.makedirs(modelFolder, exist_ok=True)

    from makeStableBaselinesEnv import makeStableBaselinesEnv

    import tensorflow as tf

    from sbUtils import linear_schedule, AutoSave, modelCfgSave
    from customPolicies.customCnnPolicy import CustCnnPolicy, local_nature_cnn_small

    from stable_baselines import PPO2

    # Common settings
    diambraKwargs = {}
    diambraKwargs["gameId"]   = "umk3"
    diambraKwargs["romsPath"] = os.path.join(base_path, "../../roms/mame/")

    diambraKwargs["mameDiambraStepRatio"] = 6
    diambraKwargs["lockFps"] = False
    diambraKwargs["render"]  = True

    diambraKwargs["player"] = "Random" # P1 / P2

    diambraKwargs["characters"] =[["Sektor", "Random"], ["Sektor", "Random"]]

    diambraKwargs["difficulty"]  = 4
    diambraKwargs["charOutfits"] =[2, 2]

    # DIAMBRA gym kwargs
    diambraGymKwargs = {}
    diambraGymKwargs["actionSpace"] = "discrete"
    diambraGymKwargs["attackButCombinations"] = False

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

    keyToAdd.append("ownHealth")
    keyToAdd.append("oppHealth")

    keyToAdd.append("ownPosition")
    keyToAdd.append("oppPosition")
    keyToAdd.append("stage")

    #keyToAdd.append("ownChar1")
    #keyToAdd.append("ownChar2")
    #keyToAdd.append("oppChar1")
    #keyToAdd.append("oppChar2")

    numEnv=8

    envId = "umk3_Train"
    env = makeStableBaselinesEnv(envId, numEnv, timeDepSeed, diambraKwargs, diambraGymKwargs,
                                 wrapperKwargs, keyToAdd=keyToAdd, useSubprocess=True)

    print("Obs_space = ", env.observation_space)
    print("Obs_space type = ", env.observation_space.dtype)
    print("Obs_space high = ", env.observation_space.high)
    print("Obs_space low = ", env.observation_space.low)

    print("Act_space = ", env.action_space)
    print("Act_space type = ", env.action_space.dtype)
    if diambraGymKwargs["actionSpace"] == "multiDiscrete":
        print("Act_space n = ", env.action_space.nvec)
    else:
        print("Act_space n = ", env.action_space.n)

    # Policy param
    nActions      = env.get_attr("nActions")[0][0]
    nActionsStack = env.get_attr("nActionsStack")[0]
    nChar         = env.get_attr("numberOfCharacters")[0]
    charNames     = env.get_attr("charNames")[0]

    policyKwargs={}
    policyKwargs["n_add_info"] = nActionsStack*(nActions[0]+nActions[1]) + len(keyToAdd)-1
    policyKwargs["layers"] = [64, 64]

    policyKwargs["cnn_extractor"] = local_nature_cnn_small

    print("nActions =", nActions)
    print("nChar =", nChar)
    print("nAddInfo =", policyKwargs["n_add_info"])

    # PPO param
    setGamma = 0.94
    setLearningRate = linear_schedule(2.5e-4, 2.5e-6)
    setClipRange = linear_schedule(0.15, 0.025)
    setClipRangeVf = setClipRange
    # Initialize the model
    model = PPO2(CustCnnPolicy, env, verbose=1,
                 gamma=setGamma, nminibatches=4, noptepochs=4, n_steps=128,
                 learning_rate=setLearningRate, cliprange=setClipRange,
                 cliprange_vf=setClipRangeVf, policy_kwargs=policyKwargs,
                 tensorboard_log=tensorBoardFolder)
    #OR
    '''
    setLearningRate = linear_schedule(5.0e-5, 2.5e-6)
    setClipRange    = linear_schedule(0.075, 0.025)
    setClipRangeVf  = setClipRange
    # Load the trained agent
    model = PPO2.load(os.path.join(modelFolder, "0M"), env=env,
                      policy_kwargs=policyKwargs, gamma=setGamma, learning_rate=setLearningRate,
                      cliprange=setClipRange, cliprange_vf=setClipRangeVf,
                      tensorboard_log=tensorBoardFolder)
    '''

    print("Model discount factor = ", model.gamma)

    # Create the callback: autosave every USER DEF steps
    autoSaveCallback = AutoSave(check_freq=1000000, numEnv=numEnv,
                                save_path=os.path.join(modelFolder, "0M_"))

    # Train the agent
    timeSteps = 20000000
    model.learn(total_timesteps=timeSteps, callback=autoSaveCallback)

    # Save the agent
    modelPath = os.path.join(modelFolder, "20M")
    model.save(modelPath)
    # Save the correspondent CFG file
    modelCfgSave(modelPath, "PPOSmall", nActions, charNames,
                 diambraKwargs, diambraGymKwargs, wrapperKwargs, keyToAdd)

    # Close the environment
    env.close()

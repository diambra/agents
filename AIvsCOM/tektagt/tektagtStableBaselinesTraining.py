import sys, os, time

if __name__ == '__main__':
    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_path, '../'))

    modelFolder = os.path.join(base_path, "tektagtModel/")
    tensorBoardFolder = os.path.join(base_path, "tektagtTB/")

    os.makedirs(modelFolder, exist_ok=True)

    from makeStableBaselinesEnv import makeStableBaselinesEnv
    from wrappers.tektagRewWrap import tektagRoundEndChar2Penalty, tektagHealthBarUnbalancePenalty

    import tensorflow as tf

    from sbUtils import linear_schedule, AutoSave, modelCfgSave
    from customPolicies.customCnnPolicy import CustCnnPolicy, local_nature_cnn_small

    from stable_baselines import PPO2

    # Settings
    settings = {}
    settings["gameId"]   = "tektagt"
    settings["romsPath"] = os.path.join(base_path, "../../roms/mame/")

    settings["stepRatio"] = 6
    settings["lockFps"] = False
    settings["render"]  = False

    settings["player"] = "Random" # P1 / P2

    settings["characters"] =[["Jin", "Yoshimitsu"], ["Jin", "Yoshimitsu"]]

    settings["difficulty"]  = 6
    settings["charOutfits"] =[2, 2]

    settings["continueGame"] = -2.0
    settings["showFinal"] = False

    settings["actionSpace"] = "discrete"
    settings["attackButCombination"] = False

    # Wrappers settings
    wrappersSettings = {}
    wrappersSettings["noOpMax"] = 0
    wrappersSettings["hwcObsResize"] = [128, 128, 1]
    wrappersSettings["normalizeRewards"] = True
    wrappersSettings["clipRewards"] = False
    wrappersSettings["frameStack"] = 4
    wrappersSettings["dilation"] = 1
    wrappersSettings["actionsStack"] = 12
    wrappersSettings["scale"] = True
    wrappersSettings["scaleMod"] = 0

    # Additional custom wrappers
    customWrappers = [tektagRoundEndChar2Penalty, tektagHealthBarUnbalancePenalty]

    # Additional obs key list
    keyToAdd = []
    keyToAdd.append("actions")

    keyToAdd.append("ownHealth1")
    keyToAdd.append("ownHealth2")
    keyToAdd.append("oppHealth1")
    keyToAdd.append("oppHealth2")
    keyToAdd.append("ownActiveChar")
    keyToAdd.append("oppActiveChar")

    keyToAdd.append("ownSide")
    keyToAdd.append("oppSide")
    keyToAdd.append("stage")

    #keyToAdd.append("ownChar1")
    #keyToAdd.append("ownChar2")
    #keyToAdd.append("oppChar1")
    #keyToAdd.append("oppChar2")

    numEnv=16

    envId = "tektagt_Train"
    env = makeStableBaselinesEnv(envId, numEnv, timeDepSeed, settings,
                                 wrappersSettings, customWrappers=customWrappers,
                                 keyToAdd=keyToAdd, useSubprocess=True)

    print("Obs_space = ", env.observation_space)
    print("Obs_space type = ", env.observation_space.dtype)
    print("Obs_space high = ", env.observation_space.high)
    print("Obs_space low = ", env.observation_space.low)

    print("Act_space = ", env.action_space)
    print("Act_space type = ", env.action_space.dtype)
    if settings["actionSpace"] == "multiDiscrete":
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
    modelCheckpoint = "235M_penalties"
    '''
    setLearningRate = linear_schedule(2.5e-4, 2.5e-6)
    setClipRange = linear_schedule(0.15, 0.025)
    setClipRangeVf = setClipRange
    # Initialize the model
    model = PPO2(CustCnnPolicy, env, verbose=1,
                 gamma=setGamma, nminibatches=8, noptepochs=4, n_steps=128,
                 learning_rate=setLearningRate, cliprange=setClipRange,
                 cliprange_vf=setClipRangeVf, policy_kwargs=policyKwargs,
                 tensorboard_log=tensorBoardFolder)
    #OR
    '''
    #setLearningRate = linear_schedule(8.0e-5, 2.5e-6)
    #setClipRange    = linear_schedule(0.095, 0.025)
    setLearningRate = linear_schedule(2.0e-5, 2.5e-6)
    setClipRange    = linear_schedule(0.050, 0.025)
    setClipRangeVf  = setClipRange
    # Load the trained agent
    model = PPO2.load(os.path.join(modelFolder, modelCheckpoint), env=env,
                      policy_kwargs=policyKwargs, gamma=setGamma, learning_rate=setLearningRate,
                      cliprange=setClipRange, cliprange_vf=setClipRangeVf,
                      tensorboard_log=tensorBoardFolder)

    print("Model discount factor = ", model.gamma)

    # Create the callback: autosave every USER DEF steps
    autoSaveCallback = AutoSave(check_freq=1000000, numEnv=numEnv,
                                save_path=os.path.join(modelFolder, modelCheckpoint+"_"))

    # Train the agent
    timeSteps = 10000000
    model.learn(total_timesteps=timeSteps, callback=autoSaveCallback)

    # Save the agent
    modelPath = os.path.join(modelFolder, "245M_penalties")
    model.save(modelPath)
    # Save the correspondent CFG file
    modelCfgSave(modelPath, "PPOSmall", nActions, charNames,
                 settings, wrappersSettings, keyToAdd)

    # Close the environment
    env.close()

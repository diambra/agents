import sys, os, time
import argparse

if __name__ == '__main__':
    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('--romsPath',    type=str,  required=True, help='Roms Path')
    opt = parser.parse_args()
    print(opt)

    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_path, '../'))

    from makeStableBaselinesEnv import makeStableBaselinesEnv

    import tensorflow as tf

    from sbUtils import linear_schedule, AutoSave, modelCfgSave
    from customPolicies.customCnnPolicy import CustCnnPolicy, local_nature_cnn_small

    from stable_baselines import PPO2

    # Settings settings
    settings = {}
    settings["gameId"]   = "doapp"
    settings["romsPath"] = opt.romsPath

    settings["stepRatio"] = 6
    settings["lockFps"]   = False
    settings["render"]    = False

    settings["player"] = "Random" # P1 / P2

    settings["characters"] =[["Random", "Random"], ["Random", "Random"]]

    settings["difficulty"]  = 3
    settings["charOutfits"] =[2, 2]

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

    # Additional obs key list
    keyToAdd = []
    keyToAdd.append("actions")
    keyToAdd.append("ownHealth")
    keyToAdd.append("oppHealth")
    keyToAdd.append("ownSide")
    keyToAdd.append("oppSide")
    keyToAdd.append("stage")

    keyToAdd.append("ownChar")
    keyToAdd.append("oppChar")

    numEnv=8

    envId = "doapp_Train"
    env = makeStableBaselinesEnv(envId, numEnv, timeDepSeed, settings, wrappersSettings,
                                 keyToAdd=keyToAdd, useSubprocess=True)

    # Policy param
    nActions      = env.get_attr("nActions")[0][0]
    nActionsStack = env.get_attr("nActionsStack")[0]
    nChar         = env.get_attr("numberOfCharacters")[0]
    charNames     = env.get_attr("charNames")[0]

    policyKwargs={}
    policyKwargs["n_add_info"] = nActionsStack*(nActions[0]+nActions[1]) + len(keyToAdd)-3 + 2*nChar
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
    nSteps = 128

    # Initialize the model
    model = PPO2(CustCnnPolicy, env, verbose=1,
                 gamma = setGamma, nminibatches=4, noptepochs=4, n_steps=nSteps,
                 learning_rate=setLearningRate, cliprange=setClipRange, cliprange_vf=setClipRangeVf,
                 policy_kwargs=policyKwargs)

    print("Model discount factor = ", model.gamma)

    # Train the agent
    timeSteps = nSteps*2*numEnv
    tic = time.time()
    model.learn(total_timesteps=timeSteps)
    toc = time.time()

    print("Time elapsed = ", toc-tic)

    # Close the environment
    env.close()

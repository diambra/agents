import sys, os, time

if __name__ == '__main__':
    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_path, '../../'))

    modelFolder = os.path.join(base_path, "samsh5spModel/")

    from makeStableBaselinesEnv import makeStableBaselinesEnv
    from sbUtils import showObs

    import tensorflow as tf

    from stable_baselines import PPO2

    # Settings
    settings = {}
    settings["gameId"]   = "samsh5sp"
    settings["stepRatio"] = 6
    settings["frameShape"] = [128, 128, 1]
    settings["player"] = "P1" # P1 / P2

    settings["characters"] =[["Haohmaru"], ["Haohmaru"]]

    settings["difficulty"]  = 6
    settings["charOutfits"] =[2, 2]

    settings["continueGame"] = 0.0
    settings["showFinal"] = False

    settings["actionSpace"] = "discrete"
    settings["attackButCombination"] = False

    # Wrappers settings
    wrappersSettings = {}
    wrappersSettings["noOpMax"] = 0
    wrappersSettings["rewardNormalization"] = True
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

    env, numEnv = makeStableBaselinesEnv(timeDepSeed, settings, wrappersSettings,
                                         keyToAdd=keyToAdd, noVec=True)

    # Load the trained agent
    model = PPO2.load(os.path.join(modelFolder, "293M"))

    obs = env.reset()
    cumulativeRew = 0.0

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)
        #showObs(obs, keyToAdd, env.keyToAddCount, wrapperKwargs["actionsStack"], env.nActions,
        #        waitKey, True, env.charNames, False, [0])
        cumulativeRew += reward

        if done:
            stage = 1
            print("Cumulative Rew =", cumulativeRew)
            cumulativeRew = 0.0
            obs = env.reset()

    # Close the environment
    env.close()

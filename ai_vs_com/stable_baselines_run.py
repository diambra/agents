import sys, os, time

if __name__ == '__main__':
    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_path, '../../'))

    modelFolder = os.path.join(base_path, "tektagtModel/")

    from makeStableBaselinesEnv import makeStableBaselinesEnv

    import tensorflow as tf

    from stable_baselines import PPO2

    # Settings
    settings = {}
    settings["gameId"]   = "tektagt"
    settings["stepRatio"] = 6
    settings["frameShape"] = [128, 128, 1]
    settings["player"] = "P1" # P1 / P2

    settings["characters"] =[["Jin", "Yoshimitsu"], ["Jin", "Yoshimitsu"]]

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

    env, numEnv = makeStableBaselinesEnv(timeDepSeed, settings, wrappersSettings,
                                         keyToAdd=keyToAdd, noVec=True)

    # Load the trained agent
    model = PPO2.load(os.path.join(modelFolder, "235M_penalties"))

    obs = env.reset()
    cumulativeRew = 0.0

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)
        cumulativeRew += reward

        if done:
            print("Cumulative Rew =", cumulativeRew)
            cumulativeRew = 0.0
            obs = env.reset()

    # Close the environment
    env.close()

import sys, os, time

if __name__ == '__main__':
    timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_path, '../'))

    modelFolder = os.path.join(base_path, "tektagtModel/")

    from makeStableBaselinesEnv import makeStableBaselinesEnv

    import tensorflow as tf

    from stable_baselines import PPO2

    # Common settings
    diambraKwargs = {}
    diambraKwargs["gameId"]   = "tektagt"
    diambraKwargs["romsPath"] = os.path.join(base_path, "../../roms/mame/")

    diambraKwargs["stepRatio"] = 6
    diambraKwargs["lockFps"] = False
    diambraKwargs["render"]  = True

    diambraKwargs["player"] = "P2" # P1 / P2

    diambraKwargs["characters"] =[["Jin", "Yoshimitsu"], ["Jin", "Yoshimitsu"]]

    diambraKwargs["difficulty"]  = 1
    diambraKwargs["charOutfits"] =[2, 2]

    diambraKwargs["continueGame"] = 1.0
    diambraKwargs["showFinal"] = False

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

    numEnv=1

    envId = "tektagt_Train"
    env = makeStableBaselinesEnv(envId, numEnv, timeDepSeed, diambraKwargs, diambraGymKwargs,
                                 wrapperKwargs, keyToAdd=keyToAdd, noVec=True)

    # Load the trained agent
    model = PPO2.load(os.path.join(modelFolder, "136M"))

    obs = env.reset()

    while True:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

    # Close the environment
    env.close()

#gameId = "doapp"
#gameId = "sfiii3n"
#gameId = "umk3"
gameId = "tektagt"

import sys, os
import time
import cv2
import tensorflow as tf

timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

sys.path.append(os.path.join(os.path.abspath(''), '../../games/'))
sys.path.append(os.path.join(os.path.abspath(''), '../../utils'))
sys.path.append(os.path.join(os.path.abspath(''), '../../pythonGamePadInterface'))

from diambraGamepad import diambraGamepad
from policies import gamepadPolicy, RLPolicy # To train AI against another AI or HUM

# Common settings
diambraKwargs = {}
diambraKwargs["roms_path"] = "../../roms/mame/"
diambraKwargs["binary_path"] = "../../customMAME/"
diambraKwargs["frame_ratio"] = 6
diambraKwargs["throttle"] = True
diambraKwargs["sound"] = diambraKwargs["throttle"]
if gameId != "tektagt":
    diambraKwargs["characters"] = ["Random", "Select"]
else:
    diambraKwargs["characters"] = [["Random", "Select"], ["Random", "Kuma"]]
diambraKwargs["charOutfits"] = [2, 2]

# GamePad policy initialization
gamePad_policy = gamepadPolicy(diambraGamepad)

diambraKwargs["player"] = "Random" # 1P
#diambraKwargs["player"] = "P1P2" # 2P

#keyToAdd = None
keyToAdd = []
keyToAdd.append("actionsBufP1")
if diambraKwargs["player"] == "P1P2":
    keyToAdd.append("actionsBufP2") # Only 2P

if gameId != "tektagt": # DOA, SFIII, UMK3
    keyToAdd.append("ownHealth")
    keyToAdd.append("oppHealth")
else: # TEKTAG
    keyToAdd.append("ownHealth_1")
    keyToAdd.append("ownHealth_2")
    keyToAdd.append("oppHealth_1")
    keyToAdd.append("oppHealth_2")

keyToAdd.append("ownPosition")
keyToAdd.append("oppPosition")
#keyToAdd.append("ownWins")
#keyToAdd.append("oppWins")
#keyToAdd.append("stage")
keyToAdd.append("characters")

# Recording kwargs
trajRecKwargs = None
trajRecKwargs = {}
trajRecKwargs["user_name"] = "Alex"
trajRecKwargs["file_path"] = os.path.join( "~/DIAMBRA/trajRecordings", gameId)
trajRecKwargs["ignore_p2"] = 0
trajRecKwargs["commitHash"] = "0000000"

trajRecKwargs = None

from makeDiambraEnv import *

# DIAMBRA gym kwargs
diambraGymKwargs = {}
diambraGymKwargs["P2brain"] = None
diambraGymKwargs["continue_game"] = 0.0
diambraGymKwargs["gamePads"] = [gamePad_policy, gamePad_policy]
if diambraKwargs["player"] != "P1P2":
    diambraGymKwargs["show_final"] = True
diambraGymKwargs["actionSpace"] = ["discrete", "multiDiscrete"]
diambraGymKwargs["attackButCombinations"] = [True, False]

# Wrappers kwargs
wrapperKwargs = {}
wrapperKwargs["hwc_obs_resize"] = [256, 256, 1]
wrapperKwargs["normalize_rewards"] = True
wrapperKwargs["clip_rewards"] = False
wrapperKwargs["frame_stack"] = 6
wrapperKwargs["dilation"] = 1
wrapperKwargs["scale"] = True
wrapperKwargs["scale_mod"] = 0

numEnv=1
# Environment initialization
envId = gameId + "_Test"
env = make_diambra_env(diambraMame, env_prefix=envId, num_env=numEnv, seed=timeDepSeed,
                       diambra_kwargs=diambraKwargs, diambra_gym_kwargs=diambraGymKwargs,
                       wrapper_kwargs=wrapperKwargs, traj_rec_kwargs=trajRecKwargs,
                       key_to_add=keyToAdd, no_vec=True)

# Start game
observation = env.reset()

shp = observation.shape

additionalPar = int(observation[0,0,shp[2]-1])

nScalarAddPar = additionalPar - 2*len(env.charNames)\
                - env.actBufLen*(env.n_actions[0][0]+env.n_actions[0][1]) # 1P
if diambraKwargs["player"] == "P1P2":
    nScalarAddPar = additionalPar - 2*len(env.charNames)\
                    - env.actBufLen*(env.n_actions[0][0]+env.n_actions[0][1] +\
                                     env.n_actions[1][0]+env.n_actions[1][1])# 2P

limAct = [None, None]
for idx in range(2):
    limAct[idx] = [env.actBufLen * env.n_actions[idx][0],
                   env.actBufLen * env.n_actions[idx][0] + env.actBufLen * env.n_actions[idx][1]]

print("Additional Par = ", additionalPar)
print("N scalar actions = ", nScalarAddPar)
print("Len char names = ", len(env.charNames))
#input("Pause")

cumulativeEpRew = 0.0
cumulativeEpRewAll = []

maxNumEp = 100
currNumEp = 0

#sess = tf.Session();

while currNumEp < maxNumEp:

    # 1P
    #action = [0, 0]
    action = env.action_spaces[0].sample()

    # 2P
    action2 = env.action_spaces[1].sample()
    if diambraKwargs["player"] == "P1P2":
        action = np.append(action, action2)

    #action = int(input("Action"))
    print("Action:", action)
    observation, reward, done, info = env.step(action)

    #if reward != 0:
    #    print("Reward =", info["rewards"])
    #    input("AAA")

    doit = True
    if info["round_done"] == True or doit:

        addPar = observation[:,:,shp[2]-1]
        addPar = np.reshape(addPar, (-1))
        addPar = addPar[1:additionalPar+1]
        actions = addPar[0:additionalPar-nScalarAddPar-2*env.numberOfCharacters]

        moveActionsP1   = actions[0:limAct[0][0]]
        attackActionsP1 = actions[limAct[0][0]:limAct[0][1]]
        moveActionsP1   = np.reshape(moveActionsP1, (env.actBufLen,-1))
        attackActionsP1 = np.reshape(attackActionsP1, (env.actBufLen,-1))
        print("Move actions P1 =\n", moveActionsP1)
        print("Attack actions P1 =\n ", attackActionsP1)
        #input("Pausa1")

        # 2P
        if diambraKwargs["player"] == "P1P2":
            moveActionsP2   = actions[limAct[0][1]:limAct[0][1]+limAct[1][0]]
            attackActionsP2 = actions[limAct[0][1]+limAct[1][0]:limAct[0][1]+limAct[1][1]]
            moveActionsP2   = np.reshape(moveActionsP2, (env.actBufLen,-1))
            attackActionsP2 = np.reshape(attackActionsP2, (env.actBufLen,-1))
            print("Move actions P2 =\n", moveActionsP2)
            print("Attack actions P2 =\n", attackActionsP2)
            #input("Pausa1")

        others = addPar[additionalPar-nScalarAddPar-2*env.numberOfCharacters:]
        print("ownHealth = ", others[0])
        print("oppHealth = ", others[1])
        print("ownPosition = ", others[2])
        print("oppPosition = ", others[3])
        #print("stage = ", others[4])
        print("Playing Char P1 = ", env.charNames[list(others[nScalarAddPar:
                                                              nScalarAddPar + env.numberOfCharacters]).index(1.0)])

        if diambraKwargs["player"] == "P1P2":
            print("Playing Char P2 = ", env.charNames[list(others[nScalarAddPar + env.numberOfCharacters:
                                                                  nScalarAddPar + 2*env.numberOfCharacters]).index(1.0)])

        #input("Pausa1")

        #print(np.array(observation).astype(np.float32).shape)
        #input("Pausa2")
        #print(tf.cast(observation, tf.float32).eval(session=sess))


    #if info["round_done"] == True or doit:
    #    tensor = tf.cast(observation, tf.float32)
    #    tensor2 = tensor[:,:,shp[2]-1]
    #    print("Number off additional param = ", tensor2[0,0].eval(session=sess))
    #    tensor2 = tf.reshape(tensor2, [-1])
    #    #print("After reshaping = ", tensor2)
    #    tensor2 = tensor2[1:1+additionalPar]
    #    #print("Tensor=", tensor)
    #    tensor_actions = tf.reshape(tensor2[0:additionalPar-4], [12,-1])
    #    print("Tensor=", tensor2[additionalPar-4:additionalPar+1].eval(session=sess))
    #    print("Tensor=", tensor_actions.eval(session=sess))
    #    input("Pausa")

    obs = np.array(observation).astype(np.float32)

    #for idx in range(shp[2]-1):
    #    cv2.imshow("image"+str(idx), obs[:,:,idx])

    #cv2.waitKey()

    #print("Frames shape:", observation.shape)
    #print("Reward:", reward)
    #print("Fighting = ", info["fighting"])
    #print("Rewards = ", info["rewards"])
    #print("HealthP1 = ", info["healthP1"])
    #print("HealthP2 = ", info["healthP2"])
    #print("HealthP1_1 = ", info["healthP1_1"])
    #print("HealthP1_2 = ", info["healthP1_2"])
    #print("HealthP2_1 = ", info["healthP2_1"])
    #print("HealthP2_2 = ", info["healthP2_2"])
    #print("PositionP1 = ", info["positionP1"])
    #print("PositionP2 = ", info["positionP2"])
    #print("WinP1 = ", info["winsP1"])
    #print("WinP2 = ", info["winsP2"])


    cumulativeEpRew += reward

    if np.any(done):
        currNumEp += 1
        print("Ep. # = ", currNumEp)
        print("Ep. Cumulative Rew # = ", cumulativeEpRew)
        sys.stdout.flush()
        cumulativeEpRewAll.append(cumulativeEpRew)
        cumulativeEpRew = 0.0

        observation = env.reset()

        addPar = observation[:,:,shp[2]-1]
        addPar = np.reshape(addPar, (-1))
        addPar = addPar[1:additionalPar+1]
        others = addPar[additionalPar-5-env.numberOfCharacters:]
        input("Stop")

print("Mean cumulative reward = ", np.mean(cumulativeEpRewAll))
print("Std cumulative reward = ", np.std(cumulativeEpRewAll))

env.close()

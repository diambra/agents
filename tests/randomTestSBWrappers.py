#!/usr/bin/env python
# coding: utf-8

# In[ ]:


gameId = "doapp"
#gameId = "sfiii3n"
#gameId = "umk3"
#gameId = "tektagt"


# In[ ]:


import sys, os
from os.path import expanduser 
import time
import cv2
import numpy as np

homeDir = expanduser("~") 

timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

sys.path.append(os.path.join(os.path.abspath(''), '../.'))  
sys.path.append(os.path.join(os.path.abspath(''), '../../gym/'))
                                                                                                                                 
from utils import discreteToMultiDiscreteAction
from makeStableBaselinesEnv import makeStableBaselinesEnv


# In[ ]:


# Common settings
diambraKwargs = {}
diambraKwargs["romsPath"] = "../../roms/mame/"
diambraKwargs["binaryPath"] = "../../customMAME/"
diambraKwargs["frameRatio"] = 6
diambraKwargs["throttle"] = False
diambraKwargs["sound"] = diambraKwargs["throttle"]

#diambraKwargs["player"] = "Random" # 1P
diambraKwargs["player"] = "P1P2" # 2P


# In[ ]:


if gameId != "tektagt":
    diambraKwargs["characters"] = ["Random", "Random"]
else:
    diambraKwargs["characters"] = [["Random", "Random"], ["Random", "Random"]]
diambraKwargs["charOutfits"] = [2, 2]


# In[ ]:


# DIAMBRA gym kwargs
diambraGymKwargs = {}
diambraGymKwargs["actionSpace"] = ["multiDiscrete", "multiDiscrete"]
diambraGymKwargs["attackButCombinations"] = [False, False]
diambraGymKwargs["actBufLen"] = 12
if diambraKwargs["player"] != "P1P2":
    diambraGymKwargs["showFinal"] = False
    diambraGymKwargs["continueGame"] = -1.0
    diambraGymKwargs["actionSpace"] = diambraGymKwargs["actionSpace"][0]
    diambraGymKwargs["attackButCombinations"] = diambraGymKwargs["attackButCombinations"][0]


# In[ ]:


# Recording kwargs
trajRecKwargs = {}                                                          
trajRecKwargs["userName"] = "Alex"
trajRecKwargs["filePath"] = os.path.join( homeDir, "DIAMBRA/trajRecordings", gameId)
trajRecKwargs["ignoreP2"] = 0                                    
trajRecKwargs["commitHash"] = "0000000"


# In[ ]:


trajRecKwargs = None


# In[ ]:


# Env wrappers kwargs
wrapperKwargs = {}
wrapperKwargs["noOpMax"] = 0
wrapperKwargs["hwcObsResize"] = [128, 128, 1]
wrapperKwargs["normalizeRewards"] = True
wrapperKwargs["clipRewards"] = False
wrapperKwargs["frameStack"] = 4
wrapperKwargs["dilation"] = 1
wrapperKwargs["scale"] = True
wrapperKwargs["scaleMod"] = 0


# In[ ]:


# Additional obs key list
keyToAdd = []
keyToAdd.append("actionsBuf") # env.actBufLen*(env.n_actions[0]+env.n_actions[1])

if gameId != "tektagt":                                                         
    keyToAdd.append("ownHealth")   # 1                                            
    keyToAdd.append("oppHealth")   # 1                                                
else:                                                                           
    keyToAdd.append("ownHealth1") # 1                                             
    keyToAdd.append("ownHealth2") # 1                                             
    keyToAdd.append("oppHealth1") # 1                                              
    keyToAdd.append("oppHealth2") # 1  
    keyToAdd.append("ownActiveChar") # 1
    keyToAdd.append("oppActiveChar") # 1
    
keyToAdd.append("ownPosition")     # 1
keyToAdd.append("oppPosition")     # 1
if diambraKwargs["player"] != "P1P2":
    keyToAdd.append("stage")           # 1
keyToAdd.append("ownChar")       # len(env.charNames)
keyToAdd.append("oppChar")       # len(env.charNames)


# In[ ]:


envId = gameId + "_Test"
numOfEnvs = 1
env = makeStableBaselinesEnv(envId, numOfEnvs, timeDepSeed, diambraKwargs, diambraGymKwargs, 
                             wrapperKwargs, trajRecKwargs, keyToAdd=keyToAdd, noVec=True)


# In[ ]:


print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Keys to Dict:", env.keysToDict)


# In[ ]:


limAct = [None, None]
if diambraKwargs["player"] != "P1P2":
    for idx in range(2):                                                        
        limAct[idx] = [env.actBufLen * env.nActions[0],                                
                       env.actBufLen * env.nActions[0] + env.actBufLen * env.nActions[1]]     
else:
    for idx in range(2):                                                        
        limAct[idx] = [env.actBufLen * env.nActions[idx][0],                                
                       env.actBufLen * env.nActions[idx][0] + env.actBufLen * env.nActions[idx][1]]     
                                                                                
# Visualize Obs content                                                     
def showObs(observation, limAct, waitKey=0):                                    
                                                                                
    shp = observation.shape                 
    nChars = len(env.charNames)
    additionalParP1 = int(observation[0,0,shp[2]-1])                           
                                                                                    
    # 1P
    if diambraKwargs["player"] != "P1P2":
        nScalarAddParP1 = additionalParP1 - 2*nChars - env.actBufLen*(env.nActions[0]+env.nActions[1])
    else:
        nScalarAddParP1 = additionalParP1 - 2*nChars - env.actBufLen*(env.nActions[0][0]+env.nActions[0][1])
                                                                                    
    print("Additional Par P1 =", additionalParP1)                                
    print("N scalar actions P1 =", nScalarAddParP1)                              
                                                                                    
    addPar = observation[:,:,shp[2]-1]                                       
    addPar = np.reshape(addPar, (-1))                                        
    addParP1 = addPar[1:additionalParP1+1]                                       
    actionsP1 = addParP1[0:additionalParP1-nScalarAddParP1-2*nChars]                   
                                                                                    
    moveActionsP1   = actionsP1[0:limAct[0][0]]                                
    attackActionsP1 = actionsP1[limAct[0][0]:limAct[0][1]]                     
    moveActionsP1   = np.reshape(moveActionsP1, (env.actBufLen,-1))              
    attackActionsP1 = np.reshape(attackActionsP1, (env.actBufLen,-1))            
    print("Move actions P1 =\n", moveActionsP1)                              
    print("Attack actions P1 =\n ", attackActionsP1)                         
                                                                                    
    othersP1 = addParP1[additionalParP1-nScalarAddParP1-2*nChars:]                     
    if gameId != "tektagt":
        print("ownHealthP1 = ", othersP1[0])
        print("oppHealthP1 = ", othersP1[1])
        print("ownPositionP1 = ", othersP1[2])
        print("oppPositionP1 = ", othersP1[3])
        if diambraKwargs["player"] != "P1P2":
            print("stageP1 = ", othersP1[4])
    else:
        print("ownHealth1P1 = ", othersP1[0])
        print("ownHealth2P1 = ", othersP1[1])
        print("oppHealth1P1 = ", othersP1[2])
        print("oppHealth2P1 = ", othersP1[3])
        print("ownActiveCharP1 = ", othersP1[4])
        print("oppActiveCharP1 = ", othersP1[5])
        print("ownPositionP1 = ", othersP1[6])
        print("oppPositionP1 = ", othersP1[7])
        if diambraKwargs["player"] != "P1P2":
            print("stageP1 = ", othersP1[8])                                       
    print("ownCharP1 = ", env.charNames[list(othersP1[nScalarAddParP1:           
                                                      nScalarAddParP1 + nChars]).index(1.0)])
    print("oppCharP1 = ", env.charNames[list(othersP1[nScalarAddParP1 + nChars:           
                                                      nScalarAddParP1 + 2*nChars]).index(1.0)])

    # 2P
    if diambraKwargs["player"] == "P1P2":
        
        additionalParP2 = int(observation[int(shp[0]/2),0,shp[2]-1])                                                                        
        nScalarAddParP2 = additionalParP2 - 2*nChars - env.actBufLen*(env.nActions[1][0]+env.nActions[1][1])
        
        print("Additional Par P2 =", additionalParP1)                                
        print("N scalar actions P2 =", nScalarAddParP1)  
        
        addParP2 = addPar[int((shp[0]*shp[1])/2)+1:int((shp[0]*shp[1])/2)+additionalParP2+1]
        actionsP2 = addParP2[0:additionalParP2-nScalarAddParP2-2*nChars]
            
        moveActionsP2   = actionsP2[0:limAct[1][0]]
        attackActionsP2 = actionsP2[limAct[1][0]:limAct[1][1]]
        moveActionsP2   = np.reshape(moveActionsP2, (env.actBufLen,-1))
        attackActionsP2 = np.reshape(attackActionsP2, (env.actBufLen,-1))
        print("Move actions P2 =\n", moveActionsP2)
        print("Attack actions P2 =\n", attackActionsP2)
        
        othersP2 = addParP2[additionalParP2-nScalarAddParP2-2*env.numberOfCharacters:]
        if gameId != "tektagt":
            print("ownHealthP2 = ", othersP2[0])
            print("oppHealthP2 = ", othersP2[1])
            print("ownPositionP2 = ", othersP2[2])
            print("oppPositionP2 = ", othersP2[3])
        else:
            print("ownHealth1P2 = ", othersP2[0])
            print("ownHealth2P2 = ", othersP2[1])
            print("oppHealth1P2 = ", othersP2[2])
            print("oppHealth2P2 = ", othersP2[3])
            print("ownActiveCharP2 = ", othersP2[4])
            print("oppActiveCharP2 = ", othersP2[5])
            print("ownPositionP2 = ", othersP2[6])
            print("oppPositionP2 = ", othersP2[7])
    print("ownCharP2 = ", env.charNames[list(othersP2[nScalarAddParP2:           
                                                      nScalarAddParP2 + nChars]).index(1.0)])
    print("oppCharP2 = ", env.charNames[list(othersP2[nScalarAddParP2 + nChars:           
                                                      nScalarAddParP2 + 2*nChars]).index(1.0)])    
    
    obs = np.array(observation).astype(np.float32)
    
    for idx in range(shp[2]-1):
        cv2.imshow("image"+str(idx), obs[:,:,idx])
    
    cv2.waitKey(waitKey)


# In[ ]:


actionsPrintDict = env.printActionsDict()
observation = env.reset()


# In[ ]:


showObs(observation, limAct)


# In[ ]:


cumulativeEpRew = 0.0
cumulativeEpRewAll = []

maxNumEp = 100
currNumEp = 0

while currNumEp < maxNumEp:

    actions = [None, None]
    if diambraKwargs["player"] != "P1P2":
        actions = env.action_space.sample()
        
        if diambraGymKwargs["actionSpace"] == "discrete":
            moveAction, attAction = discreteToMultiDiscreteAction(actions, env.nActions[0])
        else:
            moveAction, attAction = actions[0], actions[1]
            
        print("(P1) {} {}".format(actionsPrintDict[0][moveAction],       
                                  actionsPrintDict[1][attAction])) 
        
    else:
        
        for idx in range(2):
        
            actions[idx] = env.action_space["P{}".format(idx+1)].sample()

            if diambraGymKwargs["actionSpace"][idx] == "discrete":
                moveAction, attAction = discreteToMultiDiscreteAction(actions[idx], env.nActions[idx][0])
            else:
                moveAction, attAction = actions[idx][0], actions[idx][1]
        
            if diambraKwargs["player"] != "P1P2" and idx == 1:
                continue
            
            print("(P{}) {} {}".format(idx+1, actionsPrintDict[0][moveAction],       
                                              actionsPrintDict[1][attAction])) 
        
    if diambraKwargs["player"] == "P1P2" or diambraGymKwargs["actionSpace"] != "discrete":
        actions = np.append(actions[0], actions[1])   
    
    observation, reward, done, info = env.step(actions)
    
    print("action = ", actions)
    print("reward:", reward)
    print("done = ", done)
    for k, v in info.items():
        print("info[\"{}\"] = {}".format(k, v))
    showObs(observation, limAct, 0)
        
    print("----------")
    
    cumulativeEpRew += reward
    
    if np.any(done):
        currNumEp += 1
        print("Ep. # = ", currNumEp)
        print("Ep. Cumulative Rew # = ", cumulativeEpRew)
        cumulativeEpRewAll.append(cumulativeEpRew)
        cumulativeEpRew = 0.0

        observation = env.reset()
        showObs(observation, limAct, 0)

print("Mean cumulative reward = ", np.mean(cumulativeEpRewAll))    
print("Std cumulative reward = ", np.std(cumulativeEpRewAll))       
    
env.close()


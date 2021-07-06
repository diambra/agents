import cv2, sys, os, time
import numpy as np

# Visualize Obs content
def showObs(observation, keyToAdd, keyToAddCount, actBufLen, nActions, waitKey, viz, charList, hardCore, idxList):

    if not hardCore:
        shp = observation.shape
        for idx in idxList:
            addPar = observation[:,:,shp[2]-1]
            addPar = np.reshape(addPar, (-1))

            counter = 0 + idx*int( (shp[0]*shp[1]) / 2 )

            print("Additional Par P{} =".format(idx+1), addPar[counter])

            counter += 1

            for idK in range(len(keyToAdd)):

                var = addPar[counter:counter+keyToAddCount[idK][idx]] if keyToAddCount[idK][idx] > 1 else addPar[counter]
                counter += keyToAddCount[idK][idx]

                if "actions" in keyToAdd[idK]:

                    moveActions   = var[0:actBufLen*nActions[idx][0]]
                    attackActions = var[actBufLen*nActions[idx][0]:actBufLen*(nActions[idx][0]+nActions[idx][1])]
                    moveActions   = np.reshape(moveActions, (actBufLen,-1))
                    attackActions = np.reshape(attackActions, (actBufLen,-1))
                    print("Move actions P{} =\n".format(idx+1), moveActions)
                    print("Attack actions P{} =\n ".format(idx+1), attackActions)
                elif "Char" in keyToAdd[idK]:
                    print("{}P{} =".format(keyToAdd[idK], idx+1), charList[list(var).index(1.0)])
                else:
                    print("{}P{} =".format(keyToAdd[idK], idx+1), var)

        if viz:
            obs = np.array(observation[:,:,0:shp[2]-1]).astype(np.float32)
    else:
        if viz:
            obs = np.array(observation).astype(np.float32)

    if viz:
        for idx in range(obs.shape[2]):
            cv2.imshow("image"+str(idx), obs[:,:,idx])

        cv2.waitKey(waitKey)

# Util to copy P2 additional OBS into P1 position on last (add info dedicated) channel
def P2ToP1AddObsMove(observation):
    shp = observation.shape
    startIdx = int((shp[0]*shp[1])/2)
    observation = np.reshape(observation, (-1))
    numAddParP2 = int(observation[startIdx])
    addParP2 = observation[startIdx:startIdx+numAddParP2 + 1]
    observation[0:numAddParP2 + 1] = addParP2
    observation = np.reshape(observation, (shp[0], -1))
    return observation

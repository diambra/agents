# DIAMBRA Gym base class for two players mode with gamePad on P2
class diambraMameHardCore2PvsRL(diambraMameHardCore2P):
    def __init__(self, envId, diambraKwargs, P2brain, rewNormFac=0.5,
                 actionSpace=["multiDiscrete", "multiDiscrete"],
                 attackButCombinations=[True, True], actBufLen=12,
                 headless=False, displayNum=1):
        super.__init__( envId, diambraKwargs, rewNormFac, actionSpace,
                        attackButCombinations, actBufLen, headless, displayNum)

        # P2 action logic
        self.p2Brain = P2brain

    # Save last Observation
    def updateLastObs(self, obs):
        self.lastObs = obs

    # Update P2Brain RL policy weights
    def updateP2BrainWeights(self, weightsPath):
        self.p2Brain.updateWeights(weightsPath)

    # Step the environment
    def step(self, action):

        # Actions initialization
        movActP1 = 0
        attActP1 = 0
        movActP2 = 0
        attActP2 = 0

        # Defining move and attack actions P1/P2 as a function of actionSpace
        if self.actionSpace[0] == "multiDiscrete": # P1 MultiDiscrete Action Space
            # P1
            movActP1 = action[0]
            attActP1 = action[1]
            # P2
            if self.actionSpace[1] == "multiDiscrete": # P2 MultiDiscrete Action Space
                self.lastObs[:,:,-1] = P2ToP1AddObsMove(self.lastObs[:,:,-1])
                [movActP2, attActP2], _ = self.p2Brain.act(self.lastObs)
            else: # P2 Discrete Action Space
                self.lastObs[:,:,-1] = P2ToP1AddObsMove(self.lastObs[:,:,-1])
                brainActions, _ = self.p2Brain.act(self.lastObs)
                movActP2, attActP2 = self.discreteToMultiDiscreteAction(brainActions)

        else: # P1 Discrete Action Space
            # P2
            if self.actionSpace[1] == "multiDiscrete": # P2 MultiDiscrete Action Space
                # P1
                # Discrete to multidiscrete conversion
                movActP1, attActP1 = self.discreteToMultiDiscreteAction(action)
                self.lastObs[:,:,-1] = P2ToP1AddObsMove(self.lastObs[:,:,-1])
                [movActP2, attActP2], _ = self.p2Brain.act(self.lastObs)
            else: # P2 Discrete Action Space
                # P1
                # Discrete to multidiscrete conversion
                movActP1, attActP1 = self.discreteToMultiDiscreteAction(action)
                self.lastObs[:,:,-1] = P2ToP1AddObsMove(self.lastObs[:,:,-1])
                brainActions, _ = self.p2Brain.act(self.lastObs)
                movActP2, attActP2 = self.discreteToMultiDiscreteAction(brainActions)

        observation, reward, roundDone, done, self.internalInfo =\
            self.env.step2P(movActP1, attActP1, movActP2, attActP2)

        # Extend the actions buffer
        self.movActBuf[0].extend([movActP1])
        self.attActBuf[0].extend([attActP1])
        self.movActBuf[1].extend([movActP2])
        self.attActBuf[1].extend([attActP2])

        # Perform post step processing
        observation, info, done = self.stepPostProc(observation, roundDone,
                                                    stageDone=False, gameDone=done, done)

        return observation, reward, done, info

# DIAMBRA Gym base class for two players mode with HUM on P2
class diambraMameHardCore2PvsHum(diambraMameHardCore2P):
    def __init__(self, envId, diambraKwargs, P2brain, rewNormFac=0.5,
                 actionSpace=["multiDiscrete", "multiDiscrete"],
                 attackButCombinations=[True, True], actBufLen=12,
                 headless=False, displayNum=1):
        super.__init__( envId, diambraKwargs, rewNormFac, actionSpace,
                        attackButCombinations, actBufLen, headless, displayNum)

        # P2 action logic
        self.p2Brain = P2brain
        # If p2 action logic is gamepad, add it to self.gamepads (for char selection)
        # Check action space is prescribed as "multiDiscrete"
        self.p2Brain.initialize(self.env.actionList())
        if self.actionsSpace[1] != "multiDiscrete":
            raise Exception("Action Space for P2 must be \"multiDiscrete\" when using gamePad")
        if not self.attackButCombinations[1]:
            raise Exception("Use attack buttons combinations for P2 must be \"True\" when using gamePad")

    # Step the environment
    def step(self, action):

        # Actions initialization
        movActP1 = 0
        attActP1 = 0
        movActP2 = 0
        attActP2 = 0

        # Defining move and attack actions P1/P2 as a function of actionSpace
        if self.actionSpace[0] == "multiDiscrete": # P1 MultiDiscrete Action Space
            # P1
            movActP1 = action[0]
            attActP1 = action[1]
        else: # P1 Discrete Action Space
            # P1
            # Discrete to multidiscrete conversion
            movActP1, attActP1 = self.discreteToMultiDiscreteAction(action)

        # P2
        [movActP2, attActP2], _ = self.p2Brain.act()


        observation, reward, roundDone, done, self.internalInfo =\
            self.env.step2P(movActP1, attActP1, movActP2, attActP2)

        # Extend the actions buffer
        self.movActBuf[0].extend([movActP1])
        self.attActBuf[0].extend([attActP1])
        self.movActBuf[1].extend([movActP2])
        self.attActBuf[1].extend([attActP2])

        # Perform post step processing
        observation, info, done = self.stepPostProc(observation, roundDone,
                                                    stageDone=False, gameDone=done, done)

        return observation, reward, done, info

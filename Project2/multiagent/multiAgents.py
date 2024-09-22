# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPosition = successorGameState.getGhostPositions()
        
        successorScore = 0
        newX, newY = newPos
        
        if newPos == currentGameState.getPacmanPosition():
            successorScore -= 100
        
        def caculateDistance(pos):
            return abs(pos[0] - newX) + abs(pos[1] - newY)
        
        for food in newFood.asList():
            if food == newPos:
                successorScore += 20
            distance = caculateDistance(food)
            successorScore += 1 / (distance + 0.1)
        
        for index, ghostPos in enumerate(newGhostPosition):
            distance = caculateDistance(ghostPos)
            if not newScaredTimes[index]:
                successorScore -= 50 / (distance + 0.1)
        
        return successorScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        # # 极小化极大函数
        # def minimax(gameState: GameState, agentIndex=0, depth=0):
        #     # 边界条件判断：游戏结束或到达设定深度
        #     if gameState.isWin() or gameState.isLose() or depth == self.depth:
        #         return self.evaluationFunction(gameState), None
            
        #     # 递归调用过程
        #     values = [] # 二元组, (极小化极大值, 该次动作)
        #     legalActions = gameState.getLegalActions(agentIndex)
        #     for legalAction in legalActions:
        #         nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()  # 下一个行动的 agent
        #         nextState = gameState.generateSuccessor(agentIndex, legalAction)
        #         nextDepth = depth + 1 if agentIndex == gameState.getNumAgents() - 1 else depth  # 下一次的深度, 仅当走完一整轮时才增加
        #         values.append((minimax(nextState, nextAgentIndex, nextDepth)[0], legalAction))
        #     # 根据当前行动者返回极大或极小值
        #     return max(values, key=lambda x: x[0]) if agentIndex == 0 else min(values, key=lambda x: x[0])
        
        # maxValue = minimax(gameState)
        # return maxValue[1]
        def minimax(gameState: GameState, agentIndex=0, depth=0):
            # 边界条件: 游戏结束或达到最大深度
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # 递归调用
            optimalUtility = -float('inf') if agentIndex == 0 else float('inf') # 初始化最佳效用
            optimalAction = None    # 用于存储最佳动作, 当为根节点时直接返回
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgentIndex == 0 else depth # 如果轮回吃豆人, 说明深度增加
                currentUtility = minimax(successorState, nextAgentIndex, nextDepth)
                # index 为 0 时极大化, 否则极小化
                if agentIndex == 0:
                    if currentUtility > optimalUtility:
                        optimalUtility = currentUtility
                        optimalAction = action
                else:
                    optimalUtility = min(currentUtility, optimalUtility)
            # 如果为根节点, 直接返回动作, 否则返回效用值
            return optimalAction if depth == 0 and agentIndex == 0 else optimalUtility
        
        return minimax(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # 整体跟极小极大化一致, 仅加入对 alpha 和 beta 值的跟踪以及相应的剪枝操作
        def alphaBeta(gameState: GameState, agentIndex=0, depth=0, alpha=-float('inf'), beta=float('inf')):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            optimalUtility = -float('inf') if agentIndex == 0 else float('inf')
            optimalAction = None
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgentIndex == 0 else depth
                currentUtility = alphaBeta(successorState, nextAgentIndex, nextDepth, alpha, beta)
                if agentIndex == 0:
                    if currentUtility > optimalUtility:
                        optimalUtility = currentUtility
                        optimalAction = action
                    alpha = max(alpha, currentUtility)
                else:
                    optimalUtility = min(currentUtility, optimalUtility)
                    beta = min(beta, currentUtility)
                # 剪枝
                if beta < alpha:
                    break
            return optimalAction if depth == 0 and agentIndex == 0 else optimalUtility

        return alphaBeta(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(gameState: GameState, agentIndex=0, depth=0):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            finalUtility = -float('inf') if agentIndex == 0 else 0
            optimalAction = None
            # 遍历动作, 如果是吃豆人就选最优, 否则加总取平均值
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgentIndex == 0 else depth
                currentUtility = expectimax(successorState, nextAgentIndex, nextDepth)
                if agentIndex == 0:
                    if currentUtility > finalUtility:
                        finalUtility = currentUtility
                        optimalAction = action
                else:
                    finalUtility += currentUtility
            if agentIndex == 0:
                return finalUtility if depth else optimalAction
            else:
                return finalUtility / len(gameState.getLegalActions(agentIndex))
        
        return expectimax(gameState)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    currentPos = currentGameState.getPacmanPosition()   # 吃豆人的位置
    score = currentGameState.getScore() # 用自带的 getScore 作为底分
    # 根据药丸的位置加分
    if currentGameState.getCapsules():
        for capsule in currentGameState.getCapsules():
            score += 2 / util.manhattanDistance(currentPos, capsule)
        score -= len(currentGameState.getCapsules()) * 100
    else:
        # 根据食物距离加分
        for food in currentGameState.getFood().asList():
            score += 2 / util.manhattanDistance(currentPos, food)
    # 根据鬼的距离减分
    for i, ghost in enumerate(currentGameState.getGhostStates()):
        ghostPos = currentGameState.getGhostPosition(i + 1) # 鬼的编号从 1 开始
        distance = util.manhattanDistance(currentPos, ghostPos)
        if not ghost.scaredTimer:
            # 如果距离过近则大量扣分
            if distance < 2:
                score -= 100
            else:
                score -= 1 / (distance + 0.1) 
    print(score)
    return score


# Abbreviation
better = betterEvaluationFunction

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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        pos = currentGameState.getPacmanPosition()
        food = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Calculate the manhattan distance of all food from new position
        minfdist = -1
        for elem in food.asList():
            dist = util.manhattanDistance(newPos, elem)
            if (minfdist == -1):
                minfdist = dist
            elif (minfdist > dist):
                minfdist = dist

        gs = successorGameState.getGhostPositions()     # Get the current ghost position

        # Calculate the manhattan distance of all the ghosts from new Pacman position, taking the minimum out of them
        mingdist = -1
        for elem in gs:
            dist = util.manhattanDistance(newPos, elem)
            if (mingdist == -1):
                mingdist = dist
            elif (mingdist > dist):
                mingdist = dist

        #If ghost is more than 2 moves away, then Pacman will run for food since there's no issue of ghost
        if (mingdist > 2):     # if ghost is more than 2 moves away, return food distance, else return ghost distance
            if (action == "Stop"):
                return float("-inf")
            result = 1000 - minfdist
        else:
        #If ghost is less than 2 moves away, then Pacman will run away from ghost since there's chance of getting died
            result = mingdist

        return result

def scoreEvaluationFunction(currentGameState):
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
    #tlevel is total number of levels in minimax tree
    #clevel is current level in minimax tree
    def minimax(self, gameState, clevel, tlevel):

        if(clevel == tlevel):       # check if current level is top level in tree, return the eval function
            return self.evaluationFunction(gameState)

        agent = clevel % gameState.getNumAgents() # check the current agent index, 0 is Pacman, >0 is Ghosts
        legalMoves = gameState.getLegalActions(agent) # Get all the legal moves for the current agent

        #if we are processing pacman, start with minimum value and try to maximize it
        if(agent == 0):
            fvalue = float("-inf")
        # if we are processing ghost, start with maximum value and try to maximize it
        else:
            fvalue = float("inf")

        if(len(legalMoves) == 0):   # if no legal moves for current state, return the eval function
            return self.evaluationFunction(gameState)

        for action in legalMoves:   # for all the legal moves, generate successors and call minimax recursively
            successorGameState = gameState.generateSuccessor(agent, action)
            res = self.minimax(successorGameState, clevel + 1, tlevel)
            if(agent == 0):    # for Pacman move maximise the fvalue, else minimize the fvalue
                if(res > fvalue):
                    fvalue = res
                    raction = action
            else:
                if(res < fvalue):
                    fvalue = res

        if(clevel == 0):
            return raction
        return fvalue

    def getAction(self, gameState):
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
        """
        # call the minimax function for pacman index and current gamestate
        start_level = 0
        level = self.depth * gameState.getNumAgents()
        raction = self.minimax(gameState, start_level, level)
        return raction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphabeta(self, gameState, clevel, tlevel, alpha, beta):

        # tlevel is total number of levels in minimax tree
        # clevel is current level in minimax tree
        if (clevel == tlevel):   # check if current level is top level in tree, return the eval function
            return self.evaluationFunction(gameState)

        agent = clevel % gameState.getNumAgents()   # check the current agent index, 0 is Pacman, >0 is Ghosts
        legalMoves = gameState.getLegalActions(agent)    # Get all the legal moves for the current agent

        if (agent == 0):
            fvalue = float("-inf")
        else:
            fvalue = float("inf")

        if (len(legalMoves) == 0):  # if no legal moves for current state, return the eval function
            return self.evaluationFunction(gameState)

        for action in legalMoves:   # for all the legal moves, generate successors and call alphabeta recursively
            successorGameState = gameState.generateSuccessor(agent, action)
            res = self.alphabeta(successorGameState, clevel + 1, tlevel, alpha, beta)
            if (agent == 0):     # for Pacman move maximize the fvalue and assign to alpha
                if (res > fvalue):
                    fvalue = res
                    raction = action

                #Updating alpha value at pacman node
                if(alpha < fvalue):
                    alpha = fvalue
            else:               # for ghosts move minimise the fvalue and assign to beta
                if (res < fvalue):
                    fvalue = res

                #Updating beta value at ghost node
                if(beta > fvalue):
                    beta = fvalue

            #Pruning action when alpha is greater than beta
            if(alpha > beta):   # if alpha > beta, do not expand further successors
                break

        if (clevel == 0):
            return raction
        return fvalue

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # call the alphabeta function
        start_level = 0
        tlevel = self.depth * gameState.getNumAgents()
        raction = self.alphabeta(gameState, start_level, tlevel, float("-inf"), float("inf"))
        return raction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    #tlevel is total number of levels in tree
    #clevel is current level in tree
    def expectimax(self, gameState, clevel, tlevel):

        if (clevel == tlevel):
            return self.evaluationFunction(gameState)

        agent = clevel % gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agent)

        if (agent == 0):
            fvalue = float("-inf")
        else:
            fvalue = 0.0

        if (len(legalMoves) == 0):
            return self.evaluationFunction(gameState)

        #Intialize the count of successors from 0
        count = 0
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agent, action)
            res = self.expectimax(successorGameState, clevel + 1, tlevel)
            #Storing the action value in case of Max agent
            if (agent == 0):
                if (res > fvalue):
                    fvalue = res
                    raction = action
            else:       # for ghosts move, add all successor values and keep count of no. of successors
                count += 1
                fvalue += res

        if(agent > 0):      # For ghosts move, take the average of the values instead of the minimum value.
            fvalue = float(fvalue) / float(count)

        #Completed all recursions and reached starting level
        if (clevel == 0):
            return raction
        #Returning the numerical value corresponding to children
        return fvalue

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        start_level = 0
        tlevel = self.depth * gameState.getNumAgents()
        raction = self.expectimax(gameState, start_level, tlevel)
        return raction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


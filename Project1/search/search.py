# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    # 初始化 frontier 和 reached
    start_state = problem.getStartState()
    frontier = util.Stack()
    frontier.push((start_state, None, None)) # 以三元组 (state, parent, action) 的形式建立 node
    reached = set()
    
    # 开始搜索
    while not frontier.isEmpty():
        current_state, parent_node, action = frontier.pop() # 解包三元组
        print(action)
        reached.add(current_state)
        
        # 检查是否找到 goal,找到则生成 action 列表
        if problem.isGoalState(current_state):
            actions = []
            while parent_node:
                actions.append(action)
                current_state, parent_node, action = parent_node
            actions.reverse()
            return actions
        
        # 不是 goal 则遍历 successors
        successors = problem.getSuccessors(current_state)
        for successor_state, successor_action, _ in successors:
            if successor_state not in reached:
                new_node = (successor_state, (current_state, parent_node, action), successor_action)
                frontier.push(new_node)
                
    return None

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    
    # 整体跟深搜一致, 仅将 Stack 换为 Queue, 并在节点出边界时进行检查
    start_state = problem.getStartState()
    frontier = util.Queue()
    frontier.push((start_state, None, None)) 
    reached = set()
    
    while not frontier.isEmpty():
        current_state, parent_node, action = frontier.pop() 
        # 额外检查是否探索过该状态, 因为广搜的顺序可能导致从多个状态同时进入一个状态
        if current_state in reached:
            continue
        reached.add(current_state)

        if problem.isGoalState(current_state):
            actions = []
            while parent_node:
                actions.append(action)
                current_state, parent_node, action = parent_node
            actions.reverse()
            return actions
        
        successors = problem.getSuccessors(current_state)
        for successor_state, successor_action, _ in successors:
            if successor_state not in reached:
                new_node = (successor_state, (current_state, parent_node, action), successor_action)
                frontier.push(new_node)
                
    return None

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    
    # 定义节点, 便于传入函数
    class Node():
        def __init__(self, state, parent, action):
            self.state = state
            self.parent = parent
            self.action = action
    
    # 根据当前节点反推路径
    def generatePath(node):
        actions = []
        while node.parent:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions
    
    # 根据当前节点获取当前路径的 cost, 用于传递给 PriorityQueueWithFunction
    def getCost(node):
        actions = generatePath(node)
        cost = problem.getCostOfActions(actions)
        return cost
    
    # 剩余整体跟广搜一致, 将 Stack 换为 PriorityQueue
    start_state = problem.getStartState()
    frontier = util.PriorityQueueWithFunction(getCost)
    frontier.push(Node(start_state, None, None)) 
    reached = set()
    
    while not frontier.isEmpty():
        current_node = frontier.pop() 
        if current_node.state in reached:
            continue
        reached.add(current_node.state)

        if problem.isGoalState(current_node.state):
            actions = generatePath(current_node)
            return actions
        
        successors = problem.getSuccessors(current_node.state)
        for successor_state, successor_action, _ in successors:
            if successor_state not in reached:
                frontier.push(Node(successor_state, current_node, successor_action))
                
    return None

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    
    class Node():
        def __init__(self, state, parent, action, cost):
            self.state = state
            self.parent = parent
            self.action = action
            self.cost = cost
    
    def generatePath(node):
        actions = []
        while node.parent:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions
    
    # 计算当前节点的 f 函数值
    def getF(node):
        return node.cost + heuristic(node.state, problem)
    
    start_state = problem.getStartState()
    frontier = util.PriorityQueueWithFunction(getF) # 优先级函数为 f
    frontier.push(Node(start_state, None, None, 0)) 
    lowest_costs = {start_state: 0} # 不需要维护 reached 集合, 建立字典记录到达某个状态的最低 cost
    
    while not frontier.isEmpty():
        current_node = frontier.pop() 

        if problem.isGoalState(current_node.state):
            return generatePath(current_node)
        
        successors = problem.getSuccessors(current_node.state)
        for successor_state, successor_action, step_cost in successors:
            new_cost = current_node.cost + step_cost
            # 仅当后继为 cost 更低的路径时将节点加入
            if successor_state not in lowest_costs.keys() or new_cost < lowest_costs[successor_state]:
                lowest_costs[successor_state] = new_cost
                frontier.push(Node(successor_state, current_node, successor_action, new_cost))
                    
    return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

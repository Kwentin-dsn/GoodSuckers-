# baseline_team.py
# ---------------
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random

from sympy import false

from contest.capture_agents import CaptureAgent
from contest.game import Directions

import contest.util as util
from contest.util import nearest_point



#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class TeamData:
    def __init__(self):
        self.enemy_pos_offense = {} # last known ghost positions
        self.enemy_pos_defense = {}
        self.enemy_scared_timers = {}
        self.food_eaten_positions = []
team_communication = TeamData()

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.path = []
        self.path_counter = 0
        global team_communication
        self.team_communication = team_communication
        self.midpoint = None



    def update_team_data(self, game_state):
        """Update shared team info"""
        line_of_defense = self.midpoint - (1 if self.red else 0)

        for enemy_index in self.get_opponents(game_state):
            enemy_state = game_state.get_agent_state(enemy_index)
            enemy_pos = enemy_state.get_position()

            if enemy_pos is not None:
                #Update timer when ghost visible
                self.team_communication.enemy_scared_timers[enemy_index] = enemy_state.scared_timer
                # Enemy in defense or not
                if self.red:
                    Is_in_defensezone = enemy_pos[0] < line_of_defense
                else:
                    Is_in_defensezone = enemy_pos[0] >= line_of_defense

                #Update shared knowledge
                self.team_communication.enemy_pos_defense[enemy_index] = enemy_pos if Is_in_defensezone else None
                self.team_communication.enemy_pos_offense[enemy_index] = enemy_pos if not Is_in_defensezone else None
            else:
                if self.team_communication.enemy_scared_timers[enemy_index]:
                    self.team_communication.enemy_scared_timers[enemy_index] -= 1
                else:
                    self.team_communication.enemy_scared_timers[enemy_index] = None


            if game_state.get_agent_position(enemy_index) ==  game_state.get_agent_state(self.index).get_position():
                self.team_communication.enemy_pos_offense[enemy_index] = None
                self.team_communication.enemy_pos_defense[enemy_index] = None

    def optimal_path(self, state):
        """
        Returns the optimal path to the closest powerpellet
        """
        my_pos = state.get_agent_state(self.index).get_position()
        pellets = self.get_capsules(state)
        if not pellets:
            return []

        min_distance = float('inf')
        best_pellet = None

        for p in pellets:
            distance = self.get_maze_distance(my_pos, p)
            if distance < min_distance:
                min_distance = distance
                best_pellet = p

        agenda = util.PriorityQueue()
        agenda.push((state, [], 0), self.get_maze_distance(my_pos, best_pellet))
        bezochte_states = []

        while not agenda.is_empty():
            current = agenda.pop()
            search_state, path, cost = current
            if search_state.get_agent_state(self.index).get_position() == best_pellet:
                return path
            else:
                if not (search_state in bezochte_states):
                    bezochte_states.append(search_state)
                    new_actions =  search_state.get_legal_actions(self.index)
                    for a in new_actions:
                        new_state = self.get_successor(search_state, a)
                        new_path = path + [(search_state, a)]
                        new_cost = cost + 1
                        new_heuristic = new_cost + self.get_maze_distance(new_state.get_agent_state(self.index).get_position(), best_pellet)
                        agenda.push((new_state, new_path, new_cost), new_heuristic)
        return []


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.path = self.optimal_path(game_state)
        self.midpoint = game_state.data.layout.width // 2
        for i in self.get_opponents(game_state):
            self.team_communication.enemy_pos_offense[i] = None
            self.team_communication.enemy_pos_defense[i] = None
            self.team_communication.enemy_scared_timers[i] = 0

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.update_team_data(game_state)

        actions = game_state.get_legal_actions(self.index)

        # ignore stop action
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        if self.path and self.path_counter <= len(self.path):
            action = self.path[self.path_counter]
            if action in actions:
                self.path_counter += 1
                return action

        return random.choice(best_actions)



    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        # create dictionary, get successor state and position, and food list
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_state = successor.get_agent_state(self.index)
        my_pos = successor.get_agent_position(self.index)

        # get successor score depending on team, with the food score
        features['successor_score'] = self.get_score(successor) if self.red else -self.get_score(successor)
        features['rest_food'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True, but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # current eaten food
        features['eaten_food']= my_state.num_carrying

        # get enemy states
        enemy_indices = self.get_opponents(successor)
        enemies = [successor.get_agent_state(i) for i in enemy_indices]

        # get defenders who we can see
        defender_positions = []
        for enemy in enemies:
            if not enemy.is_pacman and enemy.get_position() is not None:
                defender_positions.append(enemy.get_position())

        # Check team data for additional defenders
        if len(defender_positions) == 0:  # Only use team data if we don't see any defenders directly
            for enemy_idx in enemy_indices:
                pos = self.team_communication.enemy_pos_offense[enemy_idx]
                if pos is not None:
                    # Check if position is on opponent's side (defender)
                    """CHANGE"""
                    # on_opponent_side = (self.red and pos[0] >= self.midpoint) or (not self.red and pos[0] < self.midpoint)
                    # if on_opponent_side:
                    defender_positions.append(pos)

        features['num_defenders'] = len(defender_positions)
        features['num_attackers'] = 2 - len(defender_positions)

        # Calculate distance to defenders
        if len(defender_positions) > 0:
            dists = [self.get_maze_distance(my_pos, pos) for pos in defender_positions]
            features['invader_distance'] = min(dists)
        elif len(defender_positions) == 0:

            # if no defenders visible, check the noisy distance
            dists = [game_state.agent_distances[i] for i in self.get_opponents(successor)]
            features['invader_distance'] = min(dists) if dists else 0
            if features['invader_distance'] <= 0:
                features['invader_distance'] = 1

        # Check if enemies are scared
        enemy_scared = False
        for timer in [a.scared_timer for a in enemies]:
            if timer >3:
                enemy_scared = True
                break
        # if enemy scared, reward to go closer
        if enemy_scared:
            features['invader_distance']= - features['invader_distance']
        else:
            features['invader_distance'] = 5 * features['invader_distance']

        # prevents running into ghosts
        if len(defender_positions) > 0:
            min_ghost_distance = min([self.get_maze_distance(my_pos, pos) for pos in defender_positions])
            if min_ghost_distance < 2 and not enemy_scared:
                features['ghost_danger'] = 1
            else:
                features['ghost_danger'] = 0

        # closest power pellet
        my_capsules = self.get_capsules(successor)
        if len(my_capsules) >0:
            dists = [self.get_maze_distance(my_pos, p) for p in my_capsules]
            features['powerPellet']= min(dists)

        # prevents reversing
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        boundary_x = self.midpoint - (1 if self.red else 0)
        valid_boundary_points = []

        # Find valid positions along the boundary line
        for y in range(1, game_state.data.layout.height - 1):
            # Check if the position is a valid (not a wall)
            if not game_state.has_wall(boundary_x, y):
                valid_boundary_points.append((boundary_x, y))

        # If we found valid boundary points, calculate the minimum distance
        if valid_boundary_points:
            way_to_border = min([self.get_maze_distance(my_pos, point) for point in valid_boundary_points])
            features["return_distance"] = way_to_border
        else:
            # Fallback if no valid points found
            features["return_distance"] = abs(my_pos[0] - boundary_x) * 2  # Simple heuristic



        return features



    def get_weights(self, game_state, action):
        features = self.get_features(game_state,action)
        if features['eaten_food']<5:
            return {'successor_score': 5,
                    'rest_food': 50,
                'distance_to_food': -0.4,
                'eaten_food': 10,
                'invader_distance': -50,
                'num_defenders': -10,
                'num_attackers':10,
                'powerPellet':-2,
                'reverse': -2,
                'return_distance':-0.01,
                'ghost_danger': -1000} #NEW Strongly avoid dangerous ghost positions

        elif features['eaten_food'] > 10:
            return {'successor_score': 20,
                    'rest_food': 1,
                    'distance_to_food': -0.002,
                    'eaten_food': 2,
                    'invader_distance': -100,
                    'num_defenders': -40,
                    'num_attackers': 10,
                    'powerPellet': 0,
                    'reverse': -20,
                    'return_distance': -30,
                    'ghost_danger': -2000}  #NEW Strongly avoid dangerous ghost positions

        else:
            return {'successor_score': 10,
                    'rest_food': 30,
                    'distance_to_food': -0.4,
                    'eaten_food': 2,
                    'invader_distance': -80,
                    'num_defenders': -5,
                    'num_attackers': 10,
                    'powerPellet': -4,
                    'reverse': -20,
                    'return_distance': -5,
                    'ghost_danger': -1500} #NEW Strongly avoid dangerous ghost positions




class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # update common knowledge
        self.update_team_data(game_state)

        # middle point
        boundary_x = self.midpoint - 1 if self.red else self.midpoint + 1
        # position of agent
        pos = game_state.get_agent_state(self.index).get_position()

        # reset path counter to take optimal path again when killed
        if game_state.get_agent_state(self.index).get_position() == self.start:
            self.path_counter = 0
        actions = game_state.get_legal_actions(self.index)

        # ignore stop action
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]

        # get actions with highest value
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # optimal path stops at boundary
        if pos[0] == boundary_x:
            self.path_counter = len(self.path) + 1

        # get the actions for the optimal path
        if self.path and self.path_counter <= len(self.path):
            action = self.path[self.path_counter][1]
            if action in actions:
                self.path_counter += 1
                return action
        # choose one of the best actions
        return random.choice(best_actions)


    def get_features(self, game_state, action):
        # initialize dictionary, successor state and position
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # rewards if agent is defending
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # get all the states of enemy agents
        enemies = [(successor.get_agent_state(i),i) for i in self.get_opponents(successor)]
        invader_positions = []

        # calculate distance to the middle
        border_distance = abs(my_pos[0] - self.midpoint)

        # Get enemies we can directly see
        for enemy,_ in enemies:
            invader_pos = enemy.get_position()
            if enemy.is_pacman and invader_pos is not None:
                invader_positions.append(invader_pos)
        # enemies others can see on our side when we have no data
        if len(invader_positions) == 0:
            for _, enemy_idx in enemies:
                invader_pos = self.team_communication.enemy_pos_defense[enemy_idx]
                if invader_pos is not None:
                    invader_positions.append(invader_pos)

        # get minimal distance to enemy
        features['num_invaders'] = len(invader_positions)
        if len(invader_positions) > 0:
            dists = [self.get_maze_distance(my_pos, pos) for pos in invader_positions]
            features['invader_distance'] = min(dists)


        # prevents reversing
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        # food that defending if there is any left
        defending_food = self.get_food_you_are_defending(successor).as_list()
        if defending_food:
            closest_food =  min([self.get_maze_distance(my_pos, food) for food in defending_food])
        else:
            closest_food = 0

        # hold the line and defend food depending on number of invaders
        if len(invader_positions) > 0:
            features['food_defense'] = closest_food
            features['near_border'] = 0
        else:
            features['food_defense'] = 0
            features['near_border'] = border_distance

        # encourages to go closer to the enemy when defending
        if len(invader_positions) > 0:
            min_ghost_distance = min([self.get_maze_distance(my_pos, pos) for pos in invader_positions])
            if min_ghost_distance < 2 and not my_state.is_pacman:
                features['ghost_danger'] = 2
            else:
                features['ghost_danger'] = 1

        # rewards going for the kill
        features["kills"] = 1 if my_pos in invader_positions else 0

        # changes to offense when scared
        scary = my_state.scared_timer
        if scary > 0 :
            return OffensiveReflexAgent.get_features(self, game_state, action)
        return features

    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        scary = my_state.scared_timer
        if scary==0:
            return {'num_invaders': -1000,
                'on_defense': 1000,
                'invader_distance': -200,
                 'reverse': -2,
                'food_defense': -10,
                'near_border' : -30,
                    'ghost_danger': 1000,
                    'kills': 100}
        else:
            return OffensiveReflexAgent.get_weights(self, game_state, action)

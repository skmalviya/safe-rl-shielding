#!/usr/bin/python
#
# Simulates an MDP-Strategy

import math
import os
import sys, code
import resource
import copy
import itertools
import random
from PIL import Image
import os, pygame, pygame.locals
from pybrain3.rl.environments import Environment
from pybrain3.rl.environments import Task
from pybrain3.rl.agents import LearningAgent
from pybrain3.rl.experiments import Experiment
from my_pybrain.my_explorer import MyUCBExplorer
from my_pybrain.my_explorer import MyGreedyExplorer
from my_pybrain.my_table import MyActionValueTable
from my_pybrain.my_learner import MyQ, SARSA
from pybrain3.utilities import abstractMethod

from pprint import pprint

import numpy as np
from itertools import product
from copy import deepcopy
from scipy import argmax
from scipy import where
from random import choice
import importlib

# from scenario_9x9_shield_multi3 import Shield
# from cycling_enemy_shield_incl_enemy_multi3 import Shield

np.set_printoptions(threshold=np.inf)

import argparse

parser = argparse.ArgumentParser(description='Simulator')
parser.add_argument(dest="png_file_base")
parser.add_argument("-c", "--collect-data", dest="collect_data_file", help="Provide a file for collecting convergence data")
parser.add_argument('-g', "--gen-spec", dest='gen_spec', help='Generate shield files', action='store_true', default=False)
parser.add_argument('-e', "--exact-spec", dest='exact_spec', help='Generate exact shield files', action='store_true', default=False)
parser.add_argument('-l', "--load", dest='load_file', help='Load Q-Table from file')
parser.add_argument('-s', "--save", dest='save_file', help='Save Q-Table to file')
parser.add_argument('-t', "--train", dest='train', help='Training activated', type=float, default=.2)
parser.add_argument('-o', "--shield_options", dest='shield_options', help='Number of actions the shield can choose of. 0 disables the shield', type=int, default=1)
parser.add_argument('-n', "--negative-reward", dest='neg_reward', help='Indicated whether negative reward should be used for unsafe actions', action='store_true', default=False)
parser.add_argument('-p', "--huge-negative-reward", dest='huge_neg_reward', help='Indicated whether a huge negative reward should be used for unsafe actions', action='store_true', default=False)
parser.add_argument('-r', "--sarsa", dest='sarsa', help='Indicated whether to use SARSA or default Q-learning', action='store_true', default=False)
parser.add_argument("--num-steps", dest='num_steps', help='Number of interactions', type=int, default=1000000)
parser.add_argument("--alpha", dest='alpha', help='Learning rate alpha', type=float, default=1/5)
parser.add_argument("--seed", dest='seed', help='Randomseed', type=int, default=0)
parser.add_argument('-x', "--cost_per_step", dest='cost_per_step', help='Estimated cost per step',
        type=float, default=None)


args = parser.parse_args()

collect_data_file = args.collect_data_file
gen_spec = args.gen_spec
exact_spec = args.exact_spec
# Overwrite gen_spec variable as it denotes abstractive generation of spec
if gen_spec and exact_spec:
    gen_spec = False
specFile = args.png_file_base
shield_options = args.shield_options
load_file = args.load_file
save_file = args.save_file
exploration = args.train
neg_reward = args.neg_reward
huge_neg_reward = args.huge_neg_reward
alpha = args.alpha
seed = args.seed
cost_per_step = args.cost_per_step
MAX_STEPS = args.num_steps
MIN_EPS = .00
MIN_ALPHA = .00
random.seed(seed)

cooldown_epsilon = np.linspace(exploration, MIN_EPS, MAX_STEPS)
cooldown_alpha = np.linspace(alpha, MIN_ALPHA, MAX_STEPS)

danger_zone = [(7, 6), (7, 7), (7, 8), (8, 6), (8, 7), (8, 8), (9, 6), (9, 7), (9, 8)]
goal_zones = [((8,0),), ((7,7),(7,8)), ((2,7),), ((0,1),(0,2),(1,1),(1,2))]

pngfile = Image.open(specFile)
pngFileBasis = specFile[0:specFile.rfind(".png")]
path = pngFileBasis[:pngFileBasis.rfind(os.path.sep)]

# ==================================
# Settings
# ==================================
MAGNIFY = 64

# ==================================
# Read parameter file
# ==================================
parameterFileName = pngFileBasis+".params"
allParams = {}
for a in open(parameterFileName,"r").readlines():
    a = a.strip()
    if len(a)>0 and a[0]!='#':
        posEqual = a.index("=")
        allParams[a[0:posEqual].strip()] = a[posEqual+1:].strip()

# ==================================            
# Parse parameter file
# ==================================
initX = int(allParams["initX"])
initY = int(allParams["initY"])
positionUpdateNoise = float(allParams["positionUpdateNoise"])
WALL = int(allParams["wall"])
NORMAL_FIELD = int(allParams["normalField"])
NUMBER_OF_COLORS = int(allParams["numberOfColors"])

enemies_enabled = False
if "enemies" in allParams:
    try:
        mode_name = allParams["enemies"]
        mode_name = path + "." + mode_name[:mode_name.rfind(".py")]
        enemy_handler = importlib.import_module(mode_name.replace(os.path.sep, ".")).EnemyHandler()
        enemies_enabled = True
    except ImportError as e:
        print ("Could not find file " + enemy_handler_file)
        print (e)
        exit()

bombs = []
if "bombs" in allParams:
    # careful with evil evals
    bombs = eval(allParams["bombs"])
else:
    bombs = []

# ==================================
# Read input image
# ==================================

if shield_options > 0:
    try:
        mod_name = pngFileBasis + "_" + str(shield_options)
        Shield = importlib.import_module(mod_name.replace(os.path.sep, ".")).Shield
    except ImportError as e:
        print ("Could not find file " + pngFileBasis + "_" + str(shield_options) + ".py")
        print (e)
        exit()

else:
    from no_shield import Shield
    


xsize = pngfile.size[0]
ysize = pngfile.size[1]
imageData = pngfile.getdata()
palette = pngfile.getpalette()

# for i in range(len(imageData)):
#     print (imageData[i])
    
if "colorOrder" in allParams:
    colors = eval(allParams["colorOrder"])
else:
    assert(max(imageData) == NUMBER_OF_COLORS + 1)
    colors = range(max(imageData) + 1)
    colors.remove(WALL) 
    colors.remove(NORMAL_FIELD)

# ==================================
# Construct MDP --> States
# ==================================
# grid looks like this
#        --x-->
#     0 1 2 3 4 5 6 7 8
#   0
#   1
# | 2
# y 3
# | 4
# v 5
#   6
#   7
#   8

stateMapper = {}
bad_states = []
bad_state_keys = []
bad_cells = set()
for xA in range(0,xsize):
    for yA in range(0,ysize):
        for (csf,payoff) in [(x, 0) for x in range(NUMBER_OF_COLORS)] + [(0,1)]:
            if not exact_spec:
                stateNum = len(stateMapper)                    
                stateMapper[(xA,yA,csf,payoff)] = stateNum
                if (imageData[xA+yA*xsize] != WALL):
                    pass
                else:
                    if (xA, yA) not in bad_cells:
                        bad_cells.add((xA,yA))
                    bad_states.append(stateNum)
            else:
                stateNum = len(stateMapper)                    
                if (imageData[xA+yA*xsize] != WALL):
                    stateMapper[(xA,yA,csf,payoff)] = stateNum
                else:
                    if (xA, yA) not in bad_cells:
                        bad_cells.add((xA,yA))
                    bad_state_keys.append((xA,yA,csf,payoff))

BAD_STATE = len(stateMapper)
# Add error state
errorState = len(stateMapper)
errorStateKey = (-1,-1,0,0)
stateMapper[errorStateKey] = errorState

if exact_spec:
    for bs in bad_state_keys:
        stateMapper[bs] = BAD_STATE
else:
    bad_states.append(errorState)

def encode_action(direction):
    encoding = list(bin(direction)[2:].rjust(3, '0'))
    return encoding

def to_dfa(encoding):
    dfa_encoding = []
    for i, v in enumerate(encoding):
        i += 1
        if v == '0':
            dfa_encoding.append(str(-i))
        else:
            dfa_encoding.append(str(i))
    return dfa_encoding

def memoize(f):
    memo = {}
    def helper(x):
        if x not in mem:
            memo[x] = f(x)
        return memo[x]
    return helper

N_GOALS = 4
steps_to_goal = {}
all_cells = set()

for i in range(xsize):
    for j in range(ysize):
        cell = (i,j)
        all_cells.add(cell)

def move(cell, direction):
    (x, y) = cell
    if direction == 0:
        x = x + 1
    elif direction == 1:
        y = y + 1
    elif direction == 2:
        x = x -1
    elif direction == 3:
        y = y - 1
    return (min(max(x, 0), xsize-1), min(max(y,0),ysize-1))


legal_cells = all_cells - bad_cells
start_cell = (0,0)
max_steps_goals = {}
plan = True
if plan:
    for goal in range(N_GOALS):
        max_steps_goals[goal] = float('-inf')
        goal_zone = goal_zones[goal]
        covered_cells = set()
        for cell in goal_zone:
            steps_to_goal[(cell, goal)] = 0
            covered_cells.add(cell)
        while len(legal_cells - covered_cells) > 0:
            covered_cells_w = copy.deepcopy(covered_cells)
            for cell in covered_cells_w:
                for direction in range(4):
                    candidate = move(cell, direction)
                    path_len = steps_to_goal[(cell, goal)] + 1
                    if candidate in legal_cells:
                        if (candidate, goal) in steps_to_goal:
                            old_candidate = steps_to_goal[(candidate,goal)]
                        else:
                            old_candidate = float('inf')
                        steps_to_goal[(candidate,goal)] = min(old_candidate, path_len)
                        max_steps_goals[goal] = max(max_steps_goals[goal], path_len)
                        covered_cells.add(candidate)

# ==================================
# Construct MDP --> Transition file
# ==================================

# First, a function that computes the possible/likely
# transitions when going from a (x,y)-cell into some
# direction. It computes the image of the complete cell
# and then performs probability-weighting according to
# the areas of overlap
# Directions:
# * 0: x + 1
# * 1: y + 1
# * 2: x - 1
# * 3: y - 1
# * 4: no move
def computeSuccs(xpos,ypos,direction):

    # If direction is "4", this means no move
    if (direction==4):
        return [(xpos,ypos,1.0)]

    finalSuccs = []
    errorProb = 0.0
    if (direction==0):
        succs = [(xpos+1,ypos),(xpos+1,ypos+1)]
    elif (direction==1):
        succs = [(xpos,ypos+1),(xpos-1,ypos+1)]
    elif (direction==2):
        succs = [(xpos-1,ypos),(xpos-1,ypos-1)]
    elif (direction==3):
        succs = [(xpos,ypos-1),(xpos+1,ypos-1)]
    
    if succs[0][0]<0:
        errorProb += 1-positionUpdateNoise
    elif succs[0][0]>=xsize:
        errorProb += 1-positionUpdateNoise
    elif succs[0][1]<0:
        errorProb += 1-positionUpdateNoise
    elif succs[0][1]>=ysize:
        errorProb += 1-positionUpdateNoise
    else:
        finalSuccs.append((succs[0][0],succs[0][1],1-positionUpdateNoise))

    if succs[1][0]<0:
        errorProb += positionUpdateNoise
    elif succs[1][0]>=xsize:
        errorProb += positionUpdateNoise
    elif succs[1][1]<0:
        errorProb += positionUpdateNoise
    elif succs[1][1]>=ysize:
        errorProb += positionUpdateNoise
    else:
        finalSuccs.append((succs[1][0],succs[1][1],positionUpdateNoise))

    if errorProb>0.0:
        finalSuccs.append((-1,-1,errorProb))
        
    return finalSuccs

# Iterate over all cells and compute transition probabilities
transitionLines = []
overallNofTransitions = 0
for xA in range(0,xsize):
    for yA in range(0,ysize):
        for (csf,payoff) in [(x,0) for x in range(NUMBER_OF_COLORS)] + [(0,1)]:
            sourceState = stateMapper[(xA,yA,csf,payoff)]
            overallNofTransitions += 5
            for dirA in [0,1,2,3,4]: # Action 4 is standing still
                errorProb = 0
                succA = computeSuccs(xA,yA,dirA)
                for (destXA,destYA,probA) in succA:
                    if destXA==-1:
                        errorProb += probA
                    else:
                        if imageData[destXA+destYA*xsize]==colors[csf]:
                            csfPrime = csf + 1
                            payoffPrime = 1
                        else:
                            csfPrime = csf
                            payoffPrime = 0
                        if csfPrime==NUMBER_OF_COLORS:
                            csfPrime = 0
                        else:
                            payoffPrime = 0
                            
                        # transitionLines.append([sourceState,dirA,stateMapper[(destXA,destYA,csfPrime,payoffPrime)],probA*0.99999])
                        transitionLines.append([sourceState,dirA,stateMapper[(destXA,destYA,csfPrime,payoffPrime)],probA])
                # errorProb += 0.00001*(1-errorProb)
                if errorProb>0:
                    transitionLines.append([sourceState,dirA,errorState,errorProb])
     
# ==================================
# Prepare reverse state mapper and
# Searchable transition list
# ==================================
reverseStateMapper = {}
for (a,b) in stateMapper.items():
    reverseStateMapper[b] = a
transitionLists = {}
for (a,b,c,d) in transitionLines:
    if not (a,b) in transitionLists:
        transitionLists[(a,b)] = [(c,d)]
    else:
        transitionLists[(a,b)].append((c,d))

NUMBER_OF_BITS = int(math.ceil(math.log((len(reverseStateMapper) - 1) / (NUMBER_OF_COLORS + 1), 2))) 
# print ("Number of bits for states:", NUMBER_OF_BITS)
max_steps_in_zone = 3

num_steps_on_bomb = 3

# recharging_zone = [(5, 10)]
# max_steps_in_zone = 20
# danger_zone = []
# for state in range(0, len(reverseStateMapper) - 1, 5): #exclude error state
#     (x, y, _, _) = reverseStateMapper[state]
#     danger_zone.append((x,y))

# danger_zone = [(7, 6), (7, 7), (8, 6), (8, 7)]
# max_steps_in_zone = 3
if exact_spec:
    with open("avoid_walls_shield_exact.dfa", "w") as output_file:
        directions = [0,1,3,]
        transitions = []
        for ((state, direction), probabilities) in transitionLists.items():
            next_states = [p[0] for p in filter(lambda p: p[1] == 1.0, probabilities)]
            next_state = next_states[0]
            transitions.append((state, next_state, to_dfa(encode_action(direction))))
            if direction == 4:
                for i in range(5, 8):
                    #transitions.append((state, next_state, to_dfa(encode_action(i))))
                    transitions.append((state, next_state, to_dfa(encode_action(i))))
        output_file.write('dfa {} 0 {} 1 1 {}\n'.format(BAD_STATE + 1, len(directions), len(transitions)+1))
        output_file.write('{}\n'.format(1))
        output_file.write('{}\n'.format(BAD_STATE +1))
        # encode dynamics for all possible encodings
        for (state, next_state, encoded_action) in sorted(transitions, key=lambda x: (x[0], x[1])):
            output_file.write('{} {} {}\n'.format(state+1, next_state+1, ' '.join(map(str, encoded_action))))
        output_file.write('{} {}\n'.format(BAD_STATE + 1, BAD_STATE + 1))
        for i in range(len(directions)):
            output_file.write('{} e{}\n'.format(i+1, i+1))
    exit(1) 


if gen_spec:
    with open("avoid_walls_shield.dfa", "w") as file:
        directions = [0, 1, 2, 3]
        transitions = []
        
        for combination in sum([list(map(list, itertools.combinations(directions, i))) for i in range(5)], []):
            sensors_enc = [str(x + 1 if x in combination else -(x + 1)) for x in directions]
            for action in range(4):
                action_enc = [str(-(idx + 5) if x == '0' else (idx + 5)) for idx, x in enumerate(list(bin(action)[2:].rjust(3, '0')))]
                target_state = 1 if action not in combination else 2
                transitions.append("1 {0} {1} {2}\n".format(target_state, " ".join(sensors_enc), " ".join(action_enc)))
        action_enc = [str(-(idx + 5) if x == '0' else (idx + 5)) for idx, x in enumerate(list(bin(4)[2:].rjust(3, '0')))]
        transitions.append("1 1 {0}\n".format(" ".join(action_enc)))

        #print unused action transitions
        for action in range(5, 8):
            action_enc = [str(-(idx + 5) if x == '0' else (idx + 5)) for idx, x in enumerate(list(bin(action)[2:].rjust(3, '0')))]
            transitions.append("1 2 {0}\n".format(" ".join(action_enc)))
            
        # print 'bad' state loop
        transitions.append("2 2\n")
            
        #print header & start/end states
        file.write("dfa 2 4 3 1 1 {0}\n1\n2\n".format(len(transitions)))
        
        #print transitions
        file.write("".join(transitions))        
        file.write("1 sensor_right\n")
        file.write("2 sensor_down\n")
        file.write("3 sensor_left\n")
        file.write("4 sensor_up\n")
        
        for bit in range(1, 4):
            file.write("{0} o{1}\n".format(4 + bit, 4 - bit))
        
            
    #shield preventing collision with second robot
    with open("enemy_shield.dfa", "w") as file:
        #
        # x x x x x  1  6 11 16 21
        # x x x x x  2  7 12 17 22
        # x x o x x  3  8 13 18 23
        # x x x x x  4  9 14 19 24
        # x x x x x  5 10 15 20 25
        #
        # state 0 means no enemy in range
        
        transitions = []
        unused_states = [0, 13]
        for (enemy_x, enemy_y) in list(product(range(1, 6), repeat=2)):
            if enemy_x == 3 and enemy_y == 3:
                continue
            enemy_state = 5 * (enemy_x - 1) + enemy_y
            if abs(enemy_x - 3) + abs(enemy_y - 3) > 2:
                unused_states.append(enemy_state)
                continue
            num_state_bits = 5
            for action in range(4):
                action_allowed = True
                for enemy_action in range(5):
                    enemy_next = filter(lambda t: t[0] != -1, list(map(lambda t: (t[0], t[1]) if t[2] > 0 else (-1, -1), computeSuccs(enemy_x, enemy_y, enemy_action))))
                    next = filter(lambda t: t[0] != -1, list(map(lambda t: (t[0], t[1]) if t[2] > 0 else (-1, -1), computeSuccs(3, 3, action))))
                
                    intersection = set(next).intersection(set(enemy_next))
                    if len(intersection) > 0:
                        action_allowed = False
                        break
                state_enc = [str(-(idx + 1) if x == '0' else (idx + 1)) for idx, x in enumerate(list(bin(enemy_state)[2:].rjust(num_state_bits, '0')))]
                action_enc = [str(-(idx + 1 + num_state_bits) if x == '0' else (idx + 1 + num_state_bits)) for idx, x in enumerate(list(bin(action)[2:].rjust(3, '0')))]
                
                transitions.append("1 {0} {1} {2}\n".format(1 if action_allowed else 2, " ".join(state_enc), " ".join(action_enc)))
                
        action_enc = [str(-(idx + 1 + num_state_bits) if x == '0' else (idx + 1 + num_state_bits)) for idx, x in enumerate(list(bin(4)[2:].rjust(3, '0')))]
        transitions.append("1 1 " + " ".join(action_enc) + "\n")
  #              
        #print unused action transitions
        for action in range(5, 8):
            action_enc = [str(-(idx + 1 + num_state_bits) if x == '0' else (idx + 1 + num_state_bits)) for idx, x in enumerate(list(bin(action)[2:].rjust(3, '0')))]
            transitions.append("1 2 " + " ".join(action_enc) + "\n")
        
        for state in unused_states:
            state_enc = [str(-(idx + 1) if x == '0' else (idx + 1)) for idx, x in enumerate(list(bin(state)[2:].rjust(num_state_bits, '0')))]
            transitions.append("1 1 " + " ".join(state_enc) + "\n")
        
        #print ununsed state transitions
        for state in range(26, int(math.pow(2, num_state_bits))):
            state_enc = [str(-(idx + 1) if x == '0' else (idx + 1)) for idx, x in enumerate(list(bin(state)[2:].rjust(num_state_bits, '0')))]
            transitions.append("1 1 " + " ".join(state_enc) + "\n")
        
        #print final state transition
        transitions.append("2 2\n")
        
        # print header
        file.write("dfa 2 {0} 3 1 1 {1}\n1\n2\n".format(num_state_bits, len(transitions)))
        
        #print transitions
        for transition in transitions:
            file.write(transition)
                
        # print labels
        for bit in range(1, num_state_bits + 1):
            file.write("{0} e{1}\n".format(bit, num_state_bits + 1 - bit))
        for bit in range(1, 4):
            file.write("{0} o{1}\n".format(num_state_bits + bit, 4 - bit))
        
    #with open("bomb_shield.dfa", "w") as file:
    #    
    #    transitions = []
    #    for state in range(1, num_steps_on_bomb + 1):
    #        transitions.append("{0} 1 -1".format(state))
    #        
    #        actions = range(8)
    #        actions.remove(4) # remove stay
    #        for action in actions:
    #            action_enc = [str(-(idx + 2) if x == '0' else (idx + 2)) for idx, x in enumerate(list(bin(action)[2:].rjust(3, '0')))]
    #            transitions.append("{0} 1 1 {1}".format(state, " ".join(action_enc)))
    #        
    #        action_enc = [str(-(idx + 2) if x == '0' else (idx + 2)) for idx, x in enumerate(list(bin(4)[2:].rjust(3, '0')))]
    #        transitions.append("{0} {1} 1 {2}".format(state, state + 1, " ".join(action_enc)))
    #        
    #    transitions.append("{0} {0}".format(num_steps_on_bomb + 1))
    #            
    #    file.write("dfa {0} 1 3 1 1 {1}\n1\n{0}\n".format(num_steps_on_bomb + 1, len(transitions)))
    #    file.write("\n".join(transitions))
    #    file.write("\n1 b\n")
    #    for bit in range(1, 4):
    #        file.write("{0} o{1}\n".format(1 + bit, 4 - bit))
    
    #shield for danger zones
    with open("danger_zone_shield.dfa", "w") as file:
        zone = set(danger_zone)
        zones = {}
        #compute zones:
        current_zone = 1
        while len(zone) > 0:
            zones[current_zone] = set()
            for (x, y) in zone:
                at_boundary = False
                for action in range(4):
                    # we are looking for an action which leads for sure out of the danger zone
                    succs = computeSuccs(x, y, action)
                    at_boundary = True
                    for (new_x, new_y, prob) in succs:
                        if prob > 0 and ((new_x,new_y) in zone or (new_x,new_y,0,0) not in stateMapper):
                            # in the danger zone
                            at_boundary = False
                            break
                    if at_boundary:
                        break
                if at_boundary:
                   zones[current_zone].add((x,y))
                    
            zone -= zones[current_zone]
            current_zone += 1
        
        # print (zones)
        
        end_state = max_steps_in_zone + 1
        transitions = []
        num_state_bits = 7
        for num_steps_in_zone in range(1,end_state):
            print ("state " + str(num_steps_in_zone))
            for state in range(0, len(reverseStateMapper) - 1, 5): #exclude error state
                (x,y,_,_) = reverseStateMapper[state]
                state_enc = [str(-(idx + 1) if bit == '0' else (idx + 1)) for idx, bit in enumerate(list(bin(state / 5)[2:].rjust(num_state_bits, '0')))]
                zone_idx = 0
                for idx, zone in zones.iteritems():
                    if (x,y) in zone:
                        zone_idx = idx
                        break
        
                if zone_idx == 0:
                    transitions.append("{0} 1 {1}\n".format(num_steps_in_zone, " ".join(state_enc)))
                    continue
                max_acceptable_zone = max_steps_in_zone - num_steps_in_zone # maximal acceptable zone as target
                if zone_idx < max_acceptable_zone or max(zones.keys()) <= max_acceptable_zone:
                    transitions.append("{0} {1} {2}\n".format(num_steps_in_zone, num_steps_in_zone + 1, " ".join(state_enc)))
                    continue
                
                print( "max_zone: " + str(max_acceptable_zone))
                if zone_idx <= max_acceptable_zone + 1:
                    for action in range(5):
                        succs = computeSuccs(x, y, action)
                        next_zone_idx = 0
                        for (next_x,next_y,prob) in succs:
                            if prob == 0: continue
                            for idx, zone in zones.iteritems():
                                if (next_x,next_y) in zone:
                                    next_zone_idx = max(next_zone_idx, idx)
                        print( "action " + str(action) + " leads to zone: " + str(next_zone_idx))
                        next_state = end_state if next_zone_idx > max_acceptable_zone else (num_steps_in_zone + 1 if next_zone_idx > 0 else 1)
                        action_enc = [str(-(idx + 1 + num_state_bits) if bit == '0' else (idx + 1 + num_state_bits)) for idx, bit in enumerate(list(bin(action)[2:].rjust(3, '0')))]
                        transitions.append("{0} {1} {2} {3}\n".format(num_steps_in_zone, next_state, " ".join(state_enc), " ".join(action_enc)))
                    for action in range(5,8):
                        action_enc = [str(-(idx + 1 + num_state_bits) if bit == '0' else (idx + 1 + num_state_bits)) for idx, bit in enumerate(list(bin(action)[2:].rjust(3, '0')))]
                        transitions.append("{0} {1} {2} {3}\n".format(num_steps_in_zone, end_state, " ".join(state_enc), " ".join(action_enc)))
                        
                    continue
                
                # this should never happen .. do whatever we want to
                transitions.append("{0} 1 {1}\n".format(num_steps_in_zone, " ".join(state_enc)))
          
            #print ununsed state transitions
            for state in range((len(reverseStateMapper) - 1) / 5, int(math.pow(2, num_state_bits))):
                state_enc = [str(-(idx + 1) if x == '0' else (idx + 1)) for idx, x in enumerate(list(bin(state)[2:].rjust(num_state_bits, '0')))]
                transitions.append("{0} 1 {1}\n".format(num_steps_in_zone, " ".join(state_enc)))
             
            
        transitions.append("{0} {0}\n".format(end_state))
        # print header
        file.write("dfa {0} {1} 3 1 1 {2}\n1\n{0}\n".format(num_steps_in_zone + 1, num_state_bits, len(transitions)))
        
        #print transitions
        for transition in transitions:
            file.write(transition)
                
        # print labels
        for bit in range(1, num_state_bits + 1):
            file.write("{0} i{1}\n".format(bit, num_state_bits + 1 - bit))
        for bit in range(1, 4):
            file.write("{0} o{1}\n".format(num_state_bits + bit, 4 - bit))
                
    exit()
        
# =========================================
# Initialize interactive display
# =========================================
pygame.init()
displayInfo = pygame.display.Info()
MAGNIFY = min(MAGNIFY,displayInfo.current_w*3/4/xsize)
MAGNIFY = min(MAGNIFY,displayInfo.current_h*3/4/ysize)


class Map(Environment):
    def __init__(self):
        self.logger = open('map.log', 'w')
        self.statelogger = open('states.log', 'w')
        self.reset()
        
    def reset(self):
        global shield
        shield = Shield()
        # print "reset called"
        self.state = 0
        self.penalty = 0
#        self.statelogger.write('\n\n')
#        self.statelogger.write('{} {}\n'.format(self.state, reverseStateMapper[self.state]))
        
    def performAction(self, action):   
        error = len(reverseStateMapper) - 1
        # action = int(action[0])
        
        
        actions = action[action != -1]
        actions = list(map(int, actions))
        
        # print action
        # state_enc = map(int, list(bin(self.state / (NUMBER_OF_COLORS + 1))[2:].rjust(NUMBER_OF_BITS, '0')))
       
        encoded_actions = []
        for a in actions:
            encoded_actions.append(list(map(int, list(bin(a)[2:].rjust(3, '0')))))
            
        (robotXA, robotYA, csf, payoff) = reverseStateMapper[self.state]
#        self.logger.write("{} {} {} {} {}\n".format(robotXA, robotYA, csf, payoff, action))
        
        # simulate sensors
        state_enc = []
        for a in range(4):
            # print computeSuccs(robotXA, robotYA, a)
            succs = filter(lambda t: t[2] > 0, computeSuccs(robotXA, robotYA, a))
            valid = True
            for succ in succs:
                if succ[0] == -1 or not (succ[0], succ[1], 0, 0) in stateMapper:
                    valid = False
                    break
            state_enc.append(0 if valid else 1)
            
                
        # print state_enc
 # print "action" + str(encoded_actions[0])
        
        if enemies_enabled:
            enemy_state = 0
            (robotXA,robotYA,csf,payoff) = reverseStateMapper[level.state]
            for enemy in enemy_handler.getEnemyPositions():
                x_diff = abs(enemy[0] - robotXA)
                y_diff = abs(enemy[1] - robotYA)
                if x_diff + y_diff <= 2:
                    enemy_state = (enemy[0] - robotXA + 2) * 5 + (enemy[1] - robotYA + 3)
                    break        
            enemy_state_enc = list(map(int, list(bin(enemy_state)[2:].rjust(5, '0'))))
        
            state_enc.extend(enemy_state_enc)
                    
        # print "Colors seen so far:", csf
        if len(bombs) > 0:
            state_enc.append(1 if (robotXA + 1, robotYA + 1) in bombs else 0)
        for enc_action in encoded_actions:
            state_enc.extend(enc_action)
        
#        self.logger.write('encoded_actions {}\n'.format(encoded_actions))
        state_enc = [item for sublist in encoded_actions for item in sublist]
#        self.logger.write('shield_input {}\n'.format(state_enc))
        # print state_enc
        if shield_options > 0:
            corr_action = shield.tick(state_enc)
#        self.logger.write('shield_output {}\n'.format(corr_action))

            # print corr_action
            corr_action = int("".join(list(map(str, corr_action[:len(corr_action) -1]))), 2)
    #        self.logger.write('corr_action {}\n'.format(corr_action))
        else:
            corr_action = actions[0]

        if (actions[0] != corr_action) and huge_neg_reward:
            self.penalty += 1.

        if (actions[0] != corr_action) and neg_reward and args.sarsa:
            self.penalty += 0.1
            # qvalue = self.module.getValue(self.laststate, action)
            # self.module.updateValue(self.laststate, action, qvalue + self.alpha * ((-1 if self.neg_reward else self.lastreward) - qvalue))
            #experiment.acc_reward -= .3
        #     print False
        used_actions = []
        for a in actions:
            if a == corr_action: break
            used_actions.append(a)
            # learner.explorer.n_values.params.reshape(learner.explorer.n_values.numRows,learner.explorer.n_values.numColumns)[self.state, a] += 1
        if huge_neg_reward:
            action = actions[0]
        else:
            action = corr_action
        
        used_actions.append(action)
        
        while len(used_actions) < 5:
            used_actions.append(-1)
                
        agent.lastaction = used_actions
#        self.logger.write("used action: {}\n".format(action))
       
        transitionList = transitionLists[(self.state, action)]
        
        dest = None
        randomNumber = random.random()
        for (a,b) in transitionList:
            if randomNumber<=b:
                dest = a
                randomNumber = 123.0
            else:
                randomNumber -= b
        # Rounding error?
        if (dest==None):
            dest = transitionList[0][0]

#        if dest in bad_states:
#            self.logger.write('In illegal state (walled state) {}\n'.format(dest))
             
        if dest == BAD_STATE:
#            self.logger.write('In illegal state (bad state)\n')
            experiment.acc_reward -= 1
            self.penalty += 1
            # self.reset()
            if shield_options > 0 and not args.huge_neg_reward:
                print ("Shields are not allowed to make errors!3 {} {} {}\n".format(self.state,
                    dest, action))
                exit()
#            self.logger.write("Staying in state {}\n".format(self.state))
            dest = self.state
            to_bad_state = True
        else:
            to_bad_state = False
            
        # learner.explorer.n_values.params.reshape(learner.explorer.n_values.numRows,learner.explorer.n_values.numColumns)[self.state, action] += 1
#        self.statelogger.write('{} {} - {}\n'.format(dest, dest in bad_states, action))
#        self.statelogger.write('{} {} {} - {}\n'.format(dest, reverseStateMapper[dest],
#            to_bad_state, action))
        
        self.state = dest
            
    def getSensors(self):
        return [self.state]

class VisitAllColors(Task):
    def __init__(self, env, gamma):
        Task.__init__(self, env)
        self.last_reward = 0
        self.task_logger = open('task.log', 'w')
        
    def getReward(self):
        # if (reverseStateMapper[self.env.state][3] != 0):
        # print "all colors visited"
        ret = self.last_reward
        self.last_reward = reverseStateMapper[self.env.state][3] - self.env.penalty
#        self.task_logger.write('getReward {} - {}\n'.format(self.last_reward + self.env.penalty, self.env.penalty))

        self.env.penalty = 0
        return self.last_reward

class VisitAllColorsShaped(VisitAllColors):
    def __init__(self, env, gamma, cost_per_step):
        VisitAllColors.__init__(self, env, gamma)
        self.prev_state = None
        self.cost_step = cost_per_step

    def potential(self, state):
        (x, y, goal, _) = state
        steps_taken = max_steps_goals[goal] - steps_to_goal[((x,y), goal)]
        return steps_taken * self.cost_step

    def get_shaped_reward(self, state):
        # goals are equal, potential is difference between steps
        if self.prev_state is None:
            return 0.0
        if state[3] == self.prev_state[3]:
            shaped_reward = self.potential(state) - self.potential(self.prev_state)
        else:
            # goals are not equal, potential difference is a single step cost
            shaped_reward = self.cost_step
        return shaped_reward

    def getReward(self):
        ret = self.last_reward
        self.last_reward = reverseStateMapper[self.env.state][3] - self.env.penalty
        state = reverseStateMapper[self.env.state]
        self.last_shaped_reward = self.get_shaped_reward(state)
#        self.task_logger.write('getReward {} - {}\n'.format(self.last_reward + self.env.penalty, self.env.penalty))
        self.env.penalty = 0
        self.prev_state = state 
        return self.last_reward + self.last_shaped_reward
        

        
class MyExperiment(Experiment):
    def __init__(self, task, agent):
        Experiment.__init__(self, task, agent)
        
        agent.learner.explorer.experiment = self
        # agent.learner.module.getValue()
        
        self.screen = pygame.display.set_mode(((xsize+2)*MAGNIFY,(ysize+2)*MAGNIFY))
        pygame.display.set_caption('Policy Visualizer')
        self.clock = pygame.time.Clock()

        self.screenBuffer = pygame.Surface(self.screen.get_size())
        self.screenBuffer = self.screenBuffer.convert()
        self.screenBuffer.fill((64, 64, 64)) # Dark Gray
        
        self.bombImage = pygame.image.load("bomb_image.png")
        self.bombImage = pygame.transform.scale(self.bombImage, (MAGNIFY - 2, MAGNIFY - 2))

    
        self.isPaused = False
        self.isCrashed = False
        self.speed = 10
        self.num = 0
        self.robotXA = -1
        self.robotYA = -1
        self.bomb_counter = 0
        
        self.count = 0
        self.acc_reward = 0
        self.collect_data = False
        if collect_data_file != None:
            self.collect_data = True
            self.collect_episode_data_file = open(collect_data_file + "_episodelen.data", "w")
            self.collect_reward_data_file = open(collect_data_file + "_avg_reward.data", "w")
        self.interaction_logger = open('interactions.log', 'w')
    
    def _oneInteraction(self):
        global draw
        
        resetInThisRound = False
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT or (event.type == pygame.locals.KEYDOWN and event.key in [pygame.locals.K_ESCAPE,pygame.locals.K_q]):
                return
            if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_SPACE):
                controller.params.reshape(controller.numRows, controller.numColumns).tofile("test.table")
                self.isPaused = not self.isPaused
            if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_r):
                resetInThisRound = True
            if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_PLUS):
                self.speed += 1
            if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_MINUS):
                self.speed = max(self.speed-1,1)
            if (event.type == pygame.locals.KEYDOWN and event.key == pygame.locals.K_d):
                draw = not draw
            
        # if self.isCrashed:
  #           self.isCrashed = False
  #           # level.reset()
  # 
        # Update 
        if resetInThisRound:
            print ("reset")
            level.reset()

                
        old = (self.robotXA, self.robotYA)
        (self.robotXA,self.robotYA,csf,payoff) = reverseStateMapper[level.state]
        
        if not self.isCrashed and enemies_enabled:
            enemy_handler.update(old)
            for e in enemy_handler.getEnemyPositions():
                if (self.robotXA, self.robotYA) == e:
                    self.isCrashed = True
                    level.penalty += 1
                    self.acc_reward -= 1
                    if shield_options > 0 and not args.huge_neg_reward:
                        print ("Shields are not allowed to make errors!1")
                        exit()
                    break
        
        if (self.robotXA + 1, self.robotYA + 1) in bombs:
            self.bomb_counter += 1
            self.logger.write('In illegal state (bomb)\n')
            if self.bomb_counter == 4:
                self.isCrashed = True
                level.penalty += 1
                self.acc_reward -= 1
                if shield_options > 0 and not args.huge_neg_reward:
                    print ("Shields are not allowed to make errors!2")
                    exit()
        else:
            self.bomb_counter = 0

        if draw:
            q_max = 0
            for state in range(len(reverseStateMapper) - 1):
                q_max = max(q_max, max(controller.getActionValues(state)))
                
            # Draw Field
            for x in range(0,xsize):
                for y in range(0,ysize):
                    paletteColor = imageData[y*xsize+x]
                    color = palette[paletteColor*3:paletteColor*3+3]
                    pygame.draw.rect(self.screenBuffer,color,((x+1)*MAGNIFY,(y+1)*MAGNIFY,MAGNIFY,MAGNIFY),0)
        
            # Draw boundary
            if self.robotXA==-1 or self.isCrashed:
                boundaryColor = (255,0,0)
            else:
                boundaryColor = (64,64,64)
            pygame.draw.rect(self.screenBuffer,boundaryColor,(0,0,MAGNIFY*(xsize+2),MAGNIFY),0)
            pygame.draw.rect(self.screenBuffer,boundaryColor,(0,MAGNIFY,MAGNIFY,MAGNIFY*(ysize+1)),0)
            pygame.draw.rect(self.screenBuffer,boundaryColor,(MAGNIFY*(xsize+1),MAGNIFY,MAGNIFY,MAGNIFY*(ysize+1)),0)
            pygame.draw.rect(self.screenBuffer,boundaryColor,(MAGNIFY,MAGNIFY*(ysize+1),MAGNIFY*xsize,MAGNIFY),0)
            # pygame.draw.rect(screenBuffer,boundaryColor,(0,0,MAGNIFY*(xsize+2),MAGNIFY),0)

            # Draw cell frames
            for x in range(0,xsize):
                for y in range(0,ysize):
                    pygame.draw.rect(self.screenBuffer,(0,0,0),((x+1)*MAGNIFY,(y+1)*MAGNIFY,MAGNIFY,MAGNIFY),1)
                    if (x+1,y+1) in bombs:
                        self.screenBuffer.blit(self.bombImage, ((x+1)*MAGNIFY+1,(y+1)*MAGNIFY+1))
            pygame.draw.rect(self.screenBuffer,(0,0,0),(MAGNIFY-1,MAGNIFY-1,MAGNIFY*xsize+2,MAGNIFY*ysize+2),1)

            # Draw "Good" Robot
            if self.robotXA!=-1:
                pygame.draw.circle(self.screenBuffer, (192,32,32),
                        int((self.robotXA+1)*MAGNIFY+MAGNIFY/2,(self.robotYA+1)*MAGNIFY+MAGNIFY/2)
                        , int(MAGNIFY/3-2), 0)
                pygame.draw.circle(self.screenBuffer, (255,255,255),
                        int((self.robotXA+1)*MAGNIFY+MAGNIFY/2,(self.robotYA+1)*MAGNIFY+MAGNIFY/2)
                        , int(MAGNIFY/3-1), 1)
                pygame.draw.circle(self.screenBuffer, (0,0,0),
                        int((self.robotXA+1)*MAGNIFY+MAGNIFY/2,(self.robotYA+1)*MAGNIFY+MAGNIFY/2)
                        , int(MAGNIFY/3), 1)

            # Draw "Bad" Robots
            if enemies_enabled:
                for (e_x, e_y) in enemy_handler.getEnemyPositions():
                    pygame.draw.circle(self.screenBuffer, (32,32,192),
                            int((e_x+1)*MAGNIFY+MAGNIFY/2,(e_y+1)*MAGNIFY+MAGNIFY/2) ,
                            int(MAGNIFY/3-2), 0)
                    pygame.draw.circle(self.screenBuffer, (255,255,255), ((e_x+1)*MAGNIFY+MAGNIFY/2,(e_y+1)*MAGNIFY+MAGNIFY/2) , MAGNIFY/3-1, 1)
                    pygame.draw.circle(self.screenBuffer, (0,0,0), ((e_x+1)*MAGNIFY+MAGNIFY/2,(e_y+1)*MAGNIFY+MAGNIFY/2) , MAGNIFY/3, 1)
                

            # zone_width = danger_zone[-1][0] - danger_zone[0][0] + 1
     #        zone_height = danger_zone[-1][1] - danger_zone[0][1] + 1
     # pygame.draw.rect(screenBuffer,(200,200,0),(MAGNIFY*(danger_zone[0][0]+1),MAGNIFY*(danger_zone[0][1]+1),MAGNIFY*zone_width,MAGNIFY*zone_height),5)


            # Flip!
            self.screen.blit(self.screenBuffer, (0, 0))
            pygame.display.flip()
                                
            # Make the transition
            if not self.isPaused:
                # Done
                self.clock.tick(self.speed)
            else:
                self.clock.tick(3)

        self.acc_reward += payoff * 10
        if self.collect_data:
            self.count += 1
            if payoff > 0:
                self.collect_episode_data_file.write(str(self.count) + "\n")
                self.count = 0
                level.reset()
            if self.stepid % 100 == 0:
                self.collect_reward_data_file.write(str(self.acc_reward / 100.) + "\n")
                self.acc_reward = 0
            if self.stepid % 100000 == 0:
                pass
        
        if self.stepid % 100 == 0:
            sys.stdout.write("\033[K")
            sys.stdout.write("[{2}{3}] ({0}/{1}) | alpha = {4} | epsilon = {5}\n".format(self.stepid, MAX_STEPS, '#'*int(math.floor(self.stepid/float(MAX_STEPS)*20)), ' '*int((20 - math.floor(self.stepid/float(MAX_STEPS)*20))), learner.alpha, learner.explorer.exploration))
            sys.stdout.write("\033[F")
            
        if self.stepid >= MAX_STEPS:
            print ("\nSimulation done!")
            
            sys.exit()          
            
        if payoff > 0:
            # episode done
            if save_file != None:
                controller.params.reshape(controller.numRows, controller.numColumns).tofile(save_file)
#            learner.alpha *= 0.999
            learner.alpha = cooldown_alpha[self.stepid]
            learner.explorer.exploration = cooldown_epsilon[self.stepid]
            
        self.isCrashed = False
        if not self.isPaused:
            return Experiment._oneInteraction(self)
        else: return self.stepid


        
# ==================================
# Call main program
# ==================================

    

#
# def enemy_random(enemy, good):
#     possible_next_positions = set([])
#     for action in range(5):
#         next = computeSuccs(enemy[0], enemy[1], action)
#         for t in next:
#             if t[0] != -1 and t[2] > 0 and (t[0], t[1], 0, 0) in stateMapper:
#                 possible_next_positions.add((t[0], t[1]))
#
#     next_position_invalid = True
#     while next_position_invalid:
#         idx = random.randint(0, len(possible_next_positions) - 1)
#         enemy = list(possible_next_positions)[idx]
#         # we do not allow to drive at the old position of the good robot
#         next_position_invalid = enemy == good
#
#     return enemy
#
#
#
# enemies = []

gamma = .95
shield = Shield()
level = Map()
if cost_per_step:
    task = VisitAllColorsShaped(level, gamma=gamma, cost_per_step=cost_per_step)
else:
    task = VisitAllColors(level, gamma=gamma)

controller = MyActionValueTable(len(reverseStateMapper), 5)
if load_file != None:
    controller.initialize(np.fromfile(load_file))
else:
    controller.initialize(0.)
if not args.sarsa:
    learner = MyQ(alpha, gamma, neg_reward)
    learner.explorer = MyGreedyExplorer(shield_options, exploration)
elif args.sarsa:
    learner = SARSA(alpha, gamma)
learner.explorer = MyGreedyExplorer(shield_options, exploration)
learner.explorer._setModule(controller)
agent = LearningAgent(controller, learner)

draw = False
EXPLORATION_FACTOR = exploration

experiment = MyExperiment(task, agent)  
while 1:
    experiment.doInteractions(100)
    agent.learn()
    agent.reset()

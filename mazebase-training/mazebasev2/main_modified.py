from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import choice, randint
from time import sleep
from os import system
from pprint import PrettyPrinter
from six.moves import input
import numpy as np
import random
import json
import argparse
import yaml
import torch
import time
import os
import click
import pdb

import lib.mazebase.games as games
from lib.mazebase.games import featurizers
from pprint import pprint

from lib.planning import KnowledgePlanner

#from lib.knowledge import KnowledgeSubGraph, TripletInfo, Rule, create_rule_from_triplets

import logging
logging.getLogger().setLevel(logging.DEBUG)


# Get Input Arguments
parser = argparse.ArgumentParser(description='Script for training our knowledge agent')
    
##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default='options/knowledge_planner/minimum_viable_planning.yaml', type=str,
                    help='path to a yaml options file')
 
##################################################
parser.add_argument('--trial', type=int, default=0,
                    help='keep track of what trial you are on')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-vis', action='store_true', default=False,
                    help='disables visdom visualization')
parser.add_argument('--show-game', action='store_true', default=False,
                    help='prints game to terminal if option is selected')
parser.add_argument('--port', type=int, default=8097,
                    help='port to run the server on (default: 8097)')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='if true, print step logs')
parser.add_argument('--resume', default='', type=str,
                    help='name of checkpoint to resume')
parser.add_argument('--train-mode', type=str, default='train',
                    help='training/eval mode we are in')
parser.add_argument('--save-every', default=1000000000, type=int,
                    help='how often to save our models permanently')


def action_func(actions):
    print(list(enumerate(actions)))
    ind = -1
    while ind not in range(len(actions)):
        ind = input("Input number for action to take: ")
        try:
            ind = int(ind)
        except ValueError:
            ind = -1
    return actions[ind]

# Main function
def main():
    global args
    args = parser.parse_args()

    # Set options
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options = yaml.load(handle)
    #print('## args'); pprint(vars(args))
    #print('## options'); pprint(options)

    # Get sub opts
    method_opt = options['method']
    env_opt = options['env']
    log_opt = options['logs'] 

    # Set up the mazebase environment
    # Load the true world knowledge that we use to build the actual environment
    knowledge_root = env_opt['knowledge_root']
    world_knowledge_file = os.path.join(knowledge_root, env_opt['world_knowledge'][args.train_mode])
    #print("Loading world knowledge from %s" % world_knowledge_file)
    #main_logger.debug("Loading world knowledge from %s" % world_knowledge_file) 
    with open(world_knowledge_file) as f:
        world_knowledge = json.load(f)
    
    # Make the world
    map_size = (env_opt['state_rep']['w'], env_opt['state_rep']['w'], env_opt['state_rep']['h'], env_opt['state_rep']['h'])
    all_games = [games.BasicKnowledgeGame(world_knowledge=world_knowledge, proposed_knowledge=[], options=env_opt, map_size=map_size)]

    # Game wrapper
    game = games.MazeGame(
        all_games,
        #featurizer=featurizers.SentenceFeaturesRelative(
        #   max_sentences=30, bounds=4)
        featurizer=featurizers.GridFeaturizer()
    )

     # Reset the game
    game.reset()
    game.display()
    sleep(.1)
    system('clear')

    while True:
        game.display()

        id = game.current_agent()
        actions = game.all_possible_actions()
        action = action_func(actions)
        game.act(action)

        sleep(.1)
        system('clear')
        print("\n")



if __name__ == '__main__':
    main()

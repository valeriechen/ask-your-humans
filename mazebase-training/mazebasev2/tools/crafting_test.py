from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import choice, randint
from time import sleep
from os import system
from pprint import PrettyPrinter
from six.moves import input
import numpy as np
import pdb

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import lib.mazebase.games as games
from lib.knowledge import create_rule_from_triplets
from lib.mazebase.games import featurizers
from lib.mazebase.games import curriculum

import logging
logging.getLogger().setLevel(logging.DEBUG)

# TODO - what is player_mode?
player_mode = True

# Make a few simple knowledge verify games
# TODO - we'll eventually want to generate these programatically in some way
game_1_triplets = [('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'iron_ore_vein'), ('iron_ore', 'requires_item_1', 'pickaxe')]
game_1_test_triplet = ('iron_ore', 'requires_item_1', 'pickaxe')
game_1_Kp = set(game_1_triplets)
game_1_rules_json = [create_rule_from_triplets(game_1_triplets)]
game_1_distractors = {'diamond_pickaxe': 'CraftingItem'}
game_2_triplets = [('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'iron_ore_vein'), ('iron_ore', 'requires_item_1', 'diamond_pickaxe')]
game_2_test_triplet = ('iron_ore', 'requires_item_1', 'pickaxe')
game_2_Kp = set([('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'iron_ore_vein'), ('iron_ore', 'requires_item_1', 'pickaxe')])
game_2_rules_json = [create_rule_from_triplets(game_2_triplets)]
game_2_distractors = {'pickaxe': 'CraftingItem'}
game_3_triplets = [('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'iron_ore_vein')]
game_3_test_triplet = ('iron_ore', 'requires_item_1', 'pickaxe')
game_3_Kp = set([('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'iron_ore_vein'), ('iron_ore', 'requires_item_1', 'pickaxe')])
game_3_rules_json = [create_rule_from_triplets(game_3_triplets)]
game_3_distractors = {'diamond_pickaxe': 'CraftingItem', 'pickaxe': 'CraftingItem'}
game_4_triplets = [('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'iron_ore_vein'), ('iron_ore', 'requires_item_1', 'pickaxe')]
game_4_test_triplet = ('iron_ore', 'requires_action', 'mine')
game_4_Kp = set(game_4_triplets)
game_4_rules_json = [create_rule_from_triplets(game_4_triplets)]
game_4_distractors = {}
game_5_triplets = [('iron_ore', 'requires_action', 'grab'), ('iron_ore', 'requires_being_at', 'iron_ore_vein'), ('iron_ore', 'requires_item_1', 'pickaxe')]
game_5_test_triplet = ('iron_ore', 'requires_action', 'mine')
game_5_Kp = set([('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'iron_ore_vein'), ('iron_ore', 'requires_item_1', 'pickaxe')])
game_5_rules_json = [create_rule_from_triplets(game_5_triplets)]
game_5_distractors = {}
game_6_triplets = [('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'iron_ore_vein'), ('iron_ore', 'requires_item_1', 'pickaxe')]
game_6_test_triplet = ('iron_ore', 'requires_being_at', 'iron_ore_vein')
game_6_Kp = set(game_6_triplets)
game_6_rules_json = [create_rule_from_triplets(game_6_triplets)]
game_6_distractors = {'ore_vein': 'ResourceFont'}
game_7_triplets = [('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'ore_vein'), ('iron_ore', 'requires_item_1', 'pickaxe')]
game_7_test_triplet = ('iron_ore', 'requires_being_at', 'iron_ore_vein')
game_7_Kp = set([('iron_ore', 'requires_action', 'mine'), ('iron_ore', 'requires_being_at', 'iron_ore_vein'), ('iron_ore', 'requires_item_1', 'pickaxe')])
game_7_rules_json = [create_rule_from_triplets(game_7_triplets)] 
game_7_distractors = {'iron_ore_vein': 'ResourceFont'}

# Combine games
game_rule_list = [game_1_rules_json, game_2_rules_json, game_3_rules_json, game_4_rules_json, game_5_rules_json, game_6_rules_json, game_7_rules_json] 
distractors = [game_1_distractors, game_2_distractors, game_3_distractors, game_4_distractors, game_5_distractors, game_6_distractors, game_7_distractors]
proposed_knowledge = [game_1_Kp, game_2_Kp, game_3_Kp, game_4_Kp, game_5_Kp, game_6_Kp, game_7_Kp]
test_triples = [game_1_test_triplet, game_2_test_triplet, game_3_test_triplet, game_4_test_triplet, game_5_test_triplet, game_6_test_triplet, game_7_test_triplet]
all_games = [games.SimpleKnowledgeVerifyGame(rules_json=rules, other_items=items, proposed_knowledge=Kp, test_triplet=test_k, map_size=(5, 5, 5, 5)) for rules, items, Kp, test_k in zip(game_rule_list, distractors, proposed_knowledge, test_triples)]
# TODO - set featurizer= whatever we want our new featurizer to be

# TODO - go back and make game arguments optional so this is right
# TODO ???

game = games.MazeGame(
    all_games,
    # featurizer=featurizers.SentenceFeaturesRelative(
    #   max_sentences=30, bounds=4)
    featurizer=featurizers.GridFeaturizer()
)
game.reset()
max_w, max_h = game.get_max_bounds()

pp = PrettyPrinter(indent=2, width=160)
all_actions = game.all_possible_actions()
all_features = game.all_possible_features()
print("Actions:", all_actions)
print("Features:", all_features)
# sleep(2)


def action_func(actions):
    if not player_mode:
        return choice(actions)
    else:
        print(list(enumerate(actions)))
        ind = -1
        while ind not in range(len(actions)):
            ind = input("Input number for action to take: ")
            try:
                ind = int(ind)
            except ValueError:
                ind = -1
        return actions[ind]

frame = 0
game.display()
sleep(.1)
system('clear')
while True:
    print("r: {}\ttr: {} \tguess: {}".format(
        game.reward(), game.reward_so_far(), game.approx_best_reward()))
    config = game.observe()
    pp.pprint(config['observation'][1])
    # Uncomment this to featurize into one-hot vectors
    obs, info = config['observation']
    featurizers.grid_one_hot(game, obs)
    obs = np.array(obs)
    featurizers.vocabify(game, info)
    info = np.array(obs)
    config['observation'] = obs, info
    game.display()

    id = game.current_agent()
    actions = game.all_possible_actions()
    action = action_func(actions)
    game.act(action)

    sleep(.1)
    system('clear')
    print("\n")
    frame += 1
    if game.is_over() or frame > 300:
        frame = 0
        print("Final reward is: {}, guess was {}".format(
            game.reward_so_far(), game.approx_best_reward()))
        game.make_harder()
        game.reset()

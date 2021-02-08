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

from lib.knowledge import KnowledgeSubGraph, TripletInfo, Rule, create_rule_from_triplets

import logging
logging.getLogger().setLevel(logging.DEBUG)


# Get Input Arguments
parser = argparse.ArgumentParser(description='Script for training our knowledge agent')
    
##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default='options/knowledge_planner/default.yaml', type=str,
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

# Main function
def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    # Set options
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options = yaml.load(handle)
    print('## args'); pprint(vars(args))
    print('## options'); pprint(options)

    # Get sub opts
    method_opt = options['method']
    env_opt = options['env']
    log_opt = options['logs'] 

    # Set seed - just make the seed the trial number
    seed = args.trial
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)

    # Logging
    logpath = os.path.join(log_opt['log_base'], method_opt['mode'], log_opt['exp_name'], env_opt['world_knowledge'][args.train_mode].split('.')[0], 'trial%d' % args.trial)
    if len(args.resume) == 0:
        # Make directory, check before overwriting
        if os.path.isdir(logpath):
            if click.confirm('Logs directory already exists in {}. Erase?'
                .format(logpath, default=False)):
                os.system('rm -rf ' + logpath)
            else:
                return
        os.system('mkdir -p ' + logpath) 
    main_logger = logging.getLogger('spam_application')
    main_logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logpath + '/main_%s_%f.log' % (args.train_mode, time.time()))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    main_logger.addHandler(fh)
    main_logger.addHandler(ch)

    # Set up the mazebase environment
    # Load the true world knowledge that we use to build the actual environment
    knowledge_root = env_opt['knowledge_root']
    world_knowledge_file = os.path.join(knowledge_root, env_opt['world_knowledge'][args.train_mode])
    print("Loading world knowledge from %s" % world_knowledge_file)
    main_logger.debug("Loading world knowledge from %s" % world_knowledge_file) 
    with open(world_knowledge_file) as f:
        world_knowledge = json.load(f)
    
    # Generate the proposed knowledge 
    # TODO - should move this out of main!
    if env_opt['proposed_knowledge']['source'] == 'file':
        proposed_knowledge_file = os.path.join(knowledge_root, env_opt['proposed_knowledge'][args.train_mode])
        print("Loading proposed knowledge from %s" % proposed_knowledge_file)
        main_logger.debug("Loading proposed knowledge from %s" % proposed_knowledge_file) 
        with open(proposed_knowledge_file) as f:
            proposed_knowledge = json.load(f)
    # Randomly break rules in a way that is consistent with spawn rules
    elif env_opt['proposed_knowledge']['source'] == 'random_spawns':
        # Mode has to be fixed spawns for this to make sense
        assert(env_opt['spawn']['mode'] == 'fixed_spawns')    

        # Start with true knowledge rules
        # TODO - right now assuming we keep same number and there's a 1-1 between every proposed rule and true rule
        proposed_knowledge = []
        for true_rule in world_knowledge['rules']:
            # With some probability just keep the correct rule
            if random.random() < env_opt['proposed_knowledge']['true_prob']:
                proposed_knowledge.append(true_rule)
            # If not, randomly choose how many things to change
            else:
                # Choose how many things we want to change
                num_change_probs = env_opt['proposed_knowledge']['change_num_probs']
                num_change_probs = [float(p) for p in num_change_probs.split(',')]
                assert(abs(sum(num_change_probs) - 1) < 1e-6)
                num_change = np.random.choice(range(1, len(num_change_probs)+1), p=num_change_probs)

                # Choices are drop, add, or swap
                change_type_probs = env_opt['proposed_knowledge']['change_type_probs']
                change_type_probs = [float(p) for p in change_type_probs.split(',')]
                assert(abs(sum(change_type_probs) - 1) < 1e-6)
                spawn_ind = true_rule['spawn_ind']
                spawn_items = world_knowledge['spawns'][spawn_ind]
                item_dict = world_knowledge['objects']
                    
                # Randomly change one thing from the true knowledge
                tr_graph = KnowledgeSubGraph(Rule(true_rule).create_triplet_corresp())
                preconds = list(tr_graph.get_preconditions())
                postconds = tr_graph.get_postconditions()
                changed = [False for _ in preconds]             
                for _ in range(num_change):
                    change_success = False
                    while not change_success:    
                        change_type = np.random.choice(['add', 'drop', 'swap'], p=change_type_probs)             

                        # If it's drop, drop a random precondition, (but never action)
                        if change_type == 'drop':
                            drop_ind = random.choice(range(len(preconds)))
                            if preconds[drop_ind][0] == TripletInfo.REQUIRES_ACTION:
                                continue
                            elif changed[drop_ind]:
                                continue
                            else:
                                preconds.pop(drop_ind)
                                changed.pop(drop_ind)
                                change_success = True
                        # If it's add, add a random item or location from the spawn list corresponding to the correct rule 
                        elif change_type == 'add':
                            # TODO - this is again hardcoded to our particular precond types. Need to figure out a centralized way to do this
                            # For now, we're going to make it always satisfiable, but we should actually loosen this constraint at some point
                            while True:
                                # TODO - hardcoded 1
                                precond_type = random.choice([TripletInfo.REQUIRES_ITEM_X % 1, TripletInfo.REQUIRES_LOCATION])
                                item = random.choice(list(spawn_items.keys()))

                                # Make sure item type matches the condition type
                                # TODO - remove this restriction later too
                                if precond_type == TripletInfo.REQUIRES_ITEM_X % 1 and item_dict[item] != 'CraftingItem':
                                    continue
                                elif precond_type == TripletInfo.REQUIRES_LOCATION and item_dict[item] == 'CraftingItem':
                                    continue
                            
                                # Fails if that precondition already exists
                                # TODO - actually, make it if it is in it at all (TODO - relax this later)
                                if any([precond[0] == precond_type and precond[1] == item for precond in preconds]):
                                    continue
                                # TODO TODO DEBUG
                                if any([precond[1] == item for precond in preconds]):
                                    continue
                                
                                # TODO - force it to fail for location if we have a location precondition already
                                # This helps prevent impossible preconds. Change this at some point to be okay?
                                if precond_type == TripletInfo.REQUIRES_LOCATION and any([precond[0] == TripletInfo.REQUIRES_LOCATION for precond in preconds]):
                                    continue
                                break

                            # If all this succeeds, now we can add it
                            new_precond = (precond_type, item)
                            preconds.append(new_precond)
                            changed.append(True)
                            change_success = True

                        # If it's swap, choose an original precondition and change the item or action
                        elif change_type == 'swap':
                            # Choose which piece of knowledge to change
                            swap_ind = random.choice(range(len(preconds)))
                            if changed[swap_ind]:
                                continue
                            precond = preconds[swap_ind]
                            # Change the action
                            if precond[0] == TripletInfo.REQUIRES_ACTION:
                                # TODO - again, this seems like this list should be somewhere else
                                possible_actions = ['mine', 'craft', 'grab', 'chop']
                                possible_actions.pop(possible_actions.index(precond[1]))
                                new_action = random.choice(possible_actions)
                                new_precond = (precond[0], new_action)
                            # Change the item at location
                            elif precond[0] == TripletInfo.REQUIRES_LOCATION:
                                while True:
                                    item = random.choice(list(spawn_items.keys()))

                                    # Make sure item type matches the condition type
                                    # TODO - remove this restriction later too
                                    if item_dict[item] == 'CraftingItem':
                                        continue
                                    # Also make sure it's a change
                                    elif item == precond[1]:
                                        continue
                                    else:
                                        break
                                new_precond = (precond[0], item)
                            elif precond[0] == TripletInfo.REQUIRES_ITEM_X % 1:
                                while True:
                                    item = random.choice(list(spawn_items.keys()))

                                    # Make sure item type matches the condition type
                                    # TODO - remove this restriction later too
                                    if item_dict[item] != 'CraftingItem':
                                        continue
                                    else:
                                        break
                                new_precond = (precond[0], item)
                            else:
                                pdb.set_trace()
                                raise Exception("Bad precond type")
                        
                            # Swap the precond
                            preconds[swap_ind] = new_precond
                            changed[swap_ind] = True
                            change_success = True
               
                # Add the new rule to proposed_knowledge
                rule_name = true_rule[Rule.RULE_NAME]
                triplets = set([(rule_name, pre[0], pre[1]) for pre in preconds]).union([(rule_name, post[0], post[1]) for post in postconds]) 
                new_rule = Rule(create_rule_from_triplets(set(triplets)))
                proposed_knowledge.append(new_rule.rule_dict)
    else:
        raise Exception("TODO - need to implement the random version of this where we randomly break the true knowledge")

    # Make the world
    map_size = (env_opt['state_rep']['w'], env_opt['state_rep']['w'], env_opt['state_rep']['h'], env_opt['state_rep']['h'])
    all_games = [games.BasicKnowledgeGame(world_knowledge=world_knowledge, proposed_knowledge=proposed_knowledge, options=env_opt, map_size=map_size)]

    # Game wrapper
    game = games.MazeGame(
        all_games,
        #featurizer=featurizers.SentenceFeaturesRelative(
        #   max_sentences=30, bounds=4)
        featurizer=featurizers.GridFeaturizer()
    )

    # Create the knowledge planner
    # TODO - maybe game should not be passed in since it contains priviledged info
    # Shoud at least be careful. Definitely don't save game within KnowledgePlanner
    # TODO - should maybe make this more generic at some point to run baselines with this file too
    planner = KnowledgePlanner(game, method_opt, env_opt, proposed_knowledge, world_knowledge['objects'])

    # Load from checkpoint (if applicable)
    # Save options and git information
    if len(args.resume) > 0:
        assert(os.path.isfile(os.path.join(logpath, args.resume)))
        ckpt = torch.load(os.path.join(logpath, args.resume))

        # Get start episode
        if args.train_mode == 'train':
            start_episode = ckpt['train_episode']
        else:
            raise Exception('Training mode %s is not implemented right now' % args.train_mode)    

        # Load planner models
        planner.load_state_dict(ckpt['planner'])

        # Save new options, args, and git
        option_savename = os.path.basename(args.path_opt).split('.')[0] + '_%s_ep%d.yaml' % (args.train_mode, start_episode)
        with open(os.path.join(logpath, options_savename), 'w') as f:
            yaml.dump(options, f, default_flow_style=False)
        args_savename = 'args_%s_ep%d.yaml' % (args.train_mode, start_episodes)
        with open(os.path.join(logpath, args_savename), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

        # Save git info as well
        os.system('git status > %s' % os.path.join(logpath, 'git_status_%s_ep%d.txt' % (args.train_mode, start_episode)))
        os.system('git diff > %s' % os.path.join(logpath, 'git_diff_%s_ep%d.txt' % (args.train_mode, start_episode)))
        os.system('git show > %s' % os.path.join(logpath, 'git_show_%s_ep%d.txt' % (args.train_mode, start_episode)))
    else:
        start_episode = 0

        # Save options and args
        option_savename = os.path.basename(args.path_opt).split('.')[0] + '_%s.yaml' % args.train_mode
        with open(os.path.join(logpath, os.path.basename(args.path_opt)), 'w') as f:
            yaml.dump(options, f, default_flow_style=False)
        args_savename = 'args_%s.yaml' % args.train_mode
        with open(os.path.join(logpath, args_savename), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

        # Save git info as well
        os.system('git status > %s' % os.path.join(logpath, 'git_status_%s.txt' % args.train_mode))
        os.system('git diff > %s' % os.path.join(logpath, 'git_diff_%s.txt' % args.train_mode))
        os.system('git show > %s' % os.path.join(logpath, 'git_show_%s.txt' % args.train_mode))

    # Reset the game
    game.reset()
    max_w, max_h = game.get_max_bounds()
    vocab = dict([(b, a) for a, b in enumerate(game.all_possible_features())])
    list_vocab = list(game.all_possible_features())

    # Reset the planner state
    first_obs = get_obs(game, vocab, list_vocab)
    planner.reset(None, first_obs) 

    # Display the game, if show_game is on
    if args.show_game:
        game.display()
        sleep(.1)
        system('clear')

    # TODO - add other training modes here when necessary
    if args.train_mode == 'train':
        # Main training loop
        frame = 0
        episode_count = start_episode
        episode_start_time = time.time()
        while episode_count < env_opt['num_episodes']:
            # TODO - figure out how we want to do logging
            # TODO - Log / print here
            #print("r: {}\ttr: {} \tguess: {}".format(
            #    game.reward(), game.reward_so_far(), game.approx_best_reward()))
        
            # Get observation from environment
            obs = get_obs(game, vocab, list_vocab)

            # Display game (if in display mode)
            if args.show_game:
                game.display()

            # Log the observation
            main_logger.debug("Episode %d, step %d" % (episode_count, frame))
            #main_logger.debug(obs['grid_obs'])
            #main_logger.debug(obs['side_info'])

            # Get action from planner       
            #id = game.current_agent()
            # TODO - maybe obs doesn't contain the right stuff. Deal with this later
            # TODO - need to pass in game? (Try to avoid to prevent privildged information)
            action = planner.get_action(obs) 

            # Log the action
            #main_logger.debug(action)

            # Take the planned action
            game.act(action)

            # Sleep and clear (if in display mode)
            if args.show_game:
                sleep(.1)
                system('clear')
                print("\n")

            # Update frame and possibly do reset
            frame += 1
            if game.is_over() or frame > env_opt['episode_num_steps']:
                print("Finished episode %d in %f seconds" % (episode_count+1, time.time()-episode_start_time))
                episode_start_time = time.time()

                # Get the last observation
                last_obs = get_obs(game, vocab, list_vocab)

                # Update counters
                frame = 0
                episode_count += 1

                # TODO - add / change print/logging here
                #print("Final reward is: {}, guess was {}".format(
                #    game.reward_so_far(), game.approx_best_reward()))

                # TODO - uncomment this?
                #game.make_harder()

                # Reset game and planner
                game.reset()
                first_obs = get_obs(game, vocab, list_vocab)
                planner.reset(last_obs, first_obs)

                # Save models
                save_checkpoint(logpath, planner, episode_count, args.save_every)
    else:
        raise Exception('Training mode %s is not implemented right now' % args.train_mode)    

# Get observation from game
def get_obs(game, vocab, list_vocab):
    config = game.observe()
    grid_obs, side_info = config['observation']
    one_hot = featurizers.grid_one_hot(game, grid_obs, np, vocab)
    one_hot_side = featurizers.vocabify(game, side_info, np, vocab)
    observation = {}
    observation['grid_obs'] = grid_obs
    observation['side_info'] = side_info
    observation['grid_one_hot'] = one_hot
    observation['side_one_hot'] = one_hot_side
    observation['vocab'] = list_vocab
    return observation

# Save checkpoint
def save_checkpoint(logpath, planner, episode_count, save_every, final=False):
    # Get the checkpoint info
    ckpt = planner.state_dict()
    ckpt['train_episode'] = episode_count  
    ckpt['planner'] = planner.state_dict()

    # Finally save the checkpoint
    ckpt_path = os.path.join(logpath, 'ckpt.pth.tar')
    torch.save(ckpt, ckpt_path)

    # Copy checkpoint if in save_every or final
    if final:
        final_path = os.path.join(logpath, 'final_ckpt.pth.tar')
        shutil.copyfile(ckpt_path, final_path)
    if episode_count % save_every == 0:
        int_path = os.path.join(logpath, 'ckpt_%d.pth.tar' % episode_count)
        shutil.copyfile(ckpt_path, int_path)
    return

if __name__ == '__main__':
    main()

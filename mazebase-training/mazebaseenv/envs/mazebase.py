#mazebase imports
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
import copy
import sys
import os
import time
import random
import yaml
import csv
import ast
import pickle
import codecs
from flask import Flask, render_template, request, jsonify, make_response, json

import torch
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import mazebasev2.lib.mazebase.games as games
from mazebasev2.lib.mazebase.games import featurizers
from mazebasev2.lib.mazebase.items.terrain import CraftingItem, CraftingContainer, ResourceFont, Block, Water, Switch, Door
from mazebasev2.lib.mazebase.items import agents

import torchtext.vocab as vocabtorch


def get_summed_embedding(phrase, glove, embed_size):

    phrase = phrase.split(' ')
    phrase_vector = np.zeros((embed_size), dtype=np.float32)

    for p in phrase:
        phrase_vector += glove[p.lower()]

    return phrase_vector

def get_inventory_embedding(inventory, glove, embed_size):

    inventory_embedding = np.zeros((embed_size), dtype=np.float32)

    first = True
    for item in inventory:

        if first:
            inventory_embedding = get_summed_embedding(item, glove, embed_size)
            first = False
        else:
            inventory_embedding = inventory_embedding + get_summed_embedding(item, glove, embed_size)

    return inventory_embedding
    '''

	inventory_embedding = np.zeros((10,embed_size), dtype=np.float32)

	count = 0
	for item in inventory:
	    if inventory[item] > 0:
	        inventory_embedding[count] = get_summed_embedding(item, glove, embed_size)
	        count = count + 1
	return inventory_embedding
	'''

# input: batched mazebase grid 
# output: 
def get_grid_embedding(batch_grid, glove, embed_size):

    goal_embedding_array = np.zeros((5, 5, embed_size), dtype=np.float32)

    for x in range(5):
        for y in range(5):

            for index, item in enumerate(batch_grid[x][y]):
                if item == "ResourceFont" or item == "CraftingContainer" or item == "CraftingItem":
                    goal_embedding_array[x][y] = get_summed_embedding(batch_grid[x][y][index+1], glove, embed_size)
            
    return goal_embedding_array

def get_goal_embedding(goal, glove, embed_size):

    #currently all crafts are 2 word phrases
    # goal in the format of "Make Diamond Boots (Diamond Boots=1)" --> just extract diamond boots part

    goal_embedding = np.zeros((embed_size), dtype=np.float32)

    goal = goal.split(' ')


    item1_vec = glove[goal[1].lower()]
    item2_vec = glove[goal[2].lower()]

    goal_embedding = item1_vec+item2_vec

    return goal_embedding

def one_hot_grid(grid, glove, embed_size):

    grid_embedding_array = np.zeros((5, 5, 7), dtype=np.float32)

    for x in range(5):
        for y in range(5):

            for index, item in enumerate(grid[x][y]):

                if item == 'Corner':
                    grid_embedding_array[x][y][0] = 1
                elif item == 'Agent':
                    grid_embedding_array[x][y][1] = 1
                elif 'Door' in item:
                    grid_embedding_array[x][y][2] = 1
                elif item == 'Key':
                    grid_embedding_array[x][y][3] = 1
                elif item == 'Switch':
                    grid_embedding_array[x][y][4] = 1
                elif item == 'Block':
                    grid_embedding_array[x][y][5] = 1
                elif item == 'closed': # door closed
                    grid_embedding_array[x][y][6] = 1

    return grid_embedding_array

class MazebaseGame(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, lang_model, device, vocabulary, vocab_weights):

		self.embed_size = 300
		if self.embed_size == 50:
			glove = vocabtorch.GloVe(name='6B', dim=50)
		else:
			glove = vocabtorch.GloVe(name='840B', dim=300)

		vocab = ['Gold', 'Ore', 'Vein', 'Key', "Pickaxe", "Iron", "Diamond", "Boots", "Station", "Brick", "Stairs", "Station", "Factory", "Cobblestone", "Stash", "Sword", "Ingot", "Coal", "Leggins", "Leggings", "Leather", "Rabbit", "Hide", "Chestplate", "Helmet", "Wood", "Plank", "Door", "Tree", "Wooden", "Axe", "Stick", "Stone"]

		self.glove = {}

		for word in vocab:
			try:
				self.glove[word.lower()] = glove.vectors[glove.stoi[word.lower()]].data.cpu().numpy()
			except:
				print(word, "not found")

		#initialize the game here#
		#yaml_file = 'mazebasev2/options/knowledge_planner/minimum_viable_rl.yaml'
		#yaml_file = 'mazebasev2/options/knowledge_planner/length12task.yaml'
		#yaml_file = 'mazebasev2/options/knowledge_planner/length123task.yaml'
		#yaml_file = 'mazebasev2/options/knowledge_planner/length12345task.yaml'
		#yaml_file = 'mazebasev2/options/knowledge_planner/length12345task_missing.yaml'
		
		# only 3 and only 5 tasks..
		#yaml_file = 'mazebasev2/options/knowledge_planner/length35task.yaml'
		#yaml_file = 'mazebasev2/ptions/knowledge_planner/length2task.yaml'
		#yaml_file = 'mazebasev2/options/knowledge_planner/length1task.yaml'
		#yaml_file = 'mazebasev2/options/knowledge_planner/length3task.yaml'
		#yaml_file = 'mazebasev2/options/knowledge_planner/length45common.yaml'
		yaml_file = 'mazebasev2/options/knowledge_planner/unseen_tasks.yaml'

		#yaml_file = 'mazebasev2/options/knowledge_planner/length12task_distractor.yaml'

		with open(yaml_file, 'r') as handle:
			options = yaml.load(handle)

		# Get sub opts
		method_opt = options['method']
		env_opt = options['env']
		log_opt = options['logs'] 

		# Set up the mazebase environment
		knowledge_root = env_opt['knowledge_root']
		world_knowledge_file = os.path.join('mazebasev2', knowledge_root, env_opt['world_knowledge']['train'])
		with open(world_knowledge_file) as f:
		  world_knowledge = json.load(f)

		# Make the world
		map_size = (env_opt['state_rep']['w'], env_opt['state_rep']['w'], env_opt['state_rep']['h'], env_opt['state_rep']['h'])
		self.all_games = [games.BasicKnowledgeGame(world_knowledge=world_knowledge, proposed_knowledge=[], options=env_opt, load_items=None, map_size=map_size)]

		# For 1 step tasks!
		knowledge_root = env_opt['knowledge_root']
		world_knowledge_file1 = os.path.join('mazebasev2', knowledge_root, 'data_1.json')
		with open(world_knowledge_file1) as f:
		  world_knowledge1 = json.load(f)
		self.games1 = [games.BasicKnowledgeGame(world_knowledge=world_knowledge1, proposed_knowledge=[], options=env_opt, load_items=None, map_size=map_size)]



		#self.lang_model = lang_model.to(device)
		self.device = device
		self.vocabulary = vocabulary
		self.vocab_weights = vocab_weights

		self.reset()
		
		self.action_space = gym.spaces.Discrete(8)
		dims = self.state.shape
		self.observation_space = spaces.Box(low=0, high=1, shape=(dims))

		#print(self.state.shape)
		#self.observation_space = gym.spaces.Discrete(self.state.shape[0])

	def step(self, action):

		self.count = self.count + 1

		if action == 0:
			action = 'up'
		elif action == 1:
			action = 'down'
		elif action == 2:
			action = 'right'
		elif action == 3:
			action = 'left'
		elif action == 4:
			action = 'toggle_switch'
		elif action == 5:
			action = 'grab'
		elif action == 6:
			action = 'mine'
		elif action == 7:
			action = 'craft'


		#execute action
		self.game.act(action)

		#print(self.game.game.inventory)

		## more rewards -- if has pickaxe, and nearby the item to mine.

		hasPickaxe = False
		for item in self.game.game.inventory:
			if item == 'Pickaxe':
				hasPickaxe = True

		#calculate reward
		if self.game.is_over():
			self.reward = 5
			self.done = True
			self.count = 0
		else:
			self.reward = 0
			self.done = False

		#extra info
		self.add = {}
		#self.add['episode'] = self.game.game.inventory
		self.add['episode'] = {'l':self.count,'r':self.reward}


		#get observation
		config = self.game.observe()
		grid_obs, side_info = config['observation']
		
		inventory = self.game.game.inventory
		goal = self.game.game.goal

		obs = (grid_obs, inventory, goal)

		state, inventory, goal = obs

		states_embedding = get_grid_embedding(state, self.glove, self.embed_size)
		states_onehot = one_hot_grid(state, self.glove, self.embed_size)
		goal = get_goal_embedding(goal, self.glove, self.embed_size)
		inventory = get_inventory_embedding(inventory, self.glove, self.embed_size)

		#k_prev_words = np.array([[self.vocabulary.word2idx['<start>']]])
		#top_k_scores = np.zeros((1, 1))
		#self.state = np.concatenate((k_prev_words.flatten(), top_k_scores.flatten(), states_embedding.flatten(), states_onehot.flatten(), goal.flatten(), inventory.flatten()))

		counts = np.array([self.game.game.count])
		self.state = np.concatenate((counts.flatten(), states_embedding.flatten(), states_onehot.flatten(), goal.flatten(), inventory.flatten()))

		#self.state = np.concatenate((states_embedding.flatten(), states_onehot.flatten(), goal.flatten(), inventory.flatten()))

		# all_sampled_ids = self.lang_model.get_hidden_state(states_embedding, states_onehot, inventory, goal, self.device, self.vocabulary)
		# bow_ids = [sent + [len(vocab)] * (20 - len(sent)) for sent in all_sampled_ids]
		# bow_ids = torch.Tensor(bow_ids)
		# bow_ids = bow_ids.long()
		# sampled_ids = bow_ids.to(self.device)

		#sampled_ids, hidden_state = self.lang_model.get_hidden_state(states_embedding, states_onehot, inventory, goal, self.device, self.vocabulary)		
		#sampled_ids = sampled_ids + [-1]*(32-len(sampled_ids))
		#sampled_ids = np.array(sampled_ids)
		#hidden_state = hidden_state.data.numpy()
		#self.state = np.concatenate((sampled_ids.flatten(), states_embedding.flatten(), states_onehot.flatten(), goal.flatten(), inventory.flatten()))

		#get observation
		# config = self.game.observe()
		# grid_obs, side_info = config['observation']
		# vocab = dict([(b, a) for a, b in enumerate(self.game.all_possible_features())])
		# one_hot = featurizers.grid_one_hot(self.game, grid_obs, np, vocab)
		# self.state = one_hot.flatten()

		return [self.state, self.reward, self.done, self.add]

	def reset(self):

		self.count = 0

		# Game wrapper
		self.game = games.MazeGame(
		  self.all_games,
		  featurizer=featurizers.GridFeaturizer()
		)

		#get observation
		config = self.game.observe()
		grid_obs, side_info = config['observation']
		
		inventory = self.game.game.inventory
		goal = self.game.game.goal

		print(goal)

		obs = (grid_obs, inventory, goal)

		state, inventory, goal = obs

		states_embedding = get_grid_embedding(state, self.glove, self.embed_size)
		states_onehot = one_hot_grid(state, self.glove, self.embed_size)
		goal = get_goal_embedding(goal, self.glove, self.embed_size)
		inventory = get_inventory_embedding(inventory, self.glove, self.embed_size)
		counts = np.array([self.game.game.count])
		self.state = np.concatenate((counts.flatten(), states_embedding.flatten(), states_onehot.flatten(), goal.flatten(), inventory.flatten()))

		#k_prev_words = np.array([[self.vocabulary.word2idx['<start>']]])
		#top_k_scores = np.zeros((1, 1))

		#print(top_k_scores)

		#print(k_prev_words.flatten())
		#print(top_k_scores.flatten())

		#no language:
		#self.state = np.concatenate((k_prev_words.flatten(), top_k_scores.flatten(), states_embedding.flatten(), states_onehot.flatten(), goal.flatten(), inventory.flatten()))

		#temp_embedding = torch.Tensor(states_embedding).to(self.device)
		#temp_onehot = torch.Tensor(states_onehot).to(self.device)
		#temp_inv = torch.Tensor(inventory).to(self.device)
		#temp_goal = torch.Tensor(goal).to(self.device)

		# ADD LANGUAGE MODEL STUFF!!
		#all_sampled_ids = self.lang_model.get_hidden_state(temp_embedding, temp_onehot, temp_inv, temp_goal, self.device, self.vocabulary)
		#bow_ids = [sent + [len(self.vocabulary)] * (20 - len(sent)) for sent in all_sampled_ids]
		#bow_ids = np.array(bow_ids)

		#sampled_ids, hidden_state = self.lang_model.get_hidden_state(states_embedding, states_onehot, inventory, goal, self.device, self.vocabulary)
		#sampled_ids = sampled_ids + [-1]*(32-len(sampled_ids))
		#sampled_ids = np.array(sampled_ids)
		#hidden_state = hidden_state.data.numpy()
		#self.state = np.concatenate((bow_ids.flatten(), states_embedding.flatten(), states_onehot.flatten(), goal.flatten(), inventory.flatten()))

		return self.state

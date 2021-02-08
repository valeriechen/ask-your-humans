import pandas as pd
import statistics
import fire
import ast
from os import system
import json
import re
import os
import numpy as np
import random

from spellchecker import SpellChecker

#returns true if the neighboring coordinate to x,y contains a door.
def checkNeighbor(grid, x, y):

	if x-1 >= 0:
		if 'Door' in grid[x-1][y]:
			return True

	if x+1 < 5:
		if 'Door' in grid[x+1][y]:
			return True

	if y-1 >= 0:
		if 'Door' in grid[x][y-1]:
			return True

	if y+1 < 5:
		if 'Door' in grid[x][y+1]:
			return True

	return False


## Input: json file of game traces, train/test split
## Output: train/test sets for board states, inventories, and actions

def read_dataset(filename, split):

	spell = SpellChecker()

	with open(filename) as f:
		dataset = json.load(f)

	print("**dataset loading**")

	states = []
	inventories = []
	actions = []
	goals = []
	instructions = []

	all_instructions = []

	game_counts = {}
	instr_counts = {}
	action_counts = {}

	#read in all traces

	for trace in dataset:

		# not sure why this one is problematic
		if trace == "AP9GRAU77J5G.619158":
			continue

		game = dataset[trace]
		game = str(game)
		game = ast.literal_eval(game)

		temp_compile = []

		for indiv_game in game:

			current_instruction = None
			toggled = False
			door_opened = False
			had_key = False
			game_count = 0
			#print("--")

			for i in range(len(indiv_game)):

				temp = indiv_game[i]
				if i == 0:
					game_count = temp['extra_items']['count']
					if game_count in game_counts:
						game_counts[game_count] = game_counts[game_count]+1
					else:
						game_counts[game_count] = 1

				
				## need to do this because when the json was saved, it is the resulting state, so need the previous state
				if isinstance(temp, dict):
					if i > 0:

						if game_count in action_counts:
							action_counts[game_count] = action_counts[game_count]+1
						else:
							action_counts[game_count] = 1


						temp_compile.append(temp['action'])
						temp_compile.append('blah')
						#temp_compile.append(current_instruction)

						if temp['action'] == 'toggle_switch':
							toggled = True

					grid = temp['observation'][0]
					
					#make door open
					if toggled and not door_opened:

						for x in range(5):
							for y in range(5):
								for index, item in enumerate(grid[x][y]):

									#if in the same grid as the switch
									if item == 'Switch':
										if 'Agent' in grid[x][y]:
											door_opened = True

									#if in grid next to door and has a key
									if item == 'Agent':
										if 'Key' in temp['inventory'] or 'key' in temp['inventory']:
											if checkNeighbor(grid, x, y):
												door_opened = True

					#make door whatever door_opened is
					
					for x in range(5):
						for y in range(5):
							for index, item in enumerate(grid[x][y]):

								if item == 'Door':
									if door_opened:
										#print("opened")
										grid[x][y][index] = 'Door_opened'
									else:
										#print("closed")
										grid[x][y][index] = 'Door_closed'


					temp_compile.append([grid,temp['inventory'],temp['goal']])

				if isinstance(temp, str):

					#print(temp)

					temp = temp.strip()
					if game_count in instr_counts:
						instr_counts[game_count] = instr_counts[game_count] + 1
					else:
						instr_counts[game_count] = 1

					# do spelling correction for each word:
					#updated_temp = [spell.correction(word.lower()) for word in temp.split(' ')]
					
					#all_instructions.append(updated_temp)

					#current_instruction = updated_temp

			#temp_compile = temp_compile[:-1] adding stop action
			temp_compile.append("stop")
			temp_compile.append(current_instruction)

			for i in range(0, len(temp_compile), 3):
				states.append(temp_compile[i][0])
				inventories.append(temp_compile[i][1])
				goals.append(temp_compile[i][2])
				actions.append(temp_compile[i+1])
				instructions.append(temp_compile[i+2])

	print(len(states), len(inventories), len(actions), len(goals), len(instructions))


	print(game_counts)
	print(instr_counts)
	print(action_counts)

	##split into training and validation

	assert (len(states) == len(inventories) == len(actions) == len(goals)) # == len(instructions)

	num_train = int(float(split)*len(states))

	states = np.array(states)
	inventories = np.array(inventories)
	actions = np.array(actions)
	goals = np.array(goals)
	#instructions = np.array(instructions)

	#print(instructions)

	# indices = random.sample(range(len(states)), num_train)

	# train_states = states[indices]
	# test_states = np.delete(states,indices, axis=0)

	# train_inventories = inventories[indices]
	# test_inventories = np.delete(inventories,indices, axis=0)

	# train_actions = actions[indices]
	# test_actions = np.delete(actions,indices, axis=0)

	# train_goals = goals[indices]
	# test_goals = np.delete(goals,indices, axis=0)

	# train_instructions = instructions[indices]
	# test_instructions = np.delete(instructions,indices, axis=0)

	print("**dataset loaded**")

	return states, inventories, actions, goals #, instructions, all_instructions

	#return train_states, train_inventories, train_actions, train_goals, train_instructions, test_states, test_inventories, test_actions, test_goals, test_instructions, all_instructions

	#print(len(train_inventories), len(test_inventories))



## Input: json file of game traces, train/test split
## Output: train/test sets for state, language

def load_state_language(filename, split):

	with open(filename) as f:
		dataset = json.load(f)

	print("**dataset loading**")

	states = []
	instructions = []

	#read in all traces

	for trace in dataset:

		# not sure why this one is problematic
		if trace == "AP9GRAU77J5G.619158":
			continue

		game = dataset[trace]
		game = str(game)
		game = ast.literal_eval(game)

		temp_compile = []

		for indiv_game in game:

			for i in range(len(indiv_game)):

				temp = indiv_game[i]
				
				## need to do this because when the json was saved, it is the resulting state, so need the previous state
				if isinstance(temp, dict):
					if i > 0:
						temp_compile.append(temp['action'])

					temp_compile.append([temp['observation'][0],temp['inventory'],temp['goal']])

			temp_compile = temp_compile[:-1]

			for i in range(0, len(temp_compile), 2):
				states.append(temp_compile[i][0])
				inventories.append(temp_compile[i][1])
				goals.append(temp_compile[i][2])
				actions.append(temp_compile[i+1])


	print(len(states), len(inventories), len(actions), len(goals))

	##split into training and validation

	assert (len(states) == len(inventories) == len(actions) == len(goals))

	num_train = int(float(split)*len(states))

	states = np.array(states)
	inventories = np.array(inventories)
	actions = np.array(actions)
	goals = np.array(goals)

	indices = random.sample(range(len(states)), num_train)

	train_states = states[indices]
	test_states = np.delete(states,indices, axis=0)

	train_inventories = inventories[indices]
	test_inventories = np.delete(inventories,indices, axis=0)

	train_actions = actions[indices]
	test_actions = np.delete(actions,indices, axis=0)

	train_goals = goals[indices]
	test_goals = np.delete(goals,indices, axis=0)

	print("**dataset loaded**")

	return train_states, train_inventories, train_actions, train_goals, test_states, test_inventories, test_actions, test_goals


#read_dataset('data/example.json', 0.8)

read_dataset('data/compiled_dataset_081319.json', 0.8)





import pandas as pd
import statistics
import fire
import ast
from os import system
import json
import re
import os
from scipy.ndimage import rotate
import numpy as np

#earlier tests
#ar = np.array([[[1,2,3],[4,5,6], [7,8,9]], [[1,2,3],[4,5,6], [7,8,9]],[[1,2,3],[4,5,6], [7,8,9]]], int)

#print(ar)

#ar = rotate(ar, 90)
#ar = np.rot90(ar, 1)

#print(ar)

#m1 = rotate(m, 0)
#m2 = rotate(m, -90)
#m3 = np.flip(m, 1) #rotate and flip. 

#print(m)
#print(m1)
#print(m2)
#print(m3)

def rotate_action(action, rot=0, flip=False):

	if rot == 90 and not flip:
		mapping = {'up': 'right', 'down': 'left', 'left': 'up', 'right':'down'}

	if rot == 180 and not flip:
		mapping = {'up': 'down', 'down': 'up', 'left': 'right', 'right':'left'}

	if rot == 270 and not flip:
		mapping = {'up': 'left', 'down': 'right', 'left': 'down', 'right':'up'}

	if rot == 0 and flip:
		mapping = {'up': 'up', 'down': 'down', 'left': 'right', 'right':'left'}

	if rot == 90 and flip:
		mapping = {'up': 'right', 'down': 'left', 'left': 'down', 'right':'up'}

	if rot == 180 and flip:
		mapping = {'up': 'down', 'down': 'up', 'left': 'left', 'right':'right'}

	if rot == 270 and flip:
		mapping = {'up': 'left', 'down': 'right', 'left': 'up', 'right':'down'}


	if action[0] == 1:
		lang_action = 'up'
	elif action[0] == 2:
		lang_action = 'down'
	elif action[0] == 3:
		lang_action = 'left'
	elif action[0] == 4:
		lang_action = 'right'
	elif action[0] == 5:
		lang_action = 'toggle_switch'
	elif action[0] == 6:
		lang_action = 'grab'
	elif action[0] == 7:
		lang_action = 'mine'
	elif action[0] == 0:
		lang_action = 'craft'
	elif action[0] == 8:
		lang_action = 'stop'

	if lang_action in mapping:
		new_action = mapping[lang_action]
	else:
		return action

	if new_action == 'up':
		return np.array([1])
	elif new_action == 'down':
		return np.array([2])
	elif new_action == 'left':
		return np.array([3])
	elif new_action == 'right':
		return np.array([4])
	elif new_action == 'toggle_switch':
		return np.array([5])
	elif new_action == 'grab':
		return np.array([6])
	elif new_action == 'mine':
		return np.array([7])
	elif new_action == 'craft':
		return np.array([0])
	elif new_action == 'stop':
		return np.array([8])


	#if action in mapping:
	#	new_action = mapping[action]

	#return new_action


def rotate_board(board, rot=0, flip=False):

	#rotatedGrid = deepcopy(grid)

	board = rotate(board, rot)

	'''
	if rot > 0:

		if rot == 90:
			m = 1
		elif rot == 180:
			m = 2
		elif rot == 270:
			m = 3

		board = np.rot90(board, m)
	'''

	if flip:
		board = np.flip(board, 1)

	return board


def rotate_trace(trace, rot=0, flip=False):

	for i in range(len(trace)):

		if isinstance(trace[i], dict):

			if 'action' in trace[i]:

				if rot == 90 and not flip:
					mapping = {'up': 'right', 'down': 'left', 'left': 'up', 'right':'down'}
					if trace[i]['action'] in mapping:
						trace[i]['action'] = mapping[trace[i]['action']]

				if rot == 180 and not flip:
					mapping = {'up': 'down', 'down': 'up', 'left': 'right', 'right':'left'}
					if trace[i]['action'] in mapping:
						trace[i]['action'] = mapping[trace[i]['action']]

				if rot == 270 and not flip:
					mapping = {'up': 'left', 'down': 'right', 'left': 'down', 'right':'up'}
					if trace[i]['action'] in mapping:
						trace[i]['action'] = mapping[trace[i]['action']]

				if rot == 0 and flip:
					mapping = {'up': 'up', 'down': 'down', 'left': 'right', 'right':'left'}
					if trace[i]['action'] in mapping:
						trace[i]['action'] = mapping[trace[i]['action']]

				if rot == 90 and flip:
					mapping = {'up': 'right', 'down': 'left', 'left': 'down', 'right':'up'}
					if trace[i]['action'] in mapping:
						trace[i]['action'] = mapping[trace[i]['action']]

				if rot == 180 and flip:
					mapping = {'up': 'down', 'down': 'up', 'left': 'left', 'right':'right'}
					if trace[i]['action'] in mapping:
						trace[i]['action'] = mapping[trace[i]['action']]

				if rot == 270 and flip:
					mapping = {'up': 'left', 'down': 'right', 'left': 'up', 'right':'down'}
					if trace[i]['action'] in mapping:
						trace[i]['action'] = mapping[trace[i]['action']]

			trace[i]['board'] = rotate(trace[i]['board'], rot)
			if flip:
				trace[i]['board'] = np.flip(trace[i]['board'], 1)


	return trace




#later: replace the wall?

def translate_trace(trace, direction):

	temp = [[] for i in range(6)]

	if direction == 'left':

		#check all leftmost columns are empty

		isEmpty = True
		board = trace[0]['board']
		col = board[:,0]

		for item in col:
			if len(item) > 0:
				isEmpty = False
				break

		if isEmpty:

			#iterate through, then for each board, do this operation
			board = np.delete(board, 0, axis=1)
			#append empty onto end
			board = np.hstack((board,temp))


	if direction == 'right':

		# check all rightmost columns are empty
		isEmpty = True
		board = trace[0]['board']
		col = board[:,4]

		for item in col:
			if len(item) > 0:
				isEmpty = False
				break

		if isEmpty:

			#iterate through, then for each board, do this operation
			board = np.delete(board, 4, axis=1)
			#append empty onto beginning
			board = np.hstack((temp,board))

	if direction == 'up':

		# check all topmost row is empty

		isEmpty = True
		board = trace[0]['board']
		row = board[0,:]

		for item in row:
			if len(item) > 0:
				isEmpty = False
				break

		if isEmpty:

			#iterate through, then for each board, do this operation

			board = np.delete(board, 0, axis=0)
			#append empty onto beginning
			board.append(temp)

	if direction ==  'down':

		#check all bottommost row is empty

		isEmpty = True
		board = trace[0]['board']
		row = board[4,:]

		for item in row:
			if len(item) > 0:
				isEmpty = False
				break

		if isEmpty:
			
			#iterate through, then for each board, do this operation
			board = np.delete(board, 4, axis=0)
			#append empty onto beginning
			board.insert(0, temp)

#def add_distractor_items_trace(trace, num):

	# look at items that are on board, add num items to the. board



#open json file 

# with open("compiled_dataset_081219.json") as f:
# 	dataset = json.load(f)


# # each HIT:
# for trace in dataset:

# 	game = dataset[trace]
# 	game = str(game)
# 	game = ast.literal_eval(game)

	

# 	# each game
# 	for indiv_game in game:

# 		augment_trace(indiv_game, 90, False)
# 		#augment_trace(in)

# can do rotate and then translate OR translate then rotate...?





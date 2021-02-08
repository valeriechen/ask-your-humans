import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchtext.vocab as vocabtorch
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from data_reader import read_dataset
from torchsummary import summary

import random
import os
import yaml
import json

import mazebasev2.lib.mazebase.games as games
from mazebasev2.lib.mazebase.games import featurizers

from mazebasev2.lib.mazebase.items.terrain import CraftingItem, CraftingContainer, ResourceFont, Block, Water, Switch, Door
from mazebasev2.lib.mazebase.items import agents

from train_models import StateGoalNet, StateGoalInstructionNet, LanguageNet, ActionNet, LanguageNetv1, ActionNetv1, ActionNet_Allobs, LanguageNetv2, LanguageWithAttention, AllObsPredictAtten, CNNAction, LanguageWithAttentionGLOVE, SimpleNetwork, LanguageWithAttentionSUM

import nltk
import pickle

from build_vocab import build_vocabulary, load_vocabulary
from test_models import generate_lang, play_game_w_language_v2, play_game_w_language_v3, play_game_w_language_glove
from data_augmentation import rotate_board, rotate_action


class PremadeCraftingDataset(Dataset):

    def __init__(self, embed_dim, train_states, train_onehot, train_inventories, train_actions, train_goals, train_instructions, vocab, transform=None):

        self.embed_dim = embed_dim

        self.vocab = vocab
        self.train_instructions = train_instructions

        self.train_states_embedding = train_states
        self.train_states_onehot = train_states_onehot
        self.train_inventory_embedding = train_inventories
        self.train_actions_onehot = train_actions
        self.train_goals_embedding = train_goals

    def __getitem__(self, index):

        temp_instruction = self.train_instructions[index]

        while temp_instruction == None:
            index = index + 10 
            if index > self.train_states_embedding.shape[0]:
                index = 0
            temp_instruction = self.train_instructions[index]

        instruction = []
        instruction.append(vocab('<start>'))
        instruction.extend([self.vocab(token) for token in temp_instruction])
        instruction.append(vocab('<end>'))
        target = torch.Tensor(instruction)

        states_embedding = torch.Tensor(self.train_states_embedding[index])
        states_onehot = torch.Tensor(self.train_states_onehot[index])
        action = torch.Tensor(self.train_actions_onehot[index])
        goal = torch.Tensor(self.train_goals_embedding[index])
        inventory = torch.Tensor(np.array([self.train_inventory_embedding[index]]))

        '''
        except:

            index = index + 10 # try using the neighboring example instead.. let's see if this breaks. 

            states_embedding = torch.Tensor(self.train_states_embedding[index])
            states_onehot = torch.Tensor(self.train_states_onehot[index])
            action = torch.Tensor(self.train_actions_onehot[index])
            goal = torch.Tensor(self.train_goals_embedding[index])
            inventory = torch.Tensor(np.array([self.train_inventory_embedding[index]]))

            temp_instruction = self.train_instructions[index]
        
            instruction = []
            instruction.append(vocab('<start>'))
            instruction.extend([self.vocab(token) for token in temp_instruction])
            instruction.append(vocab('<end>'))
            target = torch.Tensor(instruction)

            # instruction.append(vocab('<start>'))
            # instruction = [self.vocab('<unk>')]
            # instruction.append(vocab('<end>'))
            # target = torch.Tensor(instruction)
        '''

        return states_onehot, states_embedding, inventory, action, goal, target

    def __len__(self):
        return self.train_states_embedding.shape[0]
        #return len(self.train_states)

class CraftingDataset(Dataset):

    def __init__(self, embed_dim, train_states, train_inventories, train_actions, train_goals, train_instructions, vocab, transform=None):

        self.embed_dim = embed_dim

        self.vocab = vocab
        self.train_instructions = train_instructions

        self.train_states = train_states
        self.train_inventories = train_inventories
        self.train_actions = train_actions
        self.train_goals = train_goals

        if self.embed_dim == 50:
            self.glove = vocabtorch.GloVe(name='6B', dim=50)
        else:
            self.glove = vocabtorch.GloVe(name='840B', dim=300)

        self.train_states_embedding = [self.get_grid_embedding(state) for state in self.train_states]
        print("embedding loaded")
        self.train_states_onehot = [self.one_hot_grid(state) for state in self.train_states]
        print("one hot loaded")
        self.train_actions_onehot = [self.one_hot_actions(action) for action in self.train_actions]
        print("actions loaded")
        self.train_goals_embedding = [self.get_goal_embedding(goal) for goal in self.train_goals]
        print("goals loaded")
        self.train_inventory_embedding = [self.get_inventory_embedding(inventory) for inventory in self.train_inventories]
        print("done loading dataset")

        '''
        augmented_train_states = np.repeat(self.train_states_embedding, repeats=8, axis=0)
        augmented_train_onehot = np.repeat(self.train_states_onehot, repeats=8, axis=0)
        augmented_train_inventories = np.repeat(self.train_inventory_embedding, repeats=8, axis=0)
        augmented_train_actions = np.repeat(self.train_actions_onehot, repeats=8, axis=0)
        augmented_train_goals = np.repeat(self.train_goals_embedding, repeats=8, axis=0)
        augmented_train_instructions = np.repeat(self.train_instructions, repeats=8, axis=0)

        counter = 1
        num = 50000 #202831

        for i in range(0, augmented_train_states.shape[0], 8):

            print(i, augmented_train_states.shape[0])
            augmented_train_states[i+1] = rotate_board(augmented_train_states[i+1], 90, False)
            augmented_train_states[i+2] = rotate_board(augmented_train_states[i+2], 180, False)
            augmented_train_states[i+3] = rotate_board(augmented_train_states[i+3], 270, False)
            augmented_train_states[i+4] = rotate_board(augmented_train_states[i+4], 0, True)
            augmented_train_states[i+5] = rotate_board(augmented_train_states[i+5], 90, True)
            augmented_train_states[i+6] = rotate_board(augmented_train_states[i+6], 180, True)
            augmented_train_states[i+7] = rotate_board(augmented_train_states[i+7], 270, True)

            augmented_train_onehot[i+1] = rotate_board(augmented_train_onehot[i+1], 90, False)
            augmented_train_onehot[i+2] = rotate_board(augmented_train_onehot[i+2], 180, False)
            augmented_train_onehot[i+3] = rotate_board(augmented_train_onehot[i+3], 270, False)
            augmented_train_onehot[i+4] = rotate_board(augmented_train_onehot[i+4], 0, True)
            augmented_train_onehot[i+5] = rotate_board(augmented_train_onehot[i+5], 90, True)
            augmented_train_onehot[i+6] = rotate_board(augmented_train_onehot[i+6], 180, True)
            augmented_train_onehot[i+7] = rotate_board(augmented_train_onehot[i+7], 270, True)

            augmented_train_actions[i+1] = rotate_action(augmented_train_actions[i+1], 90, False)
            augmented_train_actions[i+2] = rotate_action(augmented_train_actions[i+2], 180, False)
            augmented_train_actions[i+3] = rotate_action(augmented_train_actions[i+3], 270, False)
            augmented_train_actions[i+4] = rotate_action(augmented_train_actions[i+4], 0, True)
            augmented_train_actions[i+5] = rotate_action(augmented_train_actions[i+5], 90, True)
            augmented_train_actions[i+6] = rotate_action(augmented_train_actions[i+6], 180, True)
            augmented_train_actions[i+7] = rotate_action(augmented_train_actions[i+7], 270, True)

            if i > counter * num:

                print("SAVED!")

                with open('data/augmented_train_states_'+str(counter), 'wb') as f:
                    pickle.dump(augmented_train_states[(counter-1)*num:counter*num], f)

                with open('data/augmented_train_onehot_'+str(counter), 'wb') as f:
                    pickle.dump(augmented_train_onehot[(counter-1)*num:counter*num], f)

                with open('data/augmented_train_inventories_'+str(counter), 'wb') as f:
                    pickle.dump(augmented_train_inventories[(counter-1)*num:counter*num], f)

                with open('data/augmented_train_actions_'+str(counter), 'wb') as f:
                    pickle.dump(augmented_train_actions[(counter-1)*num:counter*num], f)

                with open('data/augmented_train_goals_'+str(counter), 'wb') as f:
                    pickle.dump(augmented_train_goals[(counter-1)*num:counter*num], f)    
                
                with open('data/augmented_train_instructions_'+str(counter), 'wb') as f:
                    pickle.dump(augmented_train_instructions[(counter-1)*num:counter*num], f)    
            
                counter = counter + 1


        with open('data/augmented_train_states_'+str(counter), 'wb') as f:
            pickle.dump(augmented_train_states[(counter-1)*num:], f)

        with open('data/augmented_train_onehot_'+str(counter), 'wb') as f:
            pickle.dump(augmented_train_onehot[(counter-1)*num:], f)

        with open('data/augmented_train_inventories_'+str(counter), 'wb') as f:
            pickle.dump(augmented_train_inventories[(counter-1)*num:], f)

        with open('data/augmented_train_actions_'+str(counter), 'wb') as f:
            pickle.dump(augmented_train_actions[(counter-1)*num:], f)

        with open('data/augmented_train_goals_'+str(counter), 'wb') as f:
            pickle.dump(augmented_train_goals[(counter-1)*num:], f)    
        
        with open('data/augmented_train_instructions_'+str(counter), 'wb') as f:
            pickle.dump(augmented_train_instructions[(counter-1)*num:], f)    


        print("done augmenting") 
        '''

    # input: multi-word crafting item string
    # output: summed glove word embedding (50d)
    def get_summed_embedding(self, phrase):

        phrase = phrase.split(' ')
        #phrase_vector = torch.from_numpy(np.zeros((self.embed_dim), dtype=np.float32))

        phrase_vector = np.zeros((self.embed_dim), dtype=np.float32)

        for p in phrase:
            phrase_vector += self.glove.vectors[self.glove.stoi[p.lower()]].data.cpu().numpy()

        return phrase_vector

    # input: batched mazebase grid 
    # output: 
    def get_grid_embedding(self, batch_grid):

        goal_embedding_array = np.zeros((5, 5, self.embed_dim), dtype=np.float32)

        for x in range(5):
            for y in range(5):

                for index, item in enumerate(batch_grid[x][y]):
                    if item == "ResourceFont" or item == "CraftingContainer" or item == "CraftingItem":
                        goal_embedding_array[x][y] = self.get_summed_embedding(batch_grid[x][y][index+1])
                
        return goal_embedding_array

    def get_goal_embedding(self, goal):

            #currently all crafts are 2 word phrases
            # goal in the format of "Make Diamond Boots (Diamond Boots=1)" --> just extract diamond boots part
        
            goal_embedding = np.zeros((self.embed_dim), dtype=np.float32)

            goal = goal.split(' ')

            item1_vec = self.glove.vectors[self.glove.stoi[goal[1].lower()]].data.cpu().numpy()
            item2_vec = self.glove.vectors[self.glove.stoi[goal[2].lower()]].data.cpu().numpy()

            goal_embedding = item1_vec+item2_vec

            return goal_embedding

    def get_inventory_embedding(self, inventory):

        
        #summed inventory
        inventory_embedding = np.zeros((self.embed_dim), dtype=np.float32)

        first = True
        for item in inventory:

            if inventory[item] > 0:

                if first:
                    inventory_embedding = self.get_summed_embedding(item)
                    first = False
                else:
                    inventory_embedding = inventory_embedding + self.get_summed_embedding(item)

        return inventory_embedding
        '''

        inventory_embedding = np.zeros((10,self.embed_dim), dtype=np.float32)

        count = 0
        for item in inventory:
            if inventory[item] > 0:
                inventory_embedding[count] = self.get_summed_embedding(item)
                count = count + 1
        return inventory_embedding
        '''


    #TODO: later when adding traces, add stop action at the end
    def one_hot_actions(self, action):

        if action == 'up':
            return np.array([1])
        elif action == 'down':
            return np.array([2])
        elif action == 'left':
            return np.array([3])
        elif action == 'right':
            return np.array([4])
        elif action == 'toggle_switch':
            return np.array([5])
        elif action == 'grab':
            return np.array([6])
        elif action == 'mine':
            return np.array([7])
        elif action == 'craft':
            return np.array([0])
        elif action == 'stop':
            return np.array([8])

        # if action == 'up':
        #     return np.array([1, 0, 0, 0, 0, 0, 0, 0])
        # elif action == 'down':
        #     return np.array([0, 1, 0, 0, 0, 0, 0, 0])
        # elif action == 'left':
        #     return np.array([0, 0, 1, 0, 0, 0, 0, 0])
        # elif action == 'right':
        #     return np.array([0, 0, 0, 1, 0, 0, 0, 0])
        # elif action == 'toggle_switch':
        #     return np.array([0, 0, 0, 0, 1, 0, 0, 0])
        # elif action == 'grab':
        #     return np.array([0, 0, 0, 0, 0, 1, 0, 0])
        # elif action == 'mine':
        #     return np.array([0, 0, 0, 0, 0, 0, 1, 0])
        # elif action == 'craft':
        #     return np.array([0, 0, 0, 0, 0, 0, 0, 1])
        # elif action == 'stop':
        #     return np.array([8])

    def one_hot_grid(self, grid):

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
                    elif item == 'Door_closed':
                        grid_embedding_array[x][y][6] = 1

        return grid_embedding_array


    def __getitem__(self, index):

        # switch this so that it does it all in the beginning, precompute all of this.
        #states_onehot = torch.Tensor(self.one_hot_grid(self.train_states[index]))
        #states_embedding = torch.Tensor(self.get_grid_embedding(self.train_states[index]))
        #action = torch.Tensor(self.one_hot_actions(self.train_actions[index]))
        #goal = torch.Tensor(self.get_goal_embedding(self.train_goals[index]))

        states_embedding = torch.Tensor(self.train_states_embedding[index])
        states_onehot = torch.Tensor(self.train_states_onehot[index])
        action = torch.Tensor(self.train_actions_onehot[index])
        goal = torch.Tensor(self.train_goals_embedding[index])
        inventory = torch.Tensor(self.train_inventory_embedding[index])

        temp_instruction = self.train_instructions[index]

        try:
            instruction = []
            instruction.append(vocab('<start>'))
            instruction.extend([self.vocab(token) for token in temp_instruction])
            instruction.append(vocab('<end>'))
            target = torch.Tensor(instruction)
        except:

            index = index + 10 # try using the neighboring example instead.. let's see if this breaks. 

            states_embedding = torch.Tensor(self.train_states_embedding[index])
            states_onehot = torch.Tensor(self.train_states_onehot[index])
            action = torch.Tensor(self.train_actions_onehot[index])
            goal = torch.Tensor(self.train_goals_embedding[index])
            inventory = torch.Tensor(self.train_inventory_embedding[index])

            temp_instruction = self.train_instructions[index]
        
            instruction = []
            instruction.append(vocab('<start>'))
            instruction.extend([self.vocab(token) for token in temp_instruction])
            instruction.append(vocab('<end>'))
            target = torch.Tensor(instruction)

            # instruction.append(vocab('<start>'))
            # instruction = [self.vocab('<unk>')]
            # instruction.append(vocab('<end>'))
            # target = torch.Tensor(instruction)

        #print(states_onehot.size(), states_embedding.size(), action.size(), goal.size())

        return states_onehot, states_embedding, inventory, action, goal, target

    def __len__(self):
        return len(self.train_states)
        #return self.train_states.shape[0]

def collate_fn(data):

    data.sort(key=lambda x: len(x[5]), reverse=True)
    states_onehot, states_embedding, inventory_embedding, action, goal, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    #images = torch.stack(images, 0)

    states_onehot = torch.stack(states_onehot,0)
    states_embedding = torch.stack(states_embedding,0)
    action = torch.stack(action,0)
    goal = torch.stack(goal,0)
    inventory_embedding = torch.stack(inventory_embedding,0)


    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    #targets = torch.zeros(len(captions), max(lengths)).long()
    targets = torch.ones(len(captions), max(lengths)).long()*216

    for i, cap in enumerate(captions):

        end = lengths[i]
        targets[i, :end] = cap[:end]        

    return states_onehot, states_embedding, inventory_embedding, action, goal, targets, lengths

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda:0')
else:
    print("using cpu")
    device = torch.device('cpu')

import torch
from torch import nn

#name = "example"
name = "compiled_dataset_08131950" #add 50 back in
constructed = True
skip = False
embed_dim = 300 # switch this later!!
building_vocab = False
augmented = False
without_stop = True


if building_vocab:

    # for experimenting only.

    with open('data/'+name+'50_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    build_vocabulary(all_instructions, name, embed_dim)

elif augmented:
    print("using augmented")

    # read all of the parts in... 

    total = 1622648
    num = 50000

    train_states_embedding = np.zeros((total, 5, 5, 300), dtype=np.float32)
    train_states_onehot = np.zeros((total, 5, 5, 7), dtype=np.float32)
    train_actions = np.zeros((total,1), dtype=np.float32)
    train_goals = np.zeros((total,300), dtype=np.float32)
    train_inventories = np.zeros((total ,300), dtype=np.float32)
    train_instructions = None

    i = 0
    for counter in range(total):
        if counter > (i+1) * num:
            print(i)
            with open('data/augmented_train_states_'+str(i+1), 'rb') as f:
                train_states_embedding[i*50000:(i+1)*50000] = pickle.load(f)
            with open('data/augmented_train_onehot_'+str(i+1), 'rb') as f:
                train_states_onehot[i*50000:(i+1)*50000] = pickle.load(f)
            with open('data/augmented_train_inventories_'+str(i+1), 'rb') as f:
                train_inventories[i*50000:(i+1)*50000] = pickle.load(f)
            with open('data/augmented_train_goals_'+str(i+1), 'rb') as f:
                train_goals[i*50000:(i+1)*50000] = pickle.load(f)
            with open('data/augmented_train_actions_'+str(i+1), 'rb') as f:
                train_actions[i*50000:(i+1)*50000] = pickle.load(f)
            with open('data/augmented_train_instructions_'+str(i+1), 'rb') as f:
                if train_instructions is None:
                    train_instructions = pickle.load(f)
                else:
                    temp = pickle.load(f)
                    train_instructions = np.concatenate((train_instructions, temp), axis=0)
            i = i + 1

    # add the 33rd one in..
    with open('data/augmented_train_states_'+str(i+1), 'rb') as f:
        train_states_embedding[i*50000:] = pickle.load(f)
    with open('data/augmented_train_onehot_'+str(i+1), 'rb') as f:
        train_states_onehot[i*50000:] = pickle.load(f)
    with open('data/augmented_train_inventories_'+str(i+1), 'rb') as f:
        train_inventories[i*50000:] = pickle.load(f)
    with open('data/augmented_train_goals_'+str(i+1), 'rb') as f:
        train_goals[i*50000:] = pickle.load(f)
    with open('data/augmented_train_actions_'+str(i+1), 'rb') as f:
        train_actions[i*50000:] = pickle.load(f)
    with open('data/augmented_train_instructions_'+str(i+1), 'rb') as f:
        train_instructions = np.concatenate((train_instructions, temp), axis=0)


    with open('data/'+name+'_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    #comment back when we define 
    vocab_weights = torch.Tensor(vocab_weights).to(device)

#augment...
elif skip and constructed:

    with open('data/'+name+'_1_train_states', 'rb') as f:
        train_states = pickle.load(f)

    with open('data/'+name+'_1_train_inventories', 'rb') as f:
        train_inventories = pickle.load(f)

    with open('data/'+name+'_1_train_actions', 'rb') as f:
        train_actions = pickle.load(f)

    with open('data/'+name+'_1_train_goals', 'rb') as f:
        train_goals = pickle.load(f)

    with open('data/'+name+'_train_instructions', 'rb') as f:
        train_instructions = pickle.load(f)


    augmented_train_states = np.repeat(train_states, repeats=8, axis=0)
    augmented_train_inventories = np.repeat(train_inventories, repeats=8, axis=0)
    augmented_train_actions = np.repeat(train_actions, repeats=8, axis=0)
    augmented_train_goals = np.repeat(train_goals, repeats=8, axis=0)
    augmented_train_instructions = np.repeat(train_instructions, repeats=8, axis=0)

    for i in range(0, augmented_train_states.shape[0], 8):
        augmented_train_states[i+1] = rotate_board(augmented_train_states[i+1], 90, False)
        augmented_train_states[i+2] = rotate_board(augmented_train_states[i+2], 180, False)
        augmented_train_states[i+3] = rotate_board(augmented_train_states[i+3], 270, False)
        augmented_train_states[i+4] = rotate_board(augmented_train_states[i+4], 0, True)
        augmented_train_states[i+5] = rotate_board(augmented_train_states[i+5], 90, True)
        augmented_train_states[i+6] = rotate_board(augmented_train_states[i+6], 180, True)
        augmented_train_states[i+7] = rotate_board(augmented_train_states[i+7], 270, True)

        augmented_train_actions[i+1] = rotate_action(augmented_train_actions[i+1], 90, False)
        augmented_train_actions[i+2] = rotate_action(augmented_train_actions[i+2], 180, False)
        augmented_train_actions[i+3] = rotate_action(augmented_train_actions[i+3], 270, False)
        augmented_train_actions[i+4] = rotate_action(augmented_train_actions[i+4], 0, True)
        augmented_train_actions[i+5] = rotate_action(augmented_train_actions[i+5], 90, True)
        augmented_train_actions[i+6] = rotate_action(augmented_train_actions[i+6], 180, True)
        augmented_train_actions[i+7] = rotate_action(augmented_train_actions[i+7], 270, True)

elif without_stop:

    with open('data/'+name+'_1_train_states', 'rb') as f:
        temp_train_states = pickle.load(f)

    with open('data/'+name+'_1_train_inventories', 'rb') as f:
        temp_train_inventories = pickle.load(f)

    with open('data/'+name+'_1_train_actions', 'rb') as f:
        temp_train_actions = pickle.load(f)

    with open('data/'+name+'_1_train_goals', 'rb') as f:
        temp_train_goals = pickle.load(f)

    with open('data/'+name+'_train_instructions', 'rb') as f:
        temp_train_instructions = pickle.load(f)

    #weights, vocab = load_vocabulary(name)


    # with open('data/'+name+'_train_states', 'rb') as f:
    #     temp_train_states = pickle.load(f)

    # with open('data/'+name+'_train_inventories', 'rb') as f:
    #     temp_train_inventories = pickle.load(f)

    # with open('data/'+name+'_train_actions', 'rb') as f:
    #     temp_train_actions = pickle.load(f)

    # with open('data/'+name+'_train_goals', 'rb') as f:
    #     temp_train_goals = pickle.load(f)

    # with open('data/'+name+'_train_instructions', 'rb') as f:
    #     temp_train_instructions = pickle.load(f)

    train_actions = []
    train_states = []
    train_instructions = []
    train_goals = []
    train_inventories  = []

    # 3 and 5 step crafts
    keep_list = ['Diamond Pickaxe', 'Stone Pickaxe', 'Wooden Door', 'Wood stairs', 'Leather Boots', 'Leather Helmet', 'Leather Chestplate', 'Leather Leggins', 'Iron Ingot']
    keep_list = ['Diamond Pickaxe', 'Stone Pickaxe']
    keep_list = ['Cobblestone Stairs', 'Leather Boots', 'Iron Ore']

    fiveactions = []
    onefouractions1 = []
    onefouractions2 = []
    onefouractions3 = []
    for i in range(len(temp_train_goals)):
        temp_goal_split = temp_train_goals[i].split(' ')
        temp_goal = temp_goal_split[1]+ " " + temp_goal_split[2]

        if temp_goal == 'Cobblestone Stairs':
            onefouractions1.append(i)
        elif temp_goal == 'Leather Boots':
            onefouractions2.append(i)
        elif temp_goal == 'Iron Ore':
            onefouractions3.append(i)
        else:
            fiveactions.append(i)

    #total = len(onefouractions)
    #print(total)
    #limit = int(0.01*total)
    #print(limit)

    #keep_list = random.sample(onefouractions, limit) + fiveactions
    keep_list = fiveactions + random.sample(onefouractions1, int(0.01*len(onefouractions1))) + random.sample(onefouractions2, int(0.01*len(onefouractions2))) + random.sample(onefouractions3, int(0.01*len(onefouractions3)))

    #remove "stops"
    for i in range(len(temp_train_actions)):
        temp_goal_split = temp_train_goals[i].split(' ')
        temp_goal = temp_goal_split[1]+ " " + temp_goal_split[2]

        #if temp_train_actions[i] != 'stop' and temp_goal in keep_list:  
        #if temp_train_actions[i] != 'stop' and i in keep_list:  
        #if temp_train_actions[i] != 'stop':
        if temp_train_actions[i] != 'stop' and i in keep_list:
            train_actions.append(temp_train_actions[i])
            train_states.append(temp_train_states[i])
            train_instructions.append(temp_train_instructions[i])
            train_goals.append(temp_train_goals[i])
            train_inventories.append(temp_train_inventories[i])

    print(len(train_actions))

    #weights, vocab = load_vocabulary(name)

    with open('data/'+name+'_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    temp = np.zeros((1,300), dtype=np.float32)
    temp1 = np.random.uniform(-0.01, 0.01, (1,300)).astype("float32")
    # vocab_weights.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    # vocab_weights.append(np.zeros(300).astype("float32"))

    vocab_weights = np.concatenate((vocab_weights, temp), axis=0)

    vocab_weights = torch.Tensor(vocab_weights).to(device)

elif not skip and constructed:

    with open('data/'+name+'_1_train_states', 'rb') as f:
        train_states = pickle.load(f)

    with open('data/'+name+'_1_train_inventories', 'rb') as f:
        train_inventories = pickle.load(f)

    with open('data/'+name+'_1_train_actions', 'rb') as f:
        train_actions = pickle.load(f)

    with open('data/'+name+'_1_train_goals', 'rb') as f:
        train_goals = pickle.load(f)

    with open('data/'+name+'_train_instructions', 'rb') as f:
        train_instructions = pickle.load(f)

    #weights, vocab = load_vocabulary(name)

    with open('data/'+name+'_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    #comment back when we define 
    vocab_weights = torch.Tensor(vocab_weights).to(device)


lstm_embed_dim = 32 #16

#model = LanguageNetv1(len(vocab), lstm_embed_dim)
#model = LanguageNetv2(len(vocab), embed_dim, vocab_weights)
#model = LanguageWithAttention(len(vocab), embed_dim, vocab_weights)
#model = LanguageWithAttentionGLOVE(len(vocab), embed_dim, vocab_weights)
model = LanguageWithAttentionSUM(len(vocab), embed_dim, vocab_weights)
model.to(device)
#model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttention.pt"))
parameters1 = filter(lambda p: p.requires_grad, model.parameters()) # MAYBE CHANGE THIS FOR OTHER ONE!
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#optimizer = torch.optim.Adam(parameters1, lr=0.001) # this is old one
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.RMSprop(parameters1, lr=1e-2)
optimizer = torch.optim.Adam(parameters1, lr=0.001)
new_criterion = nn.MSELoss()

train_loss = []
train_loss1 = []
val_loss = []
val_loss1 = []

#model1 = CNNAction(embed_dim, vocab, vocab_weights)
#model1 = AllObsPredictAtten(embed_dim, vocab_weights, vocab_words=vocab)
model1 = SimpleNetwork(embed_dim)

#if torch.cuda.device_count() > 1:
#  model1 = nn.DataParallel(model1)

#model1 = ActionNetv1(len(vocab), lstm_embed_dim)
model1.to(device)
#optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.5)
criterion1 = nn.CrossEntropyLoss().to(device)
parameters = filter(lambda p: p.requires_grad, model1.parameters()) # MAYBE CHANGE THIS FOR OTHER ONE!
optimizer1 = torch.optim.Adam(parameters, lr=0.001)

def train_step_bothGLOVE(epoch):

    log_size = 500

    model.train()
    model1.train()

    all_losses = []
    all_losses1 = []

    running_loss = 0.0
    running_loss1 = 0.0

    for i, data in enumerate(train_loader, 0):

        states_onehot, states_embedding, inventory, action, goal, instructions, lengths = data

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        action = action.to(device, dtype=torch.int64)
        action = action.squeeze(1)
        goal = goal.to(device)
        inventory = inventory.to(device)

        instructions = instructions.to(device)

        optimizer.zero_grad()
        optimizer1.zero_grad()

        scores, encoded_captions, decode_lengths, alphas, hiddens, embeddings = model(states_embedding, states_onehot, inventory, goal, instructions, lengths, device)
        targets = instructions[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        lang_loss = criterion(scores, targets)
        lang_loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()
        nn.utils.clip_grad_norm_(parameters1, max_norm=3)

        optimizer1.zero_grad()

        #train action component
        #scores, encoded_captions, decode_lengths, alphas, hiddens = model(states_embedding, states_onehot, inventory, goal, instructions, lengths, device)         
        outputs = model1(states_embedding, states_onehot, inventory, goal, hiddens)
        
        #backprop action loss
        action_loss = criterion1(outputs, action)

        #action_loss.backward()

        total_loss = lang_loss + action_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(parameters, max_norm=3)
        
        optimizer.step()
        optimizer1.step()

        running_loss += lang_loss.item()
        running_loss1 += action_loss.item()

        if i % log_size == log_size-1:   

            all_losses.append(running_loss / log_size)
            all_losses1.append(running_loss1 / log_size)

            print('[%d, %5d] lang loss: %.3f action loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_size, running_loss1 / log_size))

            running_loss = 0.0
            running_loss1 = 0.0

    train_loss.append(np.mean(all_losses))
    train_loss1.append(np.mean(all_losses1))

def train_step_langmodelGLOVE(epoch):

    log_size = 500

    model.train()

    all_losses = []
    all_losses1 = []

    running_loss = 0.0
    running_loss1 = 0.0

    for i, data in enumerate(train_loader, 0):

        states_onehot, states_embedding, inventory, action, goal, instructions, lengths = data

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        action = action.to(device, dtype=torch.int64)
        action = action.squeeze(1)
        goal = goal.to(device)
        inventory = inventory.to(device)

        instructions = instructions.to(device)
        #targets = pack_padded_sequence(instructions, lengths, batch_first=True)[0]

        optimizer.zero_grad()

        scores, encoded_captions, decode_lengths, alphas, hiddens, embeddings = model(states_embedding, states_onehot, inventory, goal, instructions, lengths, device)
        #outputs, hidden_layer = model(states_embedding, states_onehot, inventory, goal, instructions, lengths, device)

        #targets = instructions[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        #scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        #targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        embeddings = embeddings[:, :-1, :]
        lang_loss = new_criterion(scores, embeddings)
        nn.utils.clip_grad_norm_(parameters1, max_norm=3)
        #lang_loss = criterion(scores, targets)
        #lang_loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()

        #backprop language loss
        #lang_loss = criterion(outputs, targets)
        lang_loss.backward()
        optimizer.step()

        optimizer1.zero_grad()

        #hidden_layer = hidden_layer.detach()

        #print(hidden_layer.size())

        #train action component         
        #outputs = model1(states_embedding, states_onehot, inventory, goal, hidden_layer)
        
        #backprop action loss
        #action_loss = criterion1(outputs, action)
        #action_loss.backward()
        #optimizer1.step()

        running_loss += lang_loss.item()
        #running_loss1 += action_loss.item()

        if i % log_size == log_size-1:   

            all_losses.append(running_loss / log_size)
            all_losses1.append(running_loss1 / log_size)

            #writer.add_scalar('Loss/train', np.random.random(), n_iter)
            print('[%d, %5d] lang loss: %.3f action loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_size, running_loss1 / log_size))
            # with open("loss.txt", "a") as myfile:
            #     myfile.write('[%d, %5d] loss: %.3f \n' %
            #       (epoch + 1, i + 1, running_loss / log_size))

            running_loss = 0.0
            running_loss1 = 0.0

    train_loss.append(np.mean(all_losses))
    train_loss1.append(np.mean(all_losses1))

def train_step_both(epoch):

    log_size = 500

    model.train()
    model1.train()

    all_losses = []
    all_losses1 = []

    running_loss = 0.0
    running_loss1 = 0.0

    for i, data in enumerate(train_loader, 0):

        states_onehot, states_embedding, inventory, action, goal, instructions, lengths = data

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        action = action.to(device, dtype=torch.int64)
        action = action.squeeze(1)
        goal = goal.to(device)
        inventory = inventory.to(device)

        instructions = instructions.to(device)

        optimizer.zero_grad()
        optimizer1.zero_grad()

        scores, encoded_captions, decode_lengths, alphas, hiddens = model(states_embedding, states_onehot, inventory, goal, instructions, lengths, device)

        targets = instructions[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        lang_loss = criterion(scores, targets)
        lang_loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()
        #lang_loss.backward()
        nn.utils.clip_grad_norm_(parameters1, max_norm=3)
        #optimizer.step()

        #all_sampled_ids = model.get_hidden_state(states_embedding, states_onehot, inventory, goal, device, vocab)
        

        #bow_ids = [sent + [len(vocab)] * (20 - len(sent)) for sent in all_sampled_ids]

        #bow_ids = torch.Tensor(bow_ids)


        #bow_ids = bow_ids.long()
        #bow_ids = bow_ids.to(device)

        optimizer1.zero_grad()

        #train action component
        #scores, encoded_captions, decode_lengths, alphas, hiddens = model(states_embedding, states_onehot, inventory, goal, instructions, lengths, device)         
        outputs = model1(states_embedding, states_onehot, inventory, goal, hiddens)
        
        #backprop action loss
        action_loss = criterion1(outputs, action)

        #action_loss.backward()

        total_loss = lang_loss + action_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(parameters, max_norm=3)
        
        optimizer.step()
        optimizer1.step()

        running_loss += lang_loss.item()
        running_loss1 += action_loss.item()

        if i % log_size == log_size-1:   

            all_losses.append(running_loss / log_size)
            all_losses1.append(running_loss1 / log_size)

            print('[%d, %5d] lang loss: %.3f action loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_size, running_loss1 / log_size))

            running_loss = 0.0
            running_loss1 = 0.0

    train_loss.append(np.mean(all_losses))
    train_loss1.append(np.mean(all_losses1))

def train_step_actionmodel(epoch):

    log_size = 500

    model.eval()
    model1.train()

    all_losses = []
    all_losses1 = []

    running_loss = 0.0
    running_loss1 = 0.0

    for i, data in enumerate(train_loader, 0):
        #print(i)

        states_onehot, states_embedding, inventory, action, goal, instructions, lengths = data

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        action = action.to(device, dtype=torch.int64)
        action = action.squeeze(1)
        goal = goal.to(device)
        inventory = inventory.to(device)

        instructions = instructions.data.tolist()

        #instructions = instructions.to(device)
        #targets = pack_padded_sequence(instructions, lengths, batch_first=True)[0]

        optimizer.zero_grad()

        all_sampled_ids = model.get_hidden_state(states_embedding, states_onehot, inventory, goal, device, vocab)

        # maybe remove START and STOP!
        #bow_ids = [sent[1:-1] + [len(vocab) + 1] * (MAX_LEN - len(sent[1:-1])) for sent in all_sampled_ids]
        bow_ids = [sent + [len(vocab)] * (20 - len(sent)) for sent in all_sampled_ids]

        #bow_ids = Variable(torch.LongTensor(bow_ids)).to(device)

        bow_ids = torch.Tensor(bow_ids)
        bow_ids = bow_ids.long()
        bow_ids = bow_ids.to(device)

        optimizer1.zero_grad()

        #train action component         
        outputs = model1(states_embedding, states_onehot, inventory, goal, bow_ids)
        
        #backprop action loss
        action_loss = criterion1(outputs, action)
        #action_loss = F.nll_loss(outputs, action)
        action_loss.backward()
        nn.utils.clip_grad_norm_(parameters, max_norm=0.5) # maybe add this change too
        optimizer1.step()

        #running_loss += lang_loss.item()
        running_loss1 += action_loss.item()

        if i % log_size == log_size-1:   

            all_losses.append(running_loss / log_size)
            all_losses1.append(running_loss1 / log_size)

            #writer.add_scalar('Loss/train', np.random.random(), n_iter)
            print('[%d, %5d] lang loss: %.3f action loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_size, running_loss1 / log_size))
            # with open("loss.txt", "a") as myfile:
            #     myfile.write('[%d, %5d] loss: %.3f \n' %
            #       (epoch + 1, i + 1, running_loss / log_size))

            running_loss = 0.0
            running_loss1 = 0.0

    train_loss.append(np.mean(all_losses))
    train_loss1.append(np.mean(all_losses1))

def train_step_langmodel(epoch):

    log_size = 500

    #model.eval()
    model.train()
    #model1.train()

    all_losses = []
    all_losses1 = []

    running_loss = 0.0
    running_loss1 = 0.0

    for i, data in enumerate(train_loader, 0):

        #states_onehot, states_embedding, action, goal, instructions = data
        #states_onehot, states_embedding, action, goal, instructions, lengths = data

        states_onehot, states_embedding, inventory, action, goal, instructions, lengths = data

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        action = action.to(device, dtype=torch.int64)
        action = action.squeeze(1)
        goal = goal.to(device)
        inventory = inventory.to(device)

        instructions = instructions.to(device)
        #targets = pack_padded_sequence(instructions, lengths, batch_first=True)[0]

        optimizer.zero_grad()

        scores, encoded_captions, decode_lengths, alphas, hidden = model(states_embedding, states_onehot, inventory, goal, instructions, lengths, device)
        #outputs, hidden_layer = model(states_embedding, states_onehot, inventory, goal, instructions, lengths, device)

        targets = instructions[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        lang_loss = criterion(scores, targets)
        lang_loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()

        #backprop language loss
        #lang_loss = criterion(outputs, targets)
        lang_loss.backward()
        optimizer.step()

        optimizer1.zero_grad()

        #hidden_layer = hidden_layer.detach()

        #print(hidden_layer.size())

        #train action component         
        #outputs = model1(states_embedding, states_onehot, inventory, goal, hidden_layer)
        
        #backprop action loss
        #action_loss = criterion1(outputs, action)
        #action_loss.backward()
        #optimizer1.step()

        running_loss += lang_loss.item()
        #running_loss1 += action_loss.item()

        if i % log_size == log_size-1:   

            all_losses.append(running_loss / log_size)
            all_losses1.append(running_loss1 / log_size)

            #writer.add_scalar('Loss/train', np.random.random(), n_iter)
            print('[%d, %5d] lang loss: %.3f action loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_size, running_loss1 / log_size))
            # with open("loss.txt", "a") as myfile:
            #     myfile.write('[%d, %5d] loss: %.3f \n' %
            #       (epoch + 1, i + 1, running_loss / log_size))

            running_loss = 0.0
            running_loss1 = 0.0

    train_loss.append(np.mean(all_losses))
    train_loss1.append(np.mean(all_losses1))

def validate_step(epoch):

    model.eval()
    model1.eval()

    log_size = 100

    all_losses = []
    all_losses1 = []

    running_loss = 0.0
    running_loss1 = 0.0

    for i, data in enumerate(val_loader, 0):

        #states_onehot, states_embedding, action, goal = data

        states_onehot, states_embedding, inventory, action, goal, instructions, lengths = data

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        action = action.to(device, dtype=torch.int64)
        action = action.squeeze(1)
        goal = goal.to(device)
        inventory = inventory.to(device)

        instructions = instructions.to(device)
        targets = pack_padded_sequence(instructions, lengths, batch_first=True)[0]

        outputs, hidden_layer = model(states_embedding, states_onehot, inventory, goal, instructions, lengths)

        lang_loss = criterion(outputs, targets)

        hidden_layer = hidden_layer.detach()

        print(hidden_layer.size())

        #train action component         
        outputs = model1(states_embedding, states_onehot, inventory, goal, hidden_layer)
        
        action_loss = criterion1(outputs, action)

        running_loss += lang_loss.item()
        running_loss1 += action_loss.item()

        if i % log_size == log_size-1:   

            all_losses.append(running_loss / log_size)
            all_losses1.append(running_loss1 / log_size)

            #writer.add_scalar('Loss/train', np.random.random(), n_iter)
            print('VAL [%d, %5d] lang loss: %.3f action loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_size, running_loss1 / log_size))
            # with open("loss.txt", "a") as myfile:
            #     myfile.write('[%d, %5d] loss: %.3f \n' %
            #       (epoch + 1, i + 1, running_loss / log_size))

            running_loss = 0.0
            running_loss1 = 0.0

    val_loss.append(np.mean(all_losses))
    val_loss1.append(np.mean(all_losses1))

def validate_language():

    model.eval()

    if embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)
    else:
        glove = vocabtorch.GloVe(name='840B', dim=300)

    for i in range(5):
        generate_lang(model, glove, embed_dim, vocab, device) # prints out 5 sampled languages..
        #print(lang)


def validate_game_play():

    model.eval()
    model1.eval()

    if embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)
    else:
        glove = vocabtorch.GloVe(name='840B', dim=300)

    results = []
    for i in range(100):
        #res, sentences = ply_game_w_language_glove(model, model1, glove, embed_dim, vocab, device) 
        res, sentences = play_game_w_language_v3(model, model1, glove, embed_dim, vocab, device) 
        #res, sentences = play_game_w_language_v2(model, model1, glove, embed_dim, vocab, device) 
        results.append(res)
        #if res:
        #    print('generated sentences', sentences)

    print(sum(results), 100)
    return sum(results)


#train.

#subset out a portion.. 

#dset = PremadeCraftingDataset(embed_dim, train_states_embedding, train_states_onehot, train_inventories, train_actions, train_goals, train_instructions, vocab)
dset = CraftingDataset(embed_dim, train_states, train_inventories, train_actions, train_goals, train_instructions, vocab)
train_loader = DataLoader(dset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=0, 
                          pin_memory=True,
                          collate_fn=collate_fn # only if reading instructions too
                         )
'''
dset1 = CraftingDataset(embed_dim, test_states, test_inventories, test_actions, test_goals, test_instructions, vocab)
val_loader = DataLoader(dset1,
                          batch_size=128,
                          shuffle=True,
                          num_workers=0, 
                          pin_memory=True,
                          collate_fn=collate_fn
                         )
'''
os.system("rm loss.txt")

epochs = 15

rewards = []
for epoch in range(epochs):
    #train_step_bothGLOVE(epoch)
    train_step_both(epoch)
    #train_step_langmodel(epoch)
    #train_step_langmodelGLOVE(epoch)
    #train_step_actionmodel(epoch)
    #validate_language()
    #validate_step(epoch)
    tot_rewards = validate_game_play()
    rewards.append(tot_rewards)

print(rewards)
    
    
#t = [i+1 for i in range(epochs)]
#plt.plot(t, train_loss, 'r')
#plt.plot(t, val_loss, 'b')
#plt.plot(t, train_loss1, 'g')
#plt.plot(t, val_loss1, 'c')
#plt.savefig('training_results_hierarchy.png')

#torch.save(model.state_dict(), "TRAINED_MODELS/LanguageWithAttentionSUM_onlyAdam.pt")
#torch.save(model1.state_dict(), "TRAINED_MODELS/AllObsPredictAttenGLOVE.pt")


# torch.save(model.state_dict(), "/scratch/Anon Author/Mazebase_dialog/TRAINED_MODELS/LanguageWithAttentionSUM_35.pt")
# torch.save(model1.state_dict(), "/scratch/Anon Author/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_35.pt")

#torch.save(model.state_dict(), "TRAINED_MODELS/LanguageWithAttentionSUM_missing01per.pt")
#torch.save(model1.state_dict(), "TRAINED_MODELS/SimpleNetwork_missing01per.pt")





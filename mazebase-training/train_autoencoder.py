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

#from data_reader import read_dataset
from torchsummary import summary

import random
import os
import yaml
import json

import mazebasev2.lib.mazebase.games as games
from mazebasev2.lib.mazebase.games import featurizers

from mazebasev2.lib.mazebase.items.terrain import CraftingItem, CraftingContainer, ResourceFont, Block, Water, Switch, Door
from mazebasev2.lib.mazebase.items import agents

from train_models import StateGoalNet, StateGoalInstructionNet, LanguageNet, ActionNet, LanguageNetv1, ActionNetv1, ActionNet_Allobs, LanguageNetv2, LanguageWithAttention, AllObsPredictAtten, CNNAction, LanguageWithAttentionGLOVE, SimpleNetwork, LanguageWithAttentionSUM, StateAutoencoder

import nltk
import pickle

from build_vocab import build_vocabulary, load_vocabulary
from test_models import generate_lang, play_game_w_language_v2, play_game_w_language_v3, play_game_w_language_auto
from data_augmentation import rotate_board, rotate_action



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

            index = index + 10 # in case there is an error

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

        return states_onehot, states_embedding, inventory, action, goal, target

    def __len__(self):
        return len(self.train_states)

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

embed_dim = 300 
name = ADD_DATASET_HERE
embed_dim = 300


with open(name+'_states', 'rb') as f:
    train_states = pickle.load(f)

with open(name+'inventories', 'rb') as f:
    train_inventories = pickle.load(f)

with open(name+'actions', 'rb') as f:
    train_actions = pickle.load(f)

with open(name+'goals', 'rb') as f:
    train_goals = pickle.load(f)

with open(name+'instructions', 'rb') as f:
    train_instructions = pickle.load(f)

with open(name+'all_instructions', 'rb') as f:
    all_instructions = pickle.load(f)

vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

vocab.add_word('<pad>')
vocab.add_word('<start>')
vocab.add_word('<end>')
vocab.add_word('<unk>')

temp = np.zeros((1,300), dtype=np.float32)

vocab_weights = np.concatenate((vocab_weights, temp), axis=0)


lstm_embed_dim = 32

train_loss = []
train_loss1 = []
val_loss = []
val_loss1 = []


model1 = SimpleNetwork(embed_dim)
model1.to(device)
criterion1 = nn.CrossEntropyLoss().to(device)
parameters = filter(lambda p: p.requires_grad, model1.parameters())
optimizer1 = torch.optim.Adam(parameters, lr=0.001)

model2 = StateAutoencoder()
model2.to(device)
criterion2 = nn.MSELoss()
parameters2 = filter(lambda p: p.requires_grad, model2.parameters())
optimizer2 = torch.optim.Adam(parameters2, lr=0.001, weight_decay=1e-5)
model2.train()

def train_step_bothautoencoder(epoch):

    model1.train()
    model2.train()

    log_size = 500

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

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        state_encoding, hidden, reconstruction = model2(states_embedding, states_onehot, inventory, goal)
        
        recon_loss = criterion2(reconstruction, state_encoding)

        outputs = model1(states_embedding, states_onehot, inventory, goal, hidden)

        action_loss = criterion1(outputs, action)

        total_loss = recon_loss + action_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(parameters, max_norm=3)
        nn.utils.clip_grad_norm_(parameters2, max_norm=3)

        optimizer1.step()
        optimizer2.step()

        running_loss += recon_loss.item()
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

def validate_game_play():

    model2.eval()
    model1.eval()

    if embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)
    else:
        glove = vocabtorch.GloVe(name='840B', dim=300)

    results = []
    for i in range(100):
        res = play_game_w_language_auto(model2, model1, glove, embed_dim, vocab, device)
        results.append(res)

    print(sum(results), 100)
    return sum(results)


dset = CraftingDataset(embed_dim, train_states, train_inventories, train_actions, train_goals, train_instructions, vocab)
train_loader = DataLoader(dset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=0, 
                          pin_memory=True,
                          collate_fn=collate_fn # only if reading instructions too
                         )

os.system("rm loss.txt")

epochs = 25

rewards = []
for epoch in range(epochs):
    train_step_bothautoencoder(epoch)
    tot_rewards = validate_game_play()
    rewards.append(tot_rewards)

print(rewards)

torch.save(model2.state_dict(), "TRAINED_MODELS/StateAutoencoder_both1_05.pt")
torch.save(model1.state_dict(), "TRAINED_MODELS/SimpleNetwork_autoencoder1_05.pt")




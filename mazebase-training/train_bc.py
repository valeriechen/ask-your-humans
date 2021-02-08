import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchtext.vocab as vocabtorch
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

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

from train_models import StateGoalNet, StateGoalInstructionNet, StateGoalInstructionv1Net, StateGoalNetv1, StateGoalNetv2

import nltk
import pickle

from build_vocab import build_vocabulary, load_vocabulary
from test_models import play_game

import sys

## Behavior cloning: state + goal --> action 

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
        phrase_vector = torch.from_numpy(np.zeros((self.embed_dim), dtype=np.float32))

        for p in phrase:
            try:
                phrase_vector += self.glove.vectors[self.glove.stoi[p.lower()]]
            
            # MAKE THIS ALL zeros?
            except:
                phrase_vector += self.glove.vectors[self.glove.stoi['unknown']]  #replace this later??

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

            #item1_vec = self.glove.vectors[self.glove.stoi[goal[1].lower()]]
            #item2_vec = self.glove.vectors[self.glove.stoi[goal[2].lower()]]

            #goal_embedding = item1_vec+item2_vec

            goal_embedding = self.get_summed_embedding(goal[1]+' '+goal[2])

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
        else:
            print(action)
            print("HEREEE")

    def one_hot_grid(self, grid):

        grid_embedding_array = np.zeros((5, 5, 7), dtype=np.float32)

        ## ADD information about switch and door opening!!

        for x in range(5):
            for y in range(5):

                for index, item in enumerate(grid[x][y]):

                    if item == 'Corner':
                        grid_embedding_array[x][y][0] = 1
                    elif item == 'Agent':
                        grid_embedding_array[x][y][1] = 1
                    elif item == 'Door' or item == 'Door_opened' or item == 'Door_closed':
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
            #print(index)
            instruction = [self.vocab('<unk>')]
            target = torch.Tensor(instruction)

        #print(states_onehot.size(), states_embedding.size(), action.size(), goal.size())

        return states_onehot, states_embedding, inventory, action, goal, target

    def __len__(self):
        return len(self.train_states)
        #return self.train_states.shape[0]

def collate_fn(data):

    data.sort(key=lambda x: len(x[5]), reverse=True)
    states_onehot, states_embedding, inventory_embedding, action, goal, captions = zip(*data)

    states_onehot = torch.stack(states_onehot,0)
    states_embedding = torch.stack(states_embedding,0)
    action = torch.stack(action,0)
    goal = torch.stack(goal,0)
    inventory_embedding = torch.stack(inventory_embedding,0)


    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):

        end = lengths[i]
        targets[i, :end] = cap[:end]        

    return states_onehot, states_embedding, inventory_embedding, action, goal, targets, lengths

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda')
else:
    print("using cpu")
    device = torch.device('cpu')

import torch
from torch import nn

#name = "example"
name = "compiled_dataset_08131950"
constructed = True
skip = False
embed_dim = 300


## for training

if not skip and not constructed:
    train_states, train_inventories, train_actions, train_goals, train_instructions, test_states, test_inventories, test_actions, test_goals, test_instructions, all_instructions = read_dataset('data/'+name+'.json', 0.8)

    with open('data/'+name+'_train_states', 'wb') as f:
        pickle.dump(train_states, f)

    with open('data/'+name+'_train_inventories', 'wb') as f:
        pickle.dump(train_inventories, f)

    with open('data/'+name+'_train_actions', 'wb') as f:
        pickle.dump(train_actions, f)

    with open('data/'+name+'_train_goals', 'wb') as f:
        pickle.dump(train_goals, f)

    with open('data/'+name+'_train_instructions', 'wb') as f:
        pickle.dump(train_instructions, f)

    with open('data/'+name+'_test_states', 'wb') as f:
        pickle.dump(test_states, f)

    with open('data/'+name+'_test_inventories', 'wb') as f:
        pickle.dump(test_inventories, f)

    with open('data/'+name+'_test_actions', 'wb') as f:
        pickle.dump(test_actions, f)

    with open('data/'+name+'_test_goals', 'wb') as f:
        pickle.dump(test_goals, f)

    with open('data/'+name+'_test_instructions', 'wb') as f:
        pickle.dump(test_instructions, f)

    build_vocabulary(all_instructions, name)
elif not skip and constructed:

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
    keep_list = fiveactions + random.sample(onefouractions1, int(0.05*len(onefouractions1))) + random.sample(onefouractions2, int(0.01*len(onefouractions2))) + random.sample(onefouractions3, int(0.01*len(onefouractions3)))

    train_actions = []
    train_states = []
    train_instructions = []
    train_goals = []
    train_inventories  = []

    #remove "stops"
    for i in range(len(temp_train_actions)):
        if temp_train_actions[i] != 'stop' and i in keep_list:  # check if this is right..
            train_actions.append(temp_train_actions[i])
            train_states.append(temp_train_states[i])
            train_instructions.append(temp_train_instructions[i])
            train_goals.append(temp_train_goals[i])
            train_inventories.append(temp_train_inventories[i])

    #weights, vocab = load_vocabulary(name)

    with open('data/'+name+'_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    vocab, weights = build_vocabulary(all_instructions, name, embed_dim)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')


#model = AllObsPredict(embed_dim) # kenny's

#model = StateGoalNetv2(embed_dim) # add
model = StateGoalNetv1(embed_dim) # concat

#model = StateGoalInstructionv1Net(len(vocab), 256)

#if torch.cuda.device_count() > 1:
#  model = nn.DataParallel(model)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()

train_loss = []
val_loss = []


#used for normal bc
def train_step(epoch):

    log_size = 500

    model.train()

    all_losses = []

    running_loss = 0.0
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

        outputs = model(states_embedding, states_onehot, inventory, goal)

        #outputs = model(states_embedding, states_onehot, inventory, goal, instructions, lengths)

        loss = criterion(outputs, action)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % log_size == log_size-1:   

            all_losses.append(running_loss / log_size)

            #writer.add_scalar('Loss/train', np.random.random(), n_iter)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_size))
            with open("loss.txt", "a") as myfile:
                myfile.write('[%d, %5d] loss: %.3f \n' %
                  (epoch + 1, i + 1, running_loss / log_size))

            running_loss = 0.0

    train_loss.append(np.mean(all_losses))

def validate_step(epoch):

    model.eval()

    log_size = 100

    all_losses = []

    running_loss = 0.0
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

        outputs = model(states_embedding, states_onehot, inventory, goal)

        #outputs = model(states_embedding, states_onehot, inventory, goal, instructions, lengths)

        loss = criterion(outputs, action)

        running_loss += loss.item()

        if i % log_size == log_size-1:   
            all_losses.append(running_loss / log_size)

            #writer.add_scalar('Loss/train', np.random.random(), n_iter)
            print('VAL: [%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_size))
            with open("loss.txt", "a") as myfile:
                myfile.write('VAL: [%d, %5d] loss: %.3f \n' %
                  (epoch + 1, i + 1, running_loss / log_size))

            running_loss = 0.0

    val_loss.append(np.mean(all_losses))


def validate_game():

    model.eval()

    if embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)
    else:
        glove = vocabtorch.GloVe(name='840B', dim=300)


    results = []
    for i in range(15):
        res = play_game(model, glove, embed_dim, device) 
        results.append(res)

    print(sum(results))
    return sum(results)

    #print(sum(results), 'out of 15')


#train.
dset = CraftingDataset(embed_dim, train_states, train_inventories, train_actions, train_goals, train_instructions, vocab)
train_loader = DataLoader(dset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=0, 
                          pin_memory=True,
                          collate_fn=collate_fn # only if reading instructions too
                         )

os.system("rm loss.txt")

epochs = 30

rewards = []

for epoch in range(epochs):
    train_step(epoch)
    r = validate_game()
    rewards.append(r)

print(rewards)

    
t = [i+1 for i in range(epochs)]
plt.plot(t, train_loss, 'r')
#plt.plot(t, val_loss, 'b')
plt.savefig('training_results.png')

torch.save(model.state_dict(), 'TRAINED_MODELS/TRAINED_MODELS/StateGoalNetv1_300_05per.pt')








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

from train_models import StateGoalNet, StateGoalInstructionNet, LanguageNet, ActionNet, LanguageNetv1, ActionNetv1, ActionNet_Allobs, LanguageNetv2, LanguageWithAttention, StatePredictorNetwork, CNNAction, LanguageWithAttentionGLOVE, SimpleNetwork, LanguageWithAttentionSUM, StateAutoencoder

import nltk
import pickle

from build_vocab import build_vocabulary, load_vocabulary
from test_models import generate_lang, play_game_w_language_v2, play_game_w_language_v3, play_game_w_language_state
from data_augmentation import rotate_board, rotate_action


class CraftingDataset(Dataset):

    def __init__(self, embed_dim, train_states, train_inventories, train_actions, train_goals, train_past, train_instructions, vocab, transform=None):

        self.embed_dim = embed_dim

        self.vocab = vocab
        self.train_instructions = train_instructions

        self.train_states = train_states
        self.train_inventories = train_inventories
        self.train_actions = train_actions
        self.train_goals = train_goals
        self.train_past = train_past

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

        action = torch.Tensor(self.train_actions_onehot[index])

        states_embedding = []
        states_onehot = []
        goal = []
        inventory = []

        temp_indices = self.train_past[index]
        #temp_indices.append(index)

        for ind in temp_indices:
            states_embedding.append(self.train_states_embedding[index])
            states_onehot.append(self.train_states_onehot[index])
            goal.append(self.train_goals_embedding[index])
            inventory.append(self.train_inventory_embedding[index])

        states_embedding = torch.Tensor(states_embedding)
        states_onehot = torch.Tensor(states_onehot)
        goal = torch.Tensor(goal)
        inventory = torch.Tensor(inventory)

        return states_onehot, states_embedding, inventory, action, goal

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

    train_actions = []
    train_states = []
    train_instructions = []
    train_goals = []
    train_inventories  = []
    train_past = []

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

    percent = 0.1
    keep_list = fiveactions + random.sample(onefouractions1, int(percent*len(onefouractions1))) + random.sample(onefouractions2, int(percent*len(onefouractions2))) + random.sample(onefouractions3, int(percent*len(onefouractions3)))

    queue = []

    to_append = True

    #remove "stops"
    for i in range(len(temp_train_actions)):
        temp_goal_split = temp_train_goals[i].split(' ')
        temp_goal = temp_goal_split[1]+ " " + temp_goal_split[2]

        if temp_train_actions[i] != 'stop':

            if len(queue) == 0:

                if i in keep_list:
                    to_append = True

                else:
                    to_append = False

                curr_index = len(train_actions)-1
                queue = [curr_index, curr_index, curr_index, curr_index]

            else:

                queue = queue[1:]
                queue.append(len(train_actions)-2) #add on the previous.. 

            if to_append:

                train_actions.append(temp_train_actions[i])
                train_states.append(temp_train_states[i])
                train_instructions.append(temp_train_instructions[i])
                train_goals.append(temp_train_goals[i])
                train_inventories.append(temp_train_inventories[i])
                train_past.append(queue)

        else:

            queue = []


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
# model = LanguageWithAttentionSUM(len(vocab), embed_dim, vocab_weights)
# model.to(device)
# model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttentionSUM_adam.pt"))
# model.eval()
# for params in model.parameters():
#     params.requires_grad = False
#parameters1 = filter(lambda p: p.requires_grad, model.parameters()) # MAYBE CHANGE THIS FOR OTHER ONE!
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model1 = SimpleNetwork(embed_dim)
model1.to(device)
criterion1 = nn.CrossEntropyLoss().to(device)
parameters = filter(lambda p: p.requires_grad, model1.parameters())
optimizer1 = torch.optim.Adam(parameters, lr=0.001)

#optimizer = torch.optim.Adam(parameters1, lr=0.001) # this is old one
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.RMSprop(parameters1, lr=1e-2)
#optimizer = torch.optim.Adam(parameters1, lr=0.001)
#new_criterion = nn.MSELoss()

train_loss = []
train_loss1 = []
val_loss = []
val_loss1 = []


model2 = StatePredictorNetwork(len(vocab), embed_dim, vocab_weights)
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

        states_onehot, states_embedding, inventory, action, goal = data

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        action = action.to(device, dtype=torch.int64)
        action = action.squeeze(1)
        goal = goal.to(device)
        inventory = inventory.to(device)

        # action = action.to(device, dtype=torch.int64)
        # action = action.squeeze(1)
        # states_onehot = [item.to(device) for item in states_onehot]
        # states_embedding = [item.to(device) for item in states_embedding]
        # goal = [item.to(device) for item in goal]
        # inventory = [item.to(device) for item in inventory]

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        reconstruction, hidden = model2(states_embedding[:,0:3,:,:,:], states_onehot[:,0:3,:,:,:], inventory[:,0:3,:], goal[:,0:3,:], device)
        state_encoding = model2.get_state_encoding_new(states_embedding[:,3,:,:,:], states_onehot[:,3,:,:,:], inventory[:,3,:], goal[:,3,:])
        
        recon_loss = criterion2(reconstruction, state_encoding)

        outputs = model1(states_embedding[:,3,:,:,:], states_onehot[:,3,:,:,:], inventory[:,3,:], goal[:,3,:], hidden)

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

        states_onehot, states_embedding, inventory, action, goal = data

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

    model2.eval()
    model1.eval()

    if embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)
    else:
        glove = vocabtorch.GloVe(name='840B', dim=300)

    results = []
    for i in range(100):
        res = play_game_w_language_state(model2, model1, glove, embed_dim, vocab, device)
        #res = play_game_w_language_auto(model2, model1, glove, embed_dim, vocab, device)
        #res, sentences = ply_game_w_language_glove(model, model1, glove, embed_dim, vocab, device) 
        #res, sentences = play_game_w_language_v3(model, model1, glove, embed_dim, vocab, device) 
        #res, sentences = play_game_w_language_v2(model, model1, glove, embed_dim, vocab, device) 
        results.append(res)
        #if res:
        #    print('generated sentences', sentences)

    print(sum(results), 100)
    return sum(results)


#train.

#subset out a portion.. 

#dset = PremadeCraftingDataset(embed_dim, train_states_embedding, train_states_onehot, train_inventories, train_actions, train_goals, train_instructions, vocab)
dset = CraftingDataset(embed_dim, train_states, train_inventories, train_actions, train_goals, train_past, train_instructions, vocab)
train_loader = DataLoader(dset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=0, 
                          pin_memory=True,
                          #collate_fn=collate_fn # only if reading instructions too
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

epochs = 20

rewards = []
for epoch in range(epochs):
    train_step_bothautoencoder(epoch)
    #tot_rewards = validate_game_play()
    #rewards.append(tot_rewards)
        

#print(rewards)
    
    
#t = [i+1 for i in range(epochs)]
#plt.plot(t, train_loss, 'r')
#plt.plot(t, val_loss, 'b')
#plt.plot(t, train_loss1, 'g')
#plt.plot(t, val_loss1, 'c')
#plt.savefig('training_results_hierarchy.png')

torch.save(model2.state_dict(), "TRAINED_MODELS/StatePredictor_both1_01.pt")
torch.save(model1.state_dict(), "TRAINED_MODELS/SimpleNetwork_statepred1_01.pt")







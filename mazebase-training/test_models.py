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

from train_models import StateGoalNet, StateGoalInstructionNet, StateGoalInstructionv1Net, StateGoalNetv1, StateGoalNetv2, LanguageNetv1, ActionNetv1, LanguageNetv2, LanguageWithAttention, AllObsPredictAtten, CNNAction, LanguageWithAttentionGLOVE, SimpleNetwork, LanguageWithAttentionSUM

import nltk
import pickle

from build_vocab import build_vocabulary, load_vocabulary

# Evaluated trained models


import torch
from torch import nn

def get_summed_embedding(phrase, glove, embed_size):

    phrase = phrase.split(' ')
    phrase_vector = torch.from_numpy(np.zeros((embed_size), dtype=np.float32))

    for p in phrase:
        phrase_vector += glove.vectors[glove.stoi[p.lower()]]

    return phrase_vector

def get_inventory_embedding(inventory, glove, embed_size):

    
    inventory_embedding = np.zeros((embed_size), dtype=np.float32)

    first = True
    for item in inventory:

        if inventory[item] > 0:

            if first:
                inventory_embedding = get_summed_embedding(item, glove, embed_size)
                first = False
            else:
                inventory_embedding = inventory_embedding + get_summed_embedding(item, glove, embed_size)

    return inventory_embedding

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

    goal_embedding = np.zeros((1,embed_size), dtype=np.float32)

    goal = goal.split(' ')

    item1_vec = glove.vectors[glove.stoi[goal[1].lower()]]
    item2_vec = glove.vectors[glove.stoi[goal[2].lower()]]

    goal_embedding[0] = item1_vec+item2_vec

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

# generate a new mazebase game

def generate_new_game():

    #pick a random yaml file: 

    #num = random.randint(1,4)
    #num =  2

    # if num == 1:
    #   yaml_file = 'mazebasev2/options/knowledge_planner/length45common.yaml'
    # elif num == 2:
    #   yaml_file = 'mazebasev2/options/knowledge_planner/length12task.yaml'
    # else:
    
    #yaml_file = 'mazebasev2/options/knowledge_planner/length3task.yaml' 
    #yaml_file = 'mazebasev2/options/knowledge_planner/length1task.yaml'
    #yaml_file = 'mazebasev2/options/knowledge_planner/length2task.yaml'
    #yaml_file = 'mazebasev2/options/knowledge_planner/length45common.yaml'
    #yaml_file = 'mazebasev2/options/knowledge_planner/length3task.yaml' 
    #yaml_file = 'mazebasev2/options/knowledge_planner/length12task.yaml'
    #yaml_file =  'mazebasev2/options/knowledge_planner/minimum_viable_planning.yaml'
    #yaml_file =  'mazebasev2/options/knowledge_planner/minimum_viable_rl.yaml'
    yaml_file =  'mazebasev2/options/knowledge_planner/unseen_tasks.yaml'

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
    all_games = [games.BasicKnowledgeGame(world_knowledge=world_knowledge, proposed_knowledge=[], options=env_opt, load_items=None, map_size=map_size)]

    # Game wrapper
    game = games.MazeGame(
      all_games,
      featurizer=featurizers.GridFeaturizer()
    )

    return game

def get_action_name(action):

    if action == 1:
        return 'up'
    elif action == 2:
        return 'down'
    elif action == 3:
        return 'left'
    elif action == 4:
        return 'right'
    elif action == 5:
        return  'toggle_switch'
    elif action == 6:
        return 'grab'
    elif action == 7:
        return 'mine'
    elif action  == 0:
        return 'craft'
    elif action == 8:
        return 'stop'


#language only:
def generate_lang(model, glove, embed_size, vocab, device):

    #deterministic in that if you get stuck somewhere, then your prediction will stay the same.

    if embed_size == 300:
        glove = vocabtorch.GloVe(name='840B', dim=300)
    elif embed_size == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)

    count = 0
    game = generate_new_game()

    print(game.game.goal)

    past_moves = []

    while not game.is_over() or count == 250:

        count = count + 1
        state = game.observe()['observation'][0]
        goal = game.game.goal
        inventory = game.game.inventory

        states_embedding = torch.from_numpy(np.array([get_grid_embedding(state, glove, embed_size)]))
        states_onehot = torch.from_numpy(np.array([one_hot_grid(state, glove, embed_size)]))
        goal = torch.from_numpy(get_goal_embedding(goal, glove, embed_size))
        #inventory = torch.from_numpy(get_inventory_embedding(inventory))
        #inventory = torch.Tensor(get_inventory_embedding(inventory, glove, embed_size))
        inventory = torch.Tensor(np.array([get_inventory_embedding(inventory, glove, embed_size)]))
        #print(inventory)

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        goal = goal.to(device)
        inventory = inventory.to(device)


        #if count % 10:

        sampled_ids = model.sample(states_embedding, states_onehot, inventory, goal, device, vocab)
        #sampled_ids = sampled_ids[0].cpu().numpy() 

        #also convert to words:
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption) 
        print(sentence)

        break

        #action = get_action_name(random.randint(0,8))
        #game.act(action)

    #print("----------")

#no language - use this for just bc!
def play_game(model, glove, embed_size, device):

    count = 0
    game = generate_new_game()
    state = game.observe()['observation'][0]
    #print(state)

    #print(game.game.goal)

    past_moves = []

    last_inv_size = 0

    while not game.is_over() and count < 250:

        count = count + 1
        state = game.observe()['observation'][0]
        goal = game.game.goal
        inventory = game.game.inventory

        if len(inventory) != last_inv_size:
            print(inventory)
            last_inv_size = len(inventory)

        states_embedding = torch.from_numpy(np.array([get_grid_embedding(state, glove, embed_size)]))
        states_onehot = torch.from_numpy(np.array([one_hot_grid(state, glove, embed_size)]))
        goal = torch.from_numpy(get_goal_embedding(goal, glove, embed_size))
        #inventory = torch.from_numpy(get_inventory_embedding(inventory))
        inventory = torch.Tensor(get_inventory_embedding(inventory, glove, embed_size))
        #print(inventory)

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        goal = goal.to(device)
        inventory = inventory.to(device)

        inventory = inventory.view(1, embed_size)

        outputs = model(states_embedding, states_onehot, inventory, goal)

        values, indices = outputs[0].max(0)

        #action = get_action_name(random.randint(0,8))
        action = get_action_name(indices.item())

        game.act(action)

    #print("--------")
    
    return game.is_over()
    #print(count < 100)

def play_game_w_language(model, model1, glove, embed_size, vocab, device):

    count = 0
    game = generate_new_game()

    #print(game.game.goal)

    past_moves = []

    while not game.is_over() and count < 150:

        count = count + 1
        state = game.observe()['observation'][0]
        goal = game.game.goal
        inventory = game.game.inventory

        states_embedding = torch.from_numpy(np.array([get_grid_embedding(state, glove, embed_size)]))
        states_onehot = torch.from_numpy(np.array([one_hot_grid(state, glove, embed_size)]))
        goal = torch.from_numpy(get_goal_embedding(goal, glove, embed_size))
        inventory = torch.Tensor(get_inventory_embedding(inventory, glove, embed_size))

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        goal = goal.to(device)
        inventory = inventory.to(device)

        inventory = inventory.view(1, embed_size)

        sampled_ids, hidden_layer = model.sample(states_embedding, states_onehot, inventory, goal)

        outputs = model1(states_embedding, states_onehot, inventory, goal, hidden_layer)

        values, indices = outputs[0].max(0) 

        action = get_action_name(indices.item())

        game.act(action)
    

    return count < 150

def play_game_w_language_glove(model, model1, glove, embed_size, vocab, glove_dict, device):

    count = 0
    game = generate_new_game()

    past_moves = []
    last_action = 'first'
    hdn_layer = None
    sentences = []

    last_inv_size = 0 

    while not game.is_over() and count < 150:

        count = count + 1
        state = game.observe()['observation'][0]
        goal = game.game.goal
        inventory = game.game.inventory

        states_embedding = torch.from_numpy(np.array([get_grid_embedding(state, glove, embed_size)]))
        states_onehot = torch.from_numpy(np.array([one_hot_grid(state, glove, embed_size)]))
        goal = torch.from_numpy(get_goal_embedding(goal, glove, embed_size))
        inventory = torch.Tensor(get_inventory_embedding(inventory, glove, embed_size))

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        goal = goal.to(device)
        inventory = inventory.to(device)

        inventory = inventory.view(1, 10, embed_size)

        # this needs to be written ahhh...
        seqs, hiddens = model.get_hidden_state_new(states_embedding, states_onehot, inventory, goal, device, vocab, glove_dict)         
        outputs = model1(states_embedding, states_onehot, inventory, goal, hiddens)

        values, indices = outputs[0].max(0) 

        action = get_action_name(indices.item())
        last_action = action

        game.act(action)

    return game.is_over(), sentences


def play_game_w_language_v3(model, model1, glove, embed_size, vocab, device):

    count = 0
    game = generate_new_game()

    past_moves = []
    last_action = 'first'
    hdn_layer = None
    sentences = []

    last_inv_size = 0 

    while not game.is_over() and count < 250:

        count = count + 1
        state = game.observe()['observation'][0]
        goal = game.game.goal
        inventory = game.game.inventory

        # if len(inventory) > 0:
        #     print(inventory)

        #if len(inventory) != last_inv_size:
        #    print(inventory)
        #    last_inv_size = len(inventory)

        states_embedding = torch.from_numpy(np.array([get_grid_embedding(state, glove, embed_size)]))
        states_onehot = torch.from_numpy(np.array([one_hot_grid(state, glove, embed_size)]))
        goal = torch.from_numpy(get_goal_embedding(goal, glove, embed_size))
        inventory = torch.Tensor(get_inventory_embedding(inventory, glove, embed_size))

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        goal = goal.to(device)
        inventory = inventory.to(device)

        #inventory = inventory.view(1, 10, embed_size)
        inventory = inventory.view(1, embed_size)

        seqs, hiddens = model.get_hidden_state_new(states_embedding, states_onehot, inventory, goal, device, vocab)         
        outputs = model1(states_embedding, states_onehot, inventory, goal, hiddens)

        values, indices = outputs[0].max(0) 

        action = get_action_name(indices.item())
        last_action = action

        game.act(action)

    return game.is_over(), sentences

def play_game_w_language_v2(model, model1, glove, embed_size, vocab, device):

    count = 0
    game = generate_new_game()

    past_moves = []
    last_action = 'first'
    hdn_layer = None
    sentences = []

    last_inv_size = 0 

    while not game.is_over() and count < 250:

        count = count + 1
        state = game.observe()['observation'][0]
        goal = game.game.goal
        inventory = game.game.inventory

        # if len(inventory) > 0:
        #     print(inventory)

        #if len(inventory) != last_inv_size:
        #    print(inventory)
        #    last_inv_size = len(inventory)

        states_embedding = torch.from_numpy(np.array([get_grid_embedding(state, glove, embed_size)]))
        states_onehot = torch.from_numpy(np.array([one_hot_grid(state, glove, embed_size)]))
        goal = torch.from_numpy(get_goal_embedding(goal, glove, embed_size))
        inventory = torch.Tensor(get_inventory_embedding(inventory, glove, embed_size))

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        goal = goal.to(device)
        inventory = inventory.to(device)

        inventory = inventory.view(1, 10, embed_size)

        #if sample again or never sampled before.
        #if last_action == 'stop' or last_action == 'first': HOW TO INCORPORATE TEHE STOP?
        bow_ids = []

        # sample
        all_sampled_ids = model.get_hidden_state(states_embedding, states_onehot, inventory, goal, device, vocab)

        # turn it into BOW
        '''
        for sampled_ids in all_sampled_ids:

            bow_id = [0]*len(vocab)

            for s_id in sampled_ids:
                bow_id[s_id] += 1

            bow_ids.append(bow_id)
        '''

        bow_ids = [sent + [len(vocab)] * (20 - len(sent)) for sent in all_sampled_ids]


        #FOR BOW
        bow_ids = torch.Tensor(bow_ids)
        bow_ids = bow_ids.long()
        bow_ids = bow_ids.to(device)
    
        outputs = model1(states_embedding, states_onehot, inventory, goal, bow_ids)


        # for GRU/LSTM
        #lengths = [len(sampled_ids)]
        #targets = torch.Tensor(sampled_ids)
        #targets = targets.long()
        #targets = targets.to(device)
        #targets = targets.view(1, targets.size(0))
        #outputs = model1(states_embedding, states_onehot, inventory, goal, targets, lengths)

        values, indices = outputs[0].max(0) 

        action = get_action_name(indices.item())
        last_action = action

        game.act(action)

    return game.is_over(), sentences

def load_model_play_game():

    # load model

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device('cuda')
    else:
        print("using cpu")
        device = torch.device('cpu')

    name = "compiled_dataset_08131950" #add 50 back in
    embed_dim = 300 # switch this later!!
    embed_size = embed_dim

    if embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)
    else:
        glove = vocabtorch.GloVe(name='840B', dim=300)


    # or do the all obs. 
    #action_model = AllObsPredict(embed_dim)
    action_model = StateGoalNetv1(embed_dim)
    action_model.to(device)
    #action_model.load_state_dict(torch.load("TRAINED_MODELS/TRAINED_MODELS/StateGoalNetv1_300.pt"))
    action_model.load_state_dict(torch.load("TRAINED_MODELS/TRAINED_MODELS/StateGoalNetv1_300_10per.pt"))
    action_model.eval()

    # play x number of games:
    tot_games = 100
    tot_win = 0
    for i in range(tot_games):
        print(i)
        res = play_game(action_model, glove, embed_size, device)
        tot_win = tot_win + res
        print(tot_win, i+1)

    print(tot_win, tot_games)

def load_model_play_game_with_lang():

    # load model

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device('cuda')
    else:
        print("using cpu")
        device = torch.device('cpu')

    name = "compiled_dataset_08131950" #add 50 back in
    embed_dim = 300 # switch this later!!
    embed_size = embed_dim

    if embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)
    else:
        glove = vocabtorch.GloVe(name='840B', dim=300)

    with open('data/'+name+'_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    temp = np.zeros((1,300), dtype=np.float32)
    temp1 = np.random.uniform(-0.01, 0.01, (1,300)).astype("float32")

    vocab_weights = np.concatenate((vocab_weights, temp), axis=0)

    vocab_weights = torch.Tensor(vocab_weights).to(device)

    #language_model = LanguageWithAttention(len(vocab), embed_dim, vocab_weights, training=False)
    language_model = LanguageWithAttentionSUM(len(vocab), embed_dim, vocab_weights, training=False)
    language_model.to(device)
    #language_model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttention_both.pt"))
    language_model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttentionSUM_missing10per.pt"))
    language_model.eval()

    # or do the all obs. 
    #action_model = AllObsPredictAtten(embed_dim, vocab_weights, vocab_words=vocab)
    action_model = SimpleNetwork(embed_dim)
    action_model.to(device)
    #action_model.load_state_dict(torch.load("TRAINED_MODELS/AllObsPredictAtten_both.pt"))
    action_model.load_state_dict(torch.load("TRAINED_MODELS/SimpleNetwork_missing10per.pt"))
    action_model.eval()

    #action_model = CNNAction(embed_dim, vocab, vocab_weights)
    #action_model.to(device)
    #action_model.load_state_dict(torch.load("TRAINED_MODELS/CNNAction_8epochs_nllsoftmax.pt"))
    #action_model.eval()

    # play x number of games:
    tot_games = 100
    tot_win = 0
    for i in range(tot_games):
        #print(i)
        res, sentences = play_game_w_language_v3(language_model, action_model, glove, embed_size, vocab, device)
        #print(res)
        #print(sentences)
        tot_win = tot_win + res
        print(tot_win, i+1)

    print(tot_win, tot_games)

def load_model_play_game_with_lang_glove():

    # load model

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device('cuda')
    else:
        print("using cpu")
        device = torch.device('cpu')

    name = "compiled_dataset_08131950" #add 50 back in
    embed_dim = 300 # switch this later!!
    embed_size = embed_dim

    if embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)
    else:
        glove = vocabtorch.GloVe(name='840B', dim=300)

    with open('data/'+name+'_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    temp = np.zeros((1,300), dtype=np.float32)
    temp1 = np.random.uniform(-0.01, 0.01, (1,300)).astype("float32")

    vocab_weights = np.concatenate((vocab_weights, temp), axis=0)

    vocab_weights = torch.Tensor(vocab_weights).to(device)

    language_model = LanguageWithAttentionGLOVE(len(vocab), embed_dim, vocab_weights, training=False)
    language_model.to(device)
    language_model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttentionGLOVE_clipped.pt"))
    language_model.eval()

    # or do the all obs. 
    action_model = AllObsPredictAtten(embed_dim, vocab_weights, vocab_words=vocab)
    action_model.to(device)
    action_model.load_state_dict(torch.load("TRAINED_MODELS/AllObsPredictAtten_both.pt"))
    action_model.eval()

    #action_model = CNNAction(embed_dim, vocab, vocab_weights)
    #action_model.to(device)
    #action_model.load_state_dict(torch.load("TRAINED_MODELS/CNNAction_8epochs_nllsoftmax.pt"))
    #action_model.eval()

    # play x number of games:
    tot_games = 20
    tot_win = 0
    for i in range(tot_games):
        #print(i)
        res, sentences = play_game_w_language_glove(language_model, action_model, glove, embed_size, vocab, vocab_weights, device)
        #print(res)
        #print(sentences)
        tot_win = tot_win + res
        print(tot_win, i+1)

    print(tot_win, tot_games)

def play_game_by_hand():

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device('cuda')
    else:
        print("using cpu")
        device = torch.device('cpu')

    name = "compiled_dataset_08131950" #add 50 back in
    embed_dim = 300 # switch this later!!
    embed_size = embed_dim

    with open('data/'+name+'_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    lstm_embed_dim = 16

    #model = LanguageNetv1(len(vocab), lstm_embed_dim)
    #model = LanguageNetv2(len(vocab), embed_dim, vocab_weights, training=False)
    #model = LanguageWithAttention(len(vocab), embed_dim, vocab_weights, training=False)
    model = LanguageWithAttentionSUM(len(vocab), embed_dim, vocab_weights, training=False)
    model.to(device)
    model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttentionSUM_adam.pt"))    
    #model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttentionSUM_missing10per.pt"))    
    #model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttention_both1.pt"))
    #model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttention.pt")) # trained with embeddings

    count = 0
    game = generate_new_game()

    if embed_size == 300:
        glove = vocabtorch.GloVe(name='840B', dim=300)
    elif embed_size == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)

    count = 0
    game = generate_new_game()

    print(game.game.goal)

    past_moves = []

    while not game.is_over() or count == 250:

        count = count + 1
        state = game.observe()['observation'][0]
        
        #fix this printing so it is easier.. 
        for line in state:
            print(line)

        goal = game.game.goal
        inventory = game.game.inventory

        states_embedding = torch.from_numpy(np.array([get_grid_embedding(state, glove, embed_size)]))
        states_onehot = torch.from_numpy(np.array([one_hot_grid(state, glove, embed_size)]))
        goal = torch.from_numpy(get_goal_embedding(goal, glove, embed_size))
        #inventory = torch.from_numpy(get_inventory_embedding(inventory))
        inventory = torch.Tensor(get_inventory_embedding(inventory, glove, embed_size))
        #inventory = torch.Tensor(np.array([get_inventory_embedding(inventory, glove, embed_size)]))

        #print(inventory)

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        goal = goal.to(device)
        inventory = inventory.to(device)

        inventory = inventory.view(1, embed_size)

        #if count % 10:

        sampled_ids, hiddens = model.get_hidden_state_new(states_embedding, states_onehot, inventory, goal, device, vocab)         
        #sampled_ids = model.sample(states_embedding, states_onehot, inventory, goal, device, vocab)
        #sampled_ids = sampled_ids[0].cpu().numpy() 

        #also convert to words:
        # for i in range(5):
        #     sampled_caption = []
        #     for word_id in sampled_ids[i]:
        #         word = vocab.idx2word[word_id]
        #         sampled_caption.append(word)
        #         if word == '<end>':
        #             break
        #     sentence = ' '.join(sampled_caption) 
        #     print(sentence, scores[i])
        sampled_caption = []
        for word_id in sampled_ids[0]:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption) 
        print(sentence)

        print('1:up, 2:down, 3:left, 4:right, 5:toggle, 6:grab, 7:mine, 0: craft')

        a = input("Enter a move: ")
        action = get_action_name(int(a))
        game.act(action)

def play_game_by_hand_test():

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device('cuda')
    else:
        print("using cpu")
        device = torch.device('cpu')

    name = "compiled_dataset_08131950" #add 50 back in
    embed_dim = 300 # switch this later!!
    embed_size = embed_dim

    with open('data/'+name+'_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    lstm_embed_dim = 16

    #model = LanguageNetv1(len(vocab), lstm_embed_dim)
    #model = LanguageNetv2(len(vocab), embed_dim, vocab_weights, training=False)
    model = LanguageWithAttention(len(vocab), embed_dim, vocab_weights, training=False)
    model.to(device)
    model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttention_both1.pt"))
    #model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttention.pt")) # trained with embeddings

    count = 0
    game = generate_new_game()

    if embed_size == 300:
        glove = vocabtorch.GloVe(name='840B', dim=300)
    elif embed_size == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)

    count = 0
    game = generate_new_game()

    print(game.game.goal)

    past_moves = []

    while not game.is_over() or count == 250:

        count = count + 1
        state = game.observe()['observation'][0]
        
        #fix this printing so it is easier.. 
        for line in state:
            print(line)

        goal = game.game.goal
        inventory = game.game.inventory

        states_embedding = torch.from_numpy(np.array([get_grid_embedding(state, glove, embed_size)]))
        states_onehot = torch.from_numpy(np.array([one_hot_grid(state, glove, embed_size)]))
        goal = torch.from_numpy(get_goal_embedding(goal, glove, embed_size))
        #inventory = torch.from_numpy(get_inventory_embedding(inventory))
        #inventory = torch.Tensor(get_inventory_embedding(inventory, glove, embed_size))
        inventory = torch.Tensor(np.array([get_inventory_embedding(inventory, glove, embed_size)]))
        #print(inventory)

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        goal = goal.to(device)
        inventory = inventory.to(device)

        sampled_ids, hiddens = model.get_hidden_state_new1(states_embedding, states_onehot, inventory, goal, device, vocab)         
        #sampled_ids = model.sample(states_embedding, states_onehot, inventory, goal, device, vocab)
        #sampled_ids = sampled_ids[0].cpu().numpy() 
        
        sampled_caption = []
        for word_ids in sampled_ids[1:]:
            words = []
            for word_id in word_ids[0]:
                word = vocab.idx2word[word_id]
                words.append(word)
                if word == '<end>':
                    break
            sampled_caption.append(words)
        #sentence = ' '.join(sampled_caption) 
        print(sampled_caption)


        print('1:up, 2:down, 3:left, 4:right, 5:toggle, 6:grab, 7:mine, 0: craft')

        a = input("Enter a move: ")
        action = get_action_name(int(a))
        game.act(action)

def play_game_by_hand_glove():

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device('cuda')
    else:
        print("using cpu")
        device = torch.device('cpu')

    name = "compiled_dataset_08131950" #add 50 back in
    embed_dim = 300 # switch this later!!
    embed_size = embed_dim

    with open('data/'+name+'_all_instructions', 'rb') as f:
        all_instructions = pickle.load(f)

    vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

    vocab_weights = torch.from_numpy(vocab_weights).to(device)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    lstm_embed_dim = 16

    #model = LanguageNetv1(len(vocab), lstm_embed_dim)
    #model = LanguageNetv2(len(vocab), embed_dim, vocab_weights, training=False)
    #model = LanguageWithAttention(len(vocab), embed_dim, vocab_weights, training=False)
    model = LanguageWithAttentionGLOVE(len(vocab), embed_dim, vocab_weights, training=False)
    model.to(device)
    model.load_state_dict(torch.load("TRAINED_MODELS/LanguageWithAttentionGLOVE_01RMSProp.pt"))

    count = 0
    game = generate_new_game()

    if embed_size == 300:
        glove = vocabtorch.GloVe(name='840B', dim=300)
    elif embed_size == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)

    count = 0
    game = generate_new_game()

    print(game.game.goal)

    past_moves = []

    while not game.is_over() or count == 250:

        count = count + 1
        state = game.observe()['observation'][0]
        
        #fix this printing so it is easier.. 
        for line in state:
            print(line)

        goal = game.game.goal
        inventory = game.game.inventory

        states_embedding = torch.from_numpy(np.array([get_grid_embedding(state, glove, embed_size)]))
        states_onehot = torch.from_numpy(np.array([one_hot_grid(state, glove, embed_size)]))
        goal = torch.from_numpy(get_goal_embedding(goal, glove, embed_size))
        inventory = torch.Tensor(np.array([get_inventory_embedding(inventory, glove, embed_size)]))

        states_onehot = states_onehot.to(device)
        states_embedding = states_embedding.to(device)
        goal = goal.to(device)
        inventory = inventory.to(device)

        sampled_ids, hiddens = model.get_hidden_state_new(states_embedding, states_onehot, inventory, goal, device, vocab, vocab_weights)         

        sampled_caption = []
        for word_id in sampled_ids[0]:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption) 
        print(sentence)

        print('1:up, 2:down, 3:left, 4:right, 5:toggle, 6:grab, 7:mine, 0: craft')

        a = input("Enter a move: ")
        action = get_action_name(int(a))
        game.act(action)


#load_model_play_game_with_lang_glove()
#load_model_play_game_with_lang() #this is to evaluate action conditioned on language
#load_model_play_game() # this is to evaluate action
play_game_by_hand() # this is to evaluate language
#play_game_by_hand_test() # this is to evaluate language, get the top5 of language
#play_game_by_hand_glove() #this is to evaluate language, using predicted glove embeddings.
#play_game()

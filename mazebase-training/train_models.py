import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchtext.vocab as vocab
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

#from data_reader import read_dataset
from torchsummary import summary

import seq2vec

import random
import os
import yaml
import json

import mazebasev2.lib.mazebase.games as games
from mazebasev2.lib.mazebase.games import featurizers

from mazebasev2.lib.mazebase.items.terrain import CraftingItem, CraftingContainer, ResourceFont, Block, Water, Switch, Door
from mazebasev2.lib.mazebase.items import agents

import math

## THIS FILE CONTAINS ALL OF THE MODELS USED TO TRAIN...

class LanguageWithAttentionSUM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, embed_weights, max_seq_length=20, training=True):
        super(LanguageWithAttentionSUM, self).__init__()

        encoder_dim = 128
        self.encoder_dim = encoder_dim
        attention_dim = encoder_dim
        embed_dim = embedding_dim
        decoder_dim = 32

        #self.embed = nn.Embedding(vocab_size, embedding_dim) # vocab size, 300
        
        #if training:
        #    self.embed.load_state_dict({'weight': embed_weights})
        #    self.embed.weight.requires_grad = False

        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

        self.fc_embed = nn.Linear(embedding_dim, encoder_dim)
        self.fc_onehot = nn.Linear(7, encoder_dim)
        self.fc_inv = nn.Linear(embedding_dim, encoder_dim)
        self.fc_goal = nn.Linear(embedding_dim, encoder_dim) 
        #self.fc_cat = nn.Linear(25*15+25*15+10*15+15, encoder_dim)

        '''
        self.fc1 = nn.Linear(embedding_dim, 150)
        self.fc2 = nn.Linear(7, 20)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(embedding_dim, 150) 
        self.fc5 = nn.Linear(2250+150, 512)
        self.fc_inv = nn.Linear(embedding_dim, 50) 
        self.fc55 = nn.Linear(512+50, embedding_dim)
        '''

        #self.encoding = nn.LSTM(embedding_dim, 32, num_layers=1)
        #self.linear = nn.Linear(32, vocab_size)

        self.dropout = 0.5

        # new stuff...
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        #self.attention = AttentionSmall(encoder_dim, decoder_dim)
        #self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.embedding = nn.Embedding(vocab_size+1, embed_dim, vocab_size)  # embedding layer

        if training:
            self.embedding.load_state_dict({'weight': embed_weights})
            self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        #self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, grid_embedding, grid_onehot, inventory, goal, encoded_captions, caption_lengths, device, max_seq_length=20):

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        inventory = inventory.view(-1, 1, 300)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.unsqueeze(1)
        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        # DECODER
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        num_pixels = encoder_out.size(1)

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = [caption_length-1 for caption_length in caption_lengths]
        #decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        hiddens = h.clone()

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            hiddens[:batch_size_t] = h.clone() ## ADDED!
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, hiddens


    def get_hidden_state_new(self, grid_embedding, grid_onehot, inventory, goal, device, word_map, states=None):

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c3 = c3.view(-1, 1, self.encoder_dim)
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.view(-1, 1, self.encoder_dim)

        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        k= encoder_out.size(0) #batch size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).to(device)  # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = [[word_map.word2idx['<start>']] for i in range(k)]
        incomplete_inds = [i for i in range(k)] # used to keep track of original index in complete_seqs
        
        #complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)
        hiddens = h.clone()

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            hiddens[incomplete_inds] = h.clone()
            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            values, indices = scores.max(dim=1) 

            assert(indices.size(0) == len(incomplete_inds))

            temp = []
            for i in range(indices.size(0)-1, -1, -1):
                complete_seqs[incomplete_inds[i]].append(indices.data.tolist()[i])
                if indices[i] == word_map.word2idx['<end>']:
                    del incomplete_inds[i]
                    #incomplete_inds.remove(i)
                else:
                    #not finished
                    temp.append(i)

            if len(incomplete_inds) == 0:
                break

            #subset the ones that aren't finished.
            h = h[temp]
            c = c[temp]
            encoder_out = encoder_out[temp]
            k_prev_words = indices[temp].unsqueeze(1)

            # Break if things have been going on too long
            if step > 20:
                break
            step += 1

        return complete_seqs, hiddens


class SimpleNetwork(nn.Module):
  def __init__(self, embed_dim):
    super(SimpleNetwork, self).__init__()

    self.embed_dim = embed_dim

    self.fc1 = nn.Linear(embed_dim, 150)
    self.fc2 = nn.Linear(7, 20)
    self.fc3 = nn.Linear(170, 90)
    self.fc4 = nn.Linear(embed_dim, 150) 
    self.fc5 = nn.Linear(2250+150, 512)
    self.fc_inv = nn.Linear(embed_dim, 50) 
    self.fc55 = nn.Linear(512+50, 48)

    self.fc6 = nn.Linear(48+32, 48)
    self.fc7 = nn.Linear(48, 8)


  def forward(self, grid_embedding, grid_onehot, inventory, goal, hidden):

        #encode features
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        temp_inv = F.relu(self.fc_inv(inventory))
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        features = F.relu(self.fc55(combined_inventory))


        all_comb = torch.cat((features, hidden), dim=1)
        
        c6 = F.relu(self.fc6(all_comb)) # updated with new embedding size.
        c7 = self.fc7(c6)
        #c8 = F.relu(self.fc8(c7))

        return c7

class CNNAction(nn.Module):

    #https://github.com/galsang/CNN-sentence-classification-pytorch/blob/master/run.py
    def __init__(self, embed_dim, vocab, vocab_weights):
        super(CNNAction, self).__init__()

        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(embed_dim, 150)
        self.fc2 = nn.Linear(7, 20)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(embed_dim, 150) 
        self.fc5 = nn.Linear(2250+150, 512)
        self.fc_inv = nn.Linear(embed_dim, 32) 
        self.fc55 = nn.Linear(512+320, 48)

        self.fc6 = nn.Linear(48, 48)
        self.fc7 = nn.Linear(48, 8)

        self.WORD_DIM = 300

        self.embedding = nn.Embedding(num_embeddings=len(vocab)+1,
                                          embedding_dim=self.WORD_DIM, padding_idx=len(vocab))
        #self.embedding.load_state_dict({'weight': vocab_weights})
        self.embedding.weight.data.copy_(vocab_weights)
        self.embedding.weight.requires_grad = False

        self.FILTER_NUM = 100
        self.conv1 = nn.Conv1d(1, self.FILTER_NUM, self.WORD_DIM * 2, stride=self.WORD_DIM)
        self.conv2 = nn.Conv1d(1, self.FILTER_NUM, self.WORD_DIM * 3, stride=self.WORD_DIM)
        self.conv3 = nn.Conv1d(1, self.FILTER_NUM, self.WORD_DIM * 4, stride=self.WORD_DIM)

        #self.fc = nn.Linear(sum(self.FILTER_NUM), 8)

        self.fc_comb = nn.Linear(48+300, 8)
        self.max_len = 20

    #def get_conv(self, i):
    #    return getattr(self, f'conv_{i}')

    def forward(self, grid_embedding, grid_onehot, inventory, goal, language):
        x = self.embedding(language)
        x = x.view(-1, 1, 300 * self.max_len)

        x1 = F.max_pool1d(F.relu(self.conv1(x)), self.max_len - 2 + 1)
        x1 = x1.view(-1, self.FILTER_NUM)
        x2 = F.max_pool1d(F.relu(self.conv2(x)), self.max_len - 3 + 1)
        x2 = x2.view(-1, self.FILTER_NUM)
        x3 = F.max_pool1d(F.relu(self.conv3(x)), self.max_len - 4 + 1)
        x3 = x3.view(-1, self.FILTER_NUM)

        conv_results = [x1, x2, x3]

        #F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i])

        #conv_results = [
        #    F.max_pool1d(F.relu(self.get_conv(i)(x)), 20 - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=0.5, training=self.training)

        #encode features
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))

        #changing inventory.. 
        temp_inv = F.relu(self.fc_inv(inventory))
        temp_inv = temp_inv.view(-1, 10*32)
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        features = F.relu(self.fc55(combined_inventory))
        
        #c6 = F.relu(self.fc6(features)) # updated with new embedding size.
        #c7 = self.fc7(c6)

        comb_lang_feat = torch.cat((features, x), dim = 1)

        output = F.log_softmax(self.fc_comb(comb_lang_feat), dim=1)
        return output

        #return c7


class BOW(nn.Module):
    def __init__(self, vocab, embed_weights, emb_size):
        super(BOW, self).__init__()
        self.vocab = vocab
        self.emb_size = emb_size
        #self.embedding = nn.Embedding(num_embeddings=len(self.vocab),
        #                              embedding_dim=emb_size)

        self.embedding = nn.Embedding(num_embeddings=len(self.vocab)+1,
                                          embedding_dim=emb_size, padding_idx=len(self.vocab))
        self.embedding.load_state_dict({'weight': embed_weights})
        self.embedding.weight.requires_grad = False

        self.linear = nn.Linear(300, 32)

    def forward(self, input):
        output = self.embedding(input)
        output = output.sum(1)  #try mean?
        #output = F.relu(self.linear(output))
        return output

def select_last(x, lengths):
    batch_size = x.size(0)
    seq_length = x.size(1)
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        mask[i][lengths[i]-1].fill_(1)
    mask = Variable(mask)
    x = x.mul(mask)
    x = x.sum(1).view(batch_size, x.size(2))
    return x

def process_lengths(input):
    max_length = input.size(1)
    if input.size(0) == 1:
        lengths = [max_length - input.data.eq(0).sum(1).squeeze()]
    else:
        lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
    return lengths

class GRU(nn.Module):
    def __init__(self, vocab, embed_weights, emb_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.vocab = vocab
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab)+1,
                                      embedding_dim=emb_size,
                                      padding_idx=len(self.vocab))
        self.embedding.load_state_dict({'weight': embed_weights})
        self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input):
        lengths = process_lengths(input)
        x = self.embedding(input)
        self.rnn.flatten_parameters()
        output, hn = self.rnn(x)
        output = select_last(output, lengths)
        return output

#write my own LSTM?

# Simple observation prediction model
# Uses an attention mechanism to deal with multiple inputs
# seq2vec, fc or conv heads for different obs followed by attention combination
class AllObsPredictAtten(nn.Module):
    def __init__(self, embed_dim, vocab_weights, with_pred=False, init_fn=None, opt=None, env_vocab=None, vocab_words=None, max_bounds=None, num_stack=1, add_net_pred=False, pred_size=None, **kwargs):
        super(AllObsPredictAtten, self).__init__()

        # Set state vars
        self.hid_sz = 32
        self.use_dropout = False
        self.embed_dim = embed_dim

        # Make seq2vec model for statement -- might have to change this!
        seq2vec_opt = {}
        seq2vec_opt['arch'] = 'bow'
        seq2vec_opt['dropout'] = False
        seq2vec_opt['emb_size'] = 32
        seq2vec_opt['hidden_size'] = 32
        #self.seq2vec = BOW(vocab_words, vocab_weights, 300)
        self.seq2vec = GRU(vocab_words, vocab_weights, embed_dim, 32, 1)
        
        #fc layers
        self.fc_gridembed = nn.Linear(25*embed_dim, 32)
        self.fc_onehot = nn.Linear(25*7, 32)
        self.fc_inv = nn.Linear(10*embed_dim, 32) # 10 inventory slots
        self.fc_goal = nn.Linear(embed_dim, 32) 

        self.fc_embed = nn.Linear(300, 32) # compress too much??
        self.fc_onehot_embed = nn.Linear(7, 32)
        self.fc_comb = nn.Linear(25*32, 32)
        self.fc_inv_first = nn.Linear(300, 32)
        self.fc_inv_second = nn.Linear(32*10, 32)

        # Decide what inputs go in net and key

        # key is for attention
        key_sz = 32 # 32 for GRU and 300 for BOW
        self.key_inputs = ['wids']

        # net is for regular input
        #self.net_inputs = ['grid_embed', 'grid_onehot', 'inv', 'goal']
        self.net_inputs = ['grid_comb', 'inv', 'goal']
        net_input_sizes = []
        for k in self.net_inputs:
            net_input_sizes.append(self.hid_sz)

        # Make batch modules
        module_opt = {}
        module_opt['num_modules'] = 12 #16
        module_opt['switch_sz'] = key_sz
        module_opt['hid_sz'] = self.hid_sz
        module_opt['num_layer'] = 2
        self.batch_modules = SwitchModule(net_input_sizes, len(net_input_sizes), module_opt)

        # Transfer layers
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)

        self.fc_final = nn.Linear(32, 8) # was 9

    # Forward pass
    def forward(self, grid_embedding, grid_onehot, inventory, goal, statement):

        #encode environment observation
        '''
        grid_embedding = grid_embedding.view(-1, 25*self.embed_dim)
        grid_embed = self.tanh(self.fc_gridembed(grid_embedding))
        grid_onehot = grid_onehot.view(-1, 25*7)
        grid_1hot_embed = self.tanh(self.fc_onehot(grid_onehot))
        inventory = inventory.view(-1, 10*self.embed_dim)
        inv_embed = F.relu(self.fc_inv(inventory))
        goal_embed = F.relu(self.fc_goal(goal))
        '''    

        #combined grid embedding
        
        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot_embed(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25*self.hid_sz)
        grid_comb = F.relu(self.fc_comb(grid_comb))

        c3 = F.relu(self.fc_inv_first(inventory))
        inventory = c3.view(-1, 10*self.hid_sz)
        inv_embed = F.relu(self.fc_inv_second(inventory))
        goal_embed = F.relu(self.fc_goal(goal))

        # Encode sequence
        #statement = self.seq2vec(statement) # just BOW
        #statement = self.seq2vec(statement, lengths) # for GRU/LSTM

        # Get key and net inputs
        key = []
        if 'wids' in self.key_inputs:
            key.append(statement)
        key = torch.cat(key, 1)

        net_inputs = ()
        if 'grid_comb' in self.net_inputs:
            net_inputs += (grid_comb,)
        if 'grid_embed' in self.net_inputs:
            net_inputs += (grid_embed,)
        if 'grid_onehot' in self.net_inputs:
            net_inputs += (grid_1hot_embed,)
        if 'inv' in self.net_inputs:
            net_inputs += (inv_embed,)
        if 'goal' in self.net_inputs:
            net_inputs += (goal_embed,)

        # Forward through module net
        x = self.batch_modules((net_inputs, key))

        # Optionally add dropout
        if self.use_dropout:
            x = self.drop(x)

        #add final layer in
        #x = self.softmax(self.fc_final(x))
        #x = F.log_softmax(self.fc_final(x), dim=1) 
        x = self.fc_final(self.dropout(x))

        return x

# This module implements a soft network switch
# It takes a set of network inputs and a soft selection input
# Then it runs the inputs through a network(s) depending on the selection
# And finally combined the output
class SwitchModule(nn.Module):
    def __init__(self, input_sz, num_inputs, opt):
        super(SwitchModule, self).__init__()

        # Get parameters
        self.input_sz = input_sz
        self.num_inputs = num_inputs
        self.num_modules = opt['num_modules']
        self.hid_sz = opt['hid_sz']
        num_layer = opt['num_layer']
        assert(len(self.input_sz) == num_inputs)
        assert(self.num_modules % num_inputs == 0)

        # Make batch modules
        self.batch_modules = []
        for module_in_size in self.input_sz:
            bm_input = BatchMLP(module_in_size, self.hid_sz, num_layer, self.num_modules//num_inputs)
            self.batch_modules.append(bm_input)
        self.batch_modules = ListModule(*self.batch_modules)

        # Make soft attention network components (if applicible)
        self.switch_sz = opt['switch_sz']
        self.att_in = nn.Linear(self.switch_sz, self.num_modules)
        self.softmax = nn.Softmax(dim=1)

    # Forward (mainly switch between soft and hard modes)
    def forward(self, inputs):
        net_inputs = inputs[0]

        # Compute batch module output
        assert(len(net_inputs) == self.num_inputs)
        all_module_outs = []
        for i, net_input in enumerate(net_inputs):
            batch_inputs = net_input.unsqueeze(1).expand([-1, self.num_modules//self.num_inputs, -1])
            module_outs = self.batch_modules[i](batch_inputs) # module_outs is bs x nm//ni x out_sz
            all_module_outs.append(module_outs)
        module_outs = torch.cat(all_module_outs, 1)

        # Soft attention on output
        switch_input = inputs[1]
        selection = self.softmax(self.att_in(switch_input))
        selection = selection.unsqueeze(2)
        selection = selection.repeat([1, 1, module_outs.size(2)])
        module_outs *= selection
        module_outs = module_outs.sum(1)

        return module_outs

# Batch MLP module
# Basically does a for loop over MLPs, but does this efficiently using bmm
class BatchMLP(nn.Module):
    def __init__(self, input_sz, hid_sz, num_layer, num_modules):
        super(BatchMLP, self).__init__()

        # Make network values
        self.tanh = nn.Tanh()
        self.in_fc = BatchLinear(input_sz, hid_sz, num_modules)
        assert(num_layer >= 2) # If num_layer is 2, actually no hidden layers technically
        hid_layers = []
        for i in range(0, num_layer-2):
            hid_fc = BatchLinear(hid_sz, hid_sz, num_modules)
            hid_layers.append(hid_fc)
        self.hid_layers = ListModule(*hid_layers)
        self.out_fc = BatchLinear(hid_sz, hid_sz, num_modules)

    # Input batch_size x num_modules x input_sz
    # Output batch_size x num_modules x output_sz
    def forward(self, input):
        x = self.in_fc(input)
        x = self.tanh(x)
        for hid_fc in self.hid_layers:
            x = hid_fc(x)
            x = self.tanh(x)
        x = self.out_fc(x)
        return x

# Use ListModule code from fmassa (shouldn't this be in pytorch already?)
class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

# BatchLinear module
# Same as nn.Linear, but it takes list of inputs x and outputs list of outputs y
# Equivalent to same operation if we did a for loop and did nn.Linear for each
class BatchLinear(nn.Module):
    def __init__(self, in_features, out_features, num_modules, bias=True):
        super(BatchLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modules = num_modules
        self.weight = nn.Parameter(torch.Tensor(num_modules, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_modules, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # Input batch_size x num_module x input_sz
    # Output batch_size x num_module x output_sz
    def forward(self, input):
        # Get sizes
        bs = input.size(0)
        nm = input.size(1)
        assert(input.size(2) == self.in_features)

        # Transpose input to correct shape
        input = input.transpose(0, 1).transpose(1, 2) # nm x in_sz x bs

        # Compute matrix multiply
        output = torch.bmm(self.weight, input)

        # Add bias
        if self.bias is not None:
            output += self.bias.unsqueeze(2).expand([-1, -1, bs]).contiguous()

        # Transpose back to bs x nm x out_sz
        output = output.transpose(1, 2).transpose(0, 1)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Attention(nn.Module):
    """
    Attention Network. https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class LanguageWithAttentionGLOVE(nn.Module):

    def __init__(self, vocab_size, embedding_dim, embed_weights, max_seq_length=20, training=True):
        super(LanguageWithAttentionGLOVE, self).__init__()

        encoder_dim = 128
        self.encoder_dim = encoder_dim
        attention_dim = encoder_dim
        embed_dim = embedding_dim
        decoder_dim = 32

        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

        self.embedding_dim = embedding_dim
        self.fc_embed = nn.Linear(embedding_dim, encoder_dim)
        self.fc_onehot = nn.Linear(7, encoder_dim)
        self.fc_inv = nn.Linear(embedding_dim, encoder_dim)
        self.fc_goal = nn.Linear(embedding_dim, encoder_dim) 

        self.dropout = 0.5

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size+1, embed_dim, padding_idx=vocab_size)  # embedding layer

        if training:
            self.embedding.load_state_dict({'weight': embed_weights})
            self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, embed_dim)  # linear layer to find scores over vocabulary

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, grid_embedding, grid_onehot, inventory, goal, encoded_captions, caption_lengths, device, max_seq_length=20):

        #encode features

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.unsqueeze(1)
        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        # DECODER
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        num_pixels = encoder_out.size(1)

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        embeddings_copy = embeddings.clone()

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = [caption_length-1 for caption_length in caption_lengths]
        #decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.embedding_dim).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        hiddens = h.clone()

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            hiddens[:batch_size_t] = h.clone() ## ADDED!
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, hiddens, embeddings_copy

    def get_hidden_state_new(self, grid_embedding, grid_onehot, inventory, goal, device, word_map, glove_tensor, states=None):

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c3 = c3.view(-1, 10, self.encoder_dim)
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.view(-1, 1, self.encoder_dim)

        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        k= encoder_out.size(0) #batch size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).to(device)  # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = [[word_map.word2idx['<start>']] for i in range(k)]
        incomplete_inds = [i for i in range(k)] # used to keep track of original index in complete_seqs
        
        #complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)
        hiddens = h.clone()

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            hiddens[incomplete_inds] = h.clone()
            scores = self.fc(h)  # (s, vocab_size)
            #scores = F.log_softmax(scores, dim=1)
            #values, indices = scores.max(dim=1) 

            indices = torch.LongTensor([[0]] * k).to(device) 

            # Find nearest Glove neighbor:
            for i in range(scores.size(0)):

                min_key = None
                min_dist = None
                for y in range(glove_tensor.size(0)):
                    dist = torch.norm(glove_tensor[y]-scores[i], 2)
                    if min_key == None:
                        min_key = y
                        min_dist = dist
                    elif dist < min_dist:
                        min_key = y
                        min_dist = dist

                indices[i] = min_key
                #print(word_map.idx2word[min_key])

            # End finding glove neighbor

            assert(indices.size(0) == len(incomplete_inds))

            temp = []
            for i in range(indices.size(0)-1, -1, -1):
                complete_seqs[incomplete_inds[i]].append(indices.data.tolist()[i][0])
                if indices[i] == word_map.word2idx['<end>']:
                    del incomplete_inds[i]
                    #incomplete_inds.remove(i)
                else:
                    #not finished
                    temp.append(i)

            if len(incomplete_inds) == 0:
                break

            #subset the ones that aren't finished.
            h = h[temp]
            c = c[temp]
            encoder_out = encoder_out[temp]
            k_prev_words = indices[temp] #.unsqueeze(1)

            # Break if things have been going on too long
            if step > 20:
                break
            step += 1

        return complete_seqs, hiddens


class LanguageWithAttention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, embed_weights, max_seq_length=20, training=True):
        super(LanguageWithAttention, self).__init__()

        encoder_dim = 128
        self.encoder_dim = encoder_dim
        attention_dim = encoder_dim
        embed_dim = embedding_dim
        decoder_dim = 32

        #self.embed = nn.Embedding(vocab_size, embedding_dim) # vocab size, 300
        
        #if training:
        #    self.embed.load_state_dict({'weight': embed_weights})
        #    self.embed.weight.requires_grad = False

        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

        self.fc_embed = nn.Linear(embedding_dim, encoder_dim)
        self.fc_onehot = nn.Linear(7, encoder_dim)
        self.fc_inv = nn.Linear(embedding_dim, encoder_dim)
        self.fc_goal = nn.Linear(embedding_dim, encoder_dim) 
        #self.fc_cat = nn.Linear(25*15+25*15+10*15+15, encoder_dim)

        '''
        self.fc1 = nn.Linear(embedding_dim, 150)
        self.fc2 = nn.Linear(7, 20)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(embedding_dim, 150) 
        self.fc5 = nn.Linear(2250+150, 512)
        self.fc_inv = nn.Linear(embedding_dim, 50) 
        self.fc55 = nn.Linear(512+50, embedding_dim)
        '''

        #self.encoding = nn.LSTM(embedding_dim, 32, num_layers=1)
        #self.linear = nn.Linear(32, vocab_size)

        self.dropout = 0.5

        # new stuff...
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        #self.attention = AttentionSmall(encoder_dim, decoder_dim)
        #self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.embedding = nn.Embedding(vocab_size+1, embed_dim, vocab_size)  # embedding layer

        if training:
            self.embedding.load_state_dict({'weight': embed_weights})
            self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        #self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, grid_embedding, grid_onehot, inventory, goal, encoded_captions, caption_lengths, device, max_seq_length=20):

        #encode features
        '''
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        temp_inv = F.relu(self.fc_inv(inventory))
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        encoder_out = F.relu(self.fc55(combined_inventory))
        '''

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.unsqueeze(1)
        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        # DECODER
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        num_pixels = encoder_out.size(1)

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = [caption_length-1 for caption_length in caption_lengths]
        #decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        hiddens = h.clone()

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            hiddens[:batch_size_t] = h.clone() ## ADDED!
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, hiddens

        '''
        #decoder...
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = [caption_length-1 for caption_length in caption_lengths]
        #decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), self.encoder_dim).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(h)
            #preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas
        '''

    #get new instructions, https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py
    def sample(self, grid_embedding, grid_onehot, inventory, goal, device, word_map, states=None):
        """Generate captions for given image features using greedy search."""
        
        #encode features
        '''
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        temp_inv = F.relu(self.fc_inv(inventory))
        temp_inv = temp_inv.view(-1, 50)
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        encoder_out = F.relu(self.fc55(combined_inventory))
        '''
        #encode features
        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.unsqueeze(1)

        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        num_pixels = encoder_out.size(1)
        k= 5

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, self.encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)

            awe = gate * awe

            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / self.vocab_size  # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                #complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            #print(len(incomplete_inds), len(complete_inds))

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        #print(seq)
        #return seq
        return complete_seqs
        #return complete_seqs, complete_seqs_scores

    #get new instructions, https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py
    def sample_new(self, grid_embedding, grid_onehot, inventory, goal, device, word_map, states=None):
        """Generate captions for given image features using greedy search."""
        
        #encode features
        '''
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        temp_inv = F.relu(self.fc_inv(inventory))
        temp_inv = temp_inv.view(-1, 50)
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        encoder_out = F.relu(self.fc55(combined_inventory))
        '''
        #encode features
        # c1 = F.relu(self.fc_embed(grid_embedding))
        # c2 = F.relu(self.fc_onehot(grid_onehot))
        # grid_comb = c1 + c2 
        # grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        # c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        # c3 = c3.view(-1, 10, self.encoder_dim)
        # c4 = F.relu(self.fc_goal(goal))
        # c4 = c4.view(-1, 1, self.encoder_dim)

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.unsqueeze(1)
        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        num_pixels = encoder_out.size(1)
        k= 5

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, self.encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)

            awe = gate * awe

            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / self.vocab_size  # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                #complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            #print(len(incomplete_inds), len(complete_inds))

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        #print(seq)
        return seq
        #return complete_seqs, complete_seqs_scores

    #write a get hidden state that uses CUDA.

    '''
    #NEW VERSION!
    def get_hidden_state(self, grid_embedding, grid_onehot, inventory, goal, device, word_map, states=None):
        """Generate captions for given image features using greedy search."""
        
        #encode features
        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c3 = c3.view(-1, 10, self.encoder_dim)
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.view(-1, 1, self.encoder_dim)

        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        k= encoder_out.size(0) #batch size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / self.vocab_size  # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        return complete_seqs
    '''
    #encode features
    def get_hidden_state(self, grid_embedding, grid_onehot, inventory, goal, device, word_map, states=None):

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c3 = c3.view(-1, 10, self.encoder_dim)
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.view(-1, 1, self.encoder_dim)

        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        k= encoder_out.size(0) #batch size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).to(device)  # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = [[word_map.word2idx['<start>']] for i in range(k)]
        incomplete_inds = [i for i in range(k)] # used to keep track of original index in complete_seqs
        
        #complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)
        #hiddens = h.clone()

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            #hiddens[incomplete_inds] = h.clone()
            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            values, indices = scores.max(dim=1) 

            assert(indices.size(0) == len(incomplete_inds))

            temp = []
            for i in range(indices.size(0)-1, -1, -1):
                complete_seqs[incomplete_inds[i]].append(indices.data.tolist()[i])
                if indices[i] == word_map.word2idx['<end>']:
                    del incomplete_inds[i]
                    #incomplete_inds.remove(i)
                else:
                    #not finished
                    temp.append(i)

            if len(incomplete_inds) == 0:
                break

            #subset the ones that aren't finished.
            h = h[temp]
            c = c[temp]
            encoder_out = encoder_out[temp]
            k_prev_words = indices[temp].unsqueeze(1)

            # Break if things have been going on too long
            if step > 20:
                break
            step += 1

        return complete_seqs

    def get_hidden_state_new(self, grid_embedding, grid_onehot, inventory, goal, device, word_map, states=None):

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c3 = c3.view(-1, 10, self.encoder_dim)
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.view(-1, 1, self.encoder_dim)

        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        k= encoder_out.size(0) #batch size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).to(device)  # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = [[word_map.word2idx['<start>']] for i in range(k)]
        incomplete_inds = [i for i in range(k)] # used to keep track of original index in complete_seqs
        
        #complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)
        hiddens = h.clone()

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            hiddens[incomplete_inds] = h.clone()
            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            values, indices = scores.max(dim=1) 

            assert(indices.size(0) == len(incomplete_inds))

            temp = []
            for i in range(indices.size(0)-1, -1, -1):
                complete_seqs[incomplete_inds[i]].append(indices.data.tolist()[i])
                if indices[i] == word_map.word2idx['<end>']:
                    del incomplete_inds[i]
                    #incomplete_inds.remove(i)
                else:
                    #not finished
                    temp.append(i)

            if len(incomplete_inds) == 0:
                break

            #subset the ones that aren't finished.
            h = h[temp]
            c = c[temp]
            encoder_out = encoder_out[temp]
            k_prev_words = indices[temp].unsqueeze(1)

            # Break if things have been going on too long
            if step > 20:
                break
            step += 1

        return complete_seqs, hiddens

    def get_hidden_state_new1(self, grid_embedding, grid_onehot, inventory, goal, device, word_map, states=None):

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c3 = c3.view(-1, 10, self.encoder_dim)
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.view(-1, 1, self.encoder_dim)

        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        k= encoder_out.size(0) #batch size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).to(device)  # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = [[word_map.word2idx['<start>']] for i in range(k)]
        incomplete_inds = [i for i in range(k)] # used to keep track of original index in complete_seqs
        
        #complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)
        hiddens = h.clone()

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            hiddens[incomplete_inds] = h.clone()
            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            #values, indices = scores.max(dim=1) 
            values, indices = torch.topk(scores, 5, dim=1)

            #assert(indices.size(0) == len(incomplete_inds))

            top5 = indices.data.tolist()

            complete_seqs.append(top5)

            #if word_map.word2idx['<end>'] in top5:
            #    break

            values, indices = scores.max(dim=1) 

            temp = []
            for i in range(indices.size(0)-1, -1, -1):
                #complete_seqs[incomplete_inds[i]].append(indices.data.tolist()[i])
                if indices[i] == word_map.word2idx['<end>']:
                    del incomplete_inds[i]
                    #incomplete_inds.remove(i)
                else:
                    #not finished
                    temp.append(i)

            if len(incomplete_inds) == 0:
                 break

            #subset the ones that aren't finished.
            h = h[temp]
            c = c[temp]
            encoder_out = encoder_out[temp]
            k_prev_words = indices[temp].unsqueeze(1)

            # Break if things have been going on too long
            if step > 20:
                break
            step += 1

        return complete_seqs, hiddens
    

class LanguageWithAttention1(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, embed_weights, max_seq_length=20, training=True):
        super(LanguageWithAttention1, self).__init__()

        self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
        
        if training:
            self.embed.load_state_dict({'weight': embed_weights})
            self.embed.weight.requires_grad = False

        #if not trainable.
        #self.emb_layer.weight.requires_grad = False # try this too?

        self.max_seq_length = max_seq_length

        self.fc1 = nn.Linear(embedding_dim, 150)
        self.fc2 = nn.Linear(7, 20)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(embedding_dim, 150) 
        self.fc5 = nn.Linear(2250+150, 512)
        self.fc_inv = nn.Linear(embedding_dim, 50) 
        self.fc55 = nn.Linear(512+50, embedding_dim)

        self.attention = AttentionSmall(embedding_dim, 32, 128)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.encoding = nn.LSTM(embedding_dim, 32, num_layers=1)
        self.linear = nn.Linear(32, num_embeddings)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        #mean_encoder_out = encoder_out.mean(dim=1)
        #print(mean_encoder_out.size())
        h = self.init_h(encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(encoder_out)
        return h, c

    def forward(self, grid_embedding, grid_onehot, inventory, goal, encoded_captions, caption_lengths, device, max_seq_length=20):

        #encode features

        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        temp_inv = F.relu(self.fc_inv(inventory))
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        encoder_out = F.relu(self.fc55(combined_inventory))

        #decoder...
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = [caption_length-1 for caption_length in caption_lengths]
        #decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths

#v1 of most basic behavioral cloning
class StateGoalNet(nn.Module):
  def __init__(self):
    super(StateGoalNet, self).__init__()

    self.fc1 = nn.Linear(300, 150)
    self.fc2 = nn.Linear(6, 20)
    self.fc3 = nn.Linear(170, 90)
    self.fc4 = nn.Linear(300, 150) 
    self.fc5 = nn.Linear(2250+150, 528)
    self.fc6 = nn.Linear(528, 128)
    self.fc7 = nn.Linear(128, 8)

  def forward(self, grid_embedding, grid_onehot, goal):

    c1 = F.relu(self.fc1(grid_embedding))
    c2 = F.relu(self.fc2(grid_onehot))
    c1 = c1.view(-1, 25,150)
    c2 = c2.view(-1, 25,20)
    combined_grids = torch.cat((c1, c2), dim=2)
    c3 = F.relu(self.fc3(combined_grids)) 
    c3 = c3.view(-1, 25*90)
    c4 = F.relu(self.fc4(goal))
    combined_grid_goal = torch.cat((c3, c4), dim=1)
    c5 = F.relu(self.fc5(combined_grid_goal))
    c6 = F.relu(self.fc6(c5))
    c7 = self.fc7(c6)
    return c7

#v1 of most basic behavioral cloning, with inventory -- using basically the same network as ActionNetv1 for comparison!
class StateGoalNetv1(nn.Module):
  def __init__(self, embed_dim):
    super(StateGoalNetv1, self).__init__()

    self.embed_dim = embed_dim

    self.fc1 = nn.Linear(embed_dim, 150)
    self.fc2 = nn.Linear(7, 20)
    self.fc3 = nn.Linear(170, 90)
    self.fc4 = nn.Linear(embed_dim, 150) 
    self.fc5 = nn.Linear(2250+150, 512)
    self.fc_inv = nn.Linear(embed_dim, 50) 
    self.fc55 = nn.Linear(512+50, 48)

    self.fc6 = nn.Linear(48, 48)
    self.fc7 = nn.Linear(48, 8)


  def forward(self, grid_embedding, grid_onehot, inventory, goal):

        #encode features
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        temp_inv = F.relu(self.fc_inv(inventory))
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        features = F.relu(self.fc55(combined_inventory))
        
        c6 = F.relu(self.fc6(features)) # updated with new embedding size.
        c7 = self.fc7(c6)
        #c8 = F.relu(self.fc8(c7))

        return c7


#Adding all at once. 
class StateGoalNetv2(nn.Module):
  def __init__(self, embed_dim):
    super(StateGoalNetv2, self).__init__()

    # saying that 50 is the embedding size
    # saying that 32 is the hidden size.  

    self.embed_dim = embed_dim
    self.fc_gridembed = nn.Linear(25*embed_dim, 32)
    self.fc_onehot = nn.Linear(25*6, 32)
    self.fc_inv = nn.Linear(embed_dim, 32)
    self.fc_goal = nn.Linear(embed_dim, 32) 
    self.fc1 = nn.Linear(32, 32)
    self.fc2 = nn.Linear(32, 8)
    #self.fc6 = nn.Linear(300, 50)
    #self.fc7 = nn.Linear(50, 8)

  def forward(self, grid_embedding, grid_onehot, inventory, goal):

    #reshape grid embeddings, also try Tanh
    grid_embedding = grid_embedding.view(-1, 25*self.embed_dim)
    grid_embed = F.relu(self.fc_gridembed(grid_embedding))
    grid_onehot = grid_onehot.view(-1, 25*6)
    grid_1hot_embed = F.relu(self.fc_onehot(grid_onehot))
    inv_embed = F.relu(self.fc_inv(inventory))
    goal_embed = F.relu(self.fc_goal(goal))

    total = grid_embed + grid_1hot_embed + inv_embed + goal_embed

    output = F.relu(self.fc1(total))
    output = F.relu(self.fc2(output))

    return output

    #do MLP (later- do the SwitchModule from kenny's code) 

    # c1 = F.relu(self.fc1(grid_embedding))
    # c2 = F.relu(self.fc2(grid_onehot))
    # c1 = c1.view(-1, 25*20)
    # c2 = c2.view(-1, 25*20)
    # combined_grids = c1 + c2
    # c3 = F.relu(self.fc3(combined_grids)) 
    # c4 = F.relu(self.fc4(goal))
    # combined_grid_goal = c3 + c4
    # c5 = F.relu(self.fc5(combined_grid_goal))
    # c6 = F.relu(self.fc6(inventory))
    # combined_inventory = c5 + c6
    # c7 = F.relu(self.fc7(combined_inventory))
    # #c7 = F.relu(self.fc7(c6))
    # return c7

# behavioral cloning with instruction as input
class StateGoalInstructionNet(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super(StateGoalInstructionNet, self).__init__()

    #num_embeddings, embedding_dim = embed_weights.size()

    self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
    #self.embed.load_state_dict({'weight': embed_weights})

    #if not trainable.
    #self.emb_layer.weight.requires_grad = False # try this too?

    self.fc1 = nn.Linear(300, 150)
    self.fc2 = nn.Linear(6, 20)
    self.fc3 = nn.Linear(170, 90)
    self.fc4 = nn.Linear(300, 150) 
    self.fc5 = nn.Linear(2250+150, 528)
    self.fc6 = nn.Linear(528+512, 128)
    self.fc7 = nn.Linear(128, 64)
    self.fc8 = nn.Linear(64,9) # CHANGE THIS BACK TO 8!!
    self.encoding = nn.LSTM(256, 512, num_layers=1)

  def forward(self, grid_embedding, grid_onehot, goal, instruction, lengths):

    emb = self.embed(instruction)

    packed = pack_padded_sequence(emb, lengths, batch_first=True) 

    output, hdn = self.encoding(packed)

    #print(hdn[0][0].size())

    c1 = F.relu(self.fc1(grid_embedding))
    c2 = F.relu(self.fc2(grid_onehot))
    c1 = c1.view(-1, 25,150)
    c2 = c2.view(-1, 25,20)
    combined_grids = torch.cat((c1, c2), dim=2)
    c3 = F.relu(self.fc3(combined_grids)) 
    c3 = c3.view(-1, 25*90)
    c4 = F.relu(self.fc4(goal))
    combined_grid_goal = torch.cat((c3, c4), dim=1)
    c5 = F.relu(self.fc5(combined_grid_goal))
    #print(c5.size())
    combined_instruction = torch.cat((c5, hdn[0][0]), dim=1)
    c6 = F.relu(self.fc6(combined_instruction))
    c7 = F.relu(self.fc7(c6))
    c8 = F.relu(self.fc8(c7))

    return c8

# behavioral cloning with instruction as input (WITH INVENTORY)
class StateGoalInstructionv1Net(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super(StateGoalInstructionv1Net, self).__init__()

    #num_embeddings, embedding_dim = embed_weights.size()

    self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
    #self.embed.load_state_dict({'weight': embed_weights})

    #if not trainable.
    #self.emb_layer.weight.requires_grad = False # try this too?

    self.fc1 = nn.Linear(300, 150)
    self.fc2 = nn.Linear(6, 20)
    self.fc3 = nn.Linear(170, 90)
    self.fc4 = nn.Linear(300, 150) 
    self.fc5 = nn.Linear(2250+150, 528)
    self.fc6 = nn.Linear(528+512, 128)
    self.fc7 = nn.Linear(128+300, 64)
    self.fc8 = nn.Linear(64,9)
    self.encoding = nn.LSTM(256, 512, num_layers=1)

  def forward(self, grid_embedding, grid_onehot, inventory, goal, instruction, lengths):

    emb = self.embed(instruction)

    packed = pack_padded_sequence(emb, lengths, batch_first=True) 

    output, hdn = self.encoding(packed)

    #print(hdn[0][0].size())

    c1 = F.relu(self.fc1(grid_embedding))
    c2 = F.relu(self.fc2(grid_onehot))
    c1 = c1.view(-1, 25,150)
    c2 = c2.view(-1, 25,20)
    combined_grids = torch.cat((c1, c2), dim=2)
    c3 = F.relu(self.fc3(combined_grids)) 
    c3 = c3.view(-1, 25*90)
    c4 = F.relu(self.fc4(goal))
    combined_grid_goal = torch.cat((c3, c4), dim=1)
    c5 = F.relu(self.fc5(combined_grid_goal))
    #print(c5.size())
    combined_instruction = torch.cat((c5, hdn[0][0]), dim=1)
    c6 = F.relu(self.fc6(combined_instruction))
    combined_inventory = torch.cat((c6, inventory), dim=1)
    c7 = F.relu(self.fc7(combined_inventory))
    c8 = F.relu(self.fc8(c7))

    return c8

class LanguageNet(nn.Module):

	def __init__(self, num_embeddings, embedding_dim, max_seq_length=20):
	    super(LanguageNet, self).__init__()

	    #num_embeddings is the vocab size

	    #num_embeddings, embedding_dim = embed_weights.size()


	    self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
	    #self.embed.load_state_dict({'weight': embed_weights})

	    #if not trainable.
	    #self.emb_layer.weight.requires_grad = False # try this too?

	    self.max_seq_length = max_seq_length

	    self.fc1 = nn.Linear(300, 150)
	    self.fc2 = nn.Linear(6, 20)
	    self.fc3 = nn.Linear(170, 90)
	    self.fc4 = nn.Linear(300, 150) 
	    self.fc5 = nn.Linear(2250+150, embedding_dim) # have to end up with embed_size

	    self.fc6 = nn.Linear(528+512, 128)
	    self.fc7 = nn.Linear(128, 64)
	    self.fc8 = nn.Linear(64,9)
	    self.encoding = nn.LSTM(embedding_dim, 512, num_layers=1)
	    self.linear = nn.Linear(512, num_embeddings)

	def forward(self, grid_embedding, grid_onehot, inventory, goal, instruction, lengths, max_seq_length=20):

	    #encode features
	    c1 = F.relu(self.fc1(grid_embedding))
	    c2 = F.relu(self.fc2(grid_onehot))
	    c1 = c1.view(-1, 25,150)
	    c2 = c2.view(-1, 25,20)
	    combined_grids = torch.cat((c1, c2), dim=2)
	    c3 = F.relu(self.fc3(combined_grids)) 
	    c3 = c3.view(-1, 25*90)
	    c4 = F.relu(self.fc4(goal))
	    combined_grid_goal = torch.cat((c3, c4), dim=1)
	    features = F.relu(self.fc5(combined_grid_goal)) #256

	    #encode caption
	    embeddings = self.embed(instruction)
	    embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
	    packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
	    #output, hdn = self.encoding(packed)
	    hiddens, hdn = self.encoding(packed)
	    outputs = self.linear(hiddens[0])

	    return outputs, hdn[0][0]

	#get new instructions
	def sample(self, grid_embedding, grid_onehot, goal):
		"""Generate captions for given image features using greedy search."""
		sampled_ids = []
		inputs = features.unsqueeze(1)
		for i in range(self.max_seg_length):
			hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
			outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
			_, predicted = outputs.max(1)                        # predicted: (batch_size)
			sampled_ids.append(predicted)
			inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
			inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
		sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
		return sampled_ids

class ActionNet(nn.Module):

	def __init__(self, num_embeddings, embedding_dim, max_seq_length=20):
	    super(ActionNet, self).__init__()

	    #num_embeddings is the vocab size

	    #num_embeddings, embedding_dim = embed_weights.size()


	    self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
	    #self.embed.load_state_dict({'weight': embed_weights})

	    #if not trainable.
	    #self.emb_layer.weight.requires_grad = False # try this too?

	    self.max_seg_length = max_seq_length

	    self.fc1 = nn.Linear(300, 150)
	    self.fc2 = nn.Linear(6, 20)
	    self.fc3 = nn.Linear(170, 90)
	    self.fc4 = nn.Linear(300, 150) 
	    self.fc5 = nn.Linear(2250+150, embedding_dim) # have to end up with embed_size

	    self.fc6 = nn.Linear(256+512, 128)
	    self.fc7 = nn.Linear(128, 64)
	    self.fc8 = nn.Linear(64,9) 
	    self.encoding = nn.LSTM(embedding_dim, 512, num_layers=1)
	    self.linear = nn.Linear(512, num_embeddings)

	def forward(self, grid_embedding, grid_onehot, inventory, goal, instruction_embedding):

	    #encode features
	    c1 = F.relu(self.fc1(grid_embedding))
	    c2 = F.relu(self.fc2(grid_onehot))
	    c1 = c1.view(-1, 25,150)
	    c2 = c2.view(-1, 25,20)
	    combined_grids = torch.cat((c1, c2), dim=2)
	    c3 = F.relu(self.fc3(combined_grids)) 
	    c3 = c3.view(-1, 25*90)
	    c4 = F.relu(self.fc4(goal))
	    combined_grid_goal = torch.cat((c3, c4), dim=1)
	    features = F.relu(self.fc5(combined_grid_goal))

	    combined_instruction = torch.cat((features, instruction_embedding), dim=1)
	    
	    c6 = F.relu(self.fc6(combined_instruction))
	    c7 = F.relu(self.fc7(c6))
	    c8 = F.relu(self.fc8(c7))

	    return c8

class LanguageNetv1(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, max_seq_length=20):
        super(LanguageNetv1, self).__init__()

        #num_embeddings is the vocab size

        #num_embeddings, embedding_dim = embed_weights.size()


        self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
        #self.embed.load_state_dict({'weight': embed_weights})

        #if not trainable.
        #self.emb_layer.weight.requires_grad = False # try this too?

        self.max_seq_length = max_seq_length

        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(7, 20)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(300, 150) 
        self.fc5 = nn.Linear(2250+150, embedding_dim)
        self.fc55 = nn.Linear(512+50, embedding_dim)
        self.fc6 = nn.Linear(528+512, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64,9)
        self.encoding = nn.LSTM(embedding_dim, embedding_dim*2, num_layers=1)
        self.linear = nn.Linear(embedding_dim*2, num_embeddings)

    def forward(self, grid_embedding, grid_onehot, inventory, goal, instruction, lengths, max_seq_length=20):

        #encode features
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        features = c6
        #inventory_emb = F.relu(self.fc_inv(inventory))
        #combined_inventory = torch.cat((c6, inventory_emb), dim=1)
        #features = F.relu(self.fc55(combined_inventory))

        #encode caption
        embeddings = self.embed(instruction)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # try adding these???
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, hdn = self.encoding(packed)
        outputs = self.linear(hiddens[0])

        return outputs, hdn[0][0]

    #get new instructions
    def sample(self, grid_embedding, grid_onehot, inventory, goal, states=None):
        """Generate captions for given image features using greedy search."""
        
        #encode features
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        features = c6

        sampled_ids = []
        inputs = features.unsqueeze(1)

        hdn = None

        for i in range(self.max_seq_length):
            hiddens, states = self.encoding(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
    
            #print(predicted)

            #if predicted == 8: # Check this value
            #    hdn = torch.Tensor(states[0][0])

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)

        if hdn == None:
            hdn = states[0][0]

        return sampled_ids, hdn

#uses weights
class LanguageNetv2(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, embed_weights, max_seq_length=20, training=True):
        super(LanguageNetv2, self).__init__()

        self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
        
        if training:
            self.embed.load_state_dict({'weight': embed_weights})
            self.embed.weight.requires_grad = False

        #if not trainable.
        #self.emb_layer.weight.requires_grad = False # try this too?

        self.max_seq_length = max_seq_length

        self.fc_embed = nn.Linear(embedding_dim, 15)
        self.fc_onehot = nn.Linear(7, 15)
        self.fc_inv = nn.Linear(embedding_dim, 15)
        self.fc_goal = nn.Linear(embedding_dim, 15) 
        self.fc_cat = nn.Linear(25*15+25*15+10*15+15, encoder_dim)

        '''
        self.fc1 = nn.Linear(embedding_dim, 150)
        self.fc2 = nn.Linear(7, 20)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(embedding_dim, 150) 
        self.fc5 = nn.Linear(2250+150, 512)
        self.fc_inv = nn.Linear(embedding_dim, 50) 
        self.fc55 = nn.Linear(512+50, embedding_dim)
        '''

        self.encoding = nn.LSTM(embedding_dim, 32, num_layers=1)
        self.linear = nn.Linear(32, num_embeddings)

    def forward(self, grid_embedding, grid_onehot, inventory, goal, instruction, lengths, max_seq_length=20):

        #encode features
        '''
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        temp_inv = F.relu(self.fc_inv(inventory))
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        features = F.relu(self.fc55(combined_inventory))
        '''
        c1 = F.relu(self.fc_embed(grid_embedding))
        c1 = c1.view(-1, 25*15)
        c2 = F.relu(self.fc_onehot(grid_onehot))
        c2 = c2.view(-1, 25*15)
        c3 = F.relu(self.fc_inv(inventory))
        c3 = c3.view(-1, 10*15)
        c4 = F.relu(self.fc_goal(goal))
        combined = torch.cat((c1,c2,c3,c4), dim=1)
        encoder_out = F.relu(self.fc_cat(combined))

        #encode caption
        embeddings = self.embed(instruction)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # try adding these???
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, hdn = self.encoding(packed)
        outputs = self.linear(hiddens[0])

        return outputs, hdn[0][0]

    #get new instructions
    def sample(self, grid_embedding, grid_onehot, inventory, goal, states=None):
        """Generate captions for given image features using greedy search."""
        
        #encode features
        '''
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        temp_inv = F.relu(self.fc_inv(inventory))
        temp_inv = temp_inv.view(-1, 50)
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        features = F.relu(self.fc55(combined_inventory))
        '''

        c1 = F.relu(self.fc_embed(grid_embedding))
        c1 = c1.view(-1, 25*15)
        c2 = F.relu(self.fc_onehot(grid_onehot))
        c2 = c2.view(-1, 25*15)
        c3 = F.relu(self.fc_inv(inventory))
        c3 = c3.view(-1, 10*15)
        c4 = F.relu(self.fc_goal(goal))
        combined = torch.cat((c1,c2,c3,c4), dim=1)
        encoder_out = F.relu(self.fc_cat(combined))

        sampled_ids = []
        inputs = features.unsqueeze(1)

        hdn = None

        for i in range(self.max_seq_length):
            hiddens, states = self.encoding(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1 )                        # predicted: (batch_size)
    
            #print(predicted)

            #if predicted == 8: # Check this value
            #    hdn = torch.Tensor(states[0][0])

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)

        if hdn == None:
            hdn = states[0][0]

        return sampled_ids, hdn

#uses weights
class LanguageNetv2OLD(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, embed_weights, max_seq_length=20, training=True):
        super(LanguageNetv2OLD, self).__init__()

        self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
        
        if training:
            self.embed.load_state_dict({'weight': embed_weights})
            self.embed.weight.requires_grad = False

        #if not trainable.
        #self.emb_layer.weight.requires_grad = False # try this too?

        self.max_seq_length = max_seq_length

        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(7, 20)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(300, 150) 
        self.fc5 = nn.Linear(2250+150, embedding_dim)
        self.fc55 = nn.Linear(512+50, embedding_dim)
        self.fc6 = nn.Linear(528+512, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64,9)
        self.encoding = nn.LSTM(embedding_dim, 32, num_layers=1)
        self.linear = nn.Linear(32, num_embeddings)

    def forward(self, grid_embedding, grid_onehot, inventory, goal, instruction, lengths, max_seq_length=20):

        #encode features
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        features = c6
        #inventory_emb = F.relu(self.fc_inv(inventory))
        #combined_inventory = torch.cat((c6, inventory_emb), dim=1)
        #features = F.relu(self.fc55(combined_inventory))

        #encode caption
        embeddings = self.embed(instruction)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # try adding these???
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, hdn = self.encoding(packed)
        outputs = self.linear(hiddens[0])

        return outputs, hdn[0][0]

    #get new instructions
    def sample(self, grid_embedding, grid_onehot, inventory, goal, states=None):
        """Generate captions for given image features using greedy search."""
        
        #encode features
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        features = c6

        sampled_ids = []
        inputs = features.unsqueeze(1)

        hdn = None

        for i in range(self.max_seq_length):
            hiddens, states = self.encoding(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1 )                        # predicted: (batch_size)
    
            #print(predicted)

            #if predicted == 8: # Check this value
            #    hdn = torch.Tensor(states[0][0])

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)

        if hdn == None:
            hdn = states[0][0]

        return sampled_ids, hdn


class ActionNetv1(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, max_seq_length=20):
        super(ActionNetv1, self).__init__()

        #num_embeddings is the vocab size

        #num_embeddings, embedding_dim = embed_weights.size()


        self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
        #self.embed.load_state_dict({'weight': embed_weights})

        #if not trainable.
        #self.emb_layer.weight.requires_grad = False # try this too?

        self.max_seg_length = max_seq_length

        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(6, 20)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(300, 150) 
        self.fc5 = nn.Linear(2250+150, 512) # have to end up with embed_size
        self.fc55 = nn.Linear(512+300, embedding_dim)
        
        # self.fc6 = nn.Linear(256+512, 128)
        # self.fc7 = nn.Linear(128, 64)
        # self.fc8 = nn.Linear(64,9) 

        self.fc6 = nn.Linear(48, 64)
        self.fc7 = nn.Linear(48, 9)
        self.fc8 = nn.Linear(64,9) 

        self.encoding = nn.LSTM(embedding_dim, 512, num_layers=1)
        self.linear = nn.Linear(512, num_embeddings)

    def forward(self, grid_embedding, grid_onehot, inventory, goal, instruction_embedding):

        #encode features
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,20)
        combined_grids = torch.cat((c1, c2), dim=2)
        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        combined_inventory = torch.cat((c6, inventory), dim=1)
        features = F.relu(self.fc55(combined_inventory))

        combined_instruction = torch.cat((features, instruction_embedding), dim=1)
        
        c6 = F.relu(self.fc6(combined_instruction)) # updated with new embedding size.
        c7 = F.relu(self.fc7(c6))
        #c8 = F.relu(self.fc8(c7))

        return c7

## Version 3: uses adding instead of concat
class LanguageNetv3(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, max_seq_length=20):
        super(LanguageNetv3, self).__init__()

        #num_embeddings is the vocab size

        #num_embeddings, embedding_dim = embed_weights.size()


        self.embed = nn.Embedding(num_embeddings, embedding_dim) # vocab size, 300
        #self.embed.load_state_dict({'weight': embed_weights})

        #if not trainable.
        #self.emb_layer.weight.requires_grad = False # try this too?

        self.max_seq_length = max_seq_length

        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(6, 150)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(300, 150) 
        self.fc5 = nn.Linear(2250+150, embedding_dim)
        #self.fc5 = nn.Linear(2250+150, 512) 
        #self.fc55 = nn.Linear(512+300, embedding_dim)
        self.fc6 = nn.Linear(528+512, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64,9)
        self.encoding = nn.LSTM(embedding_dim, 512, num_layers=1)
        self.linear = nn.Linear(512, num_embeddings)

    def forward(self, grid_embedding, grid_onehot, inventory, goal, instruction, lengths, max_seq_length=20):

        #encode features
        c1 = F.relu(self.fc1(grid_embedding))
        c2 = F.relu(self.fc2(grid_onehot))
        c1 = c1.view(-1, 25,150)
        c2 = c2.view(-1, 25,150)
        #combined_grids = torch.cat((c1, c2), dim=2)
        combined_grids = c1 + c2

        c3 = F.relu(self.fc3(combined_grids)) 
        c3 = c3.view(-1, 25*90)
        c4 = F.relu(self.fc4(goal))
        combined_grid_goal = torch.cat((c3, c4), dim=1)
        c6 = F.relu(self.fc5(combined_grid_goal))
        features = c6
        #combined_inventory = torch.cat((c6, inventory), dim=1)
        #features = F.relu(self.fc55(combined_inventory))

        #encode caption
        embeddings = self.embed(instruction)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        #output, hdn = self.encoding(packed)
        hiddens, hdn = self.encoding(packed)
        outputs = self.linear(hiddens[0])

        return outputs, hdn[0][0]

    #get new instructions
    def sample(self, grid_embedding, grid_onehot, goal):
        ## SWAP THIS OUT!!!

        sampled_ids = []
        inputs = features.unsqueeze(1)

        hdn = None

        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            
            if predicted == 8: # Check this value
                hdn = torch.Tensor(states[0][0])

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)

        if hdn == None:
            hdn = states[0][0]

        return sampled_ids, hdn


## Action net using all obs.. 
class ActionNet_Allobs(nn.Module):

    def __init__(self, embed_dim, max_seq_length=20):
        super(ActionNet_Allobs, self).__init__()

        self.embed_dim = embed_dim

        self.hid_sz = 32

        ### INPUTS:
        self.fc_gridembed = nn.Linear(25*embed_dim, 32)
        self.fc_onehot = nn.Linear(25*6, 32)
        self.fc_inv = nn.Linear(embed_dim, 32)
        self.fc_goal = nn.Linear(embed_dim, 32) 

        self.fc1 = nn.Linear(32, 32) # not using this right now.

        net_input_sizes = []
        for i in range(5): # now 5. 
            net_input_sizes.append(self.hid_sz)

        self.batch_modules = SwitchModule(net_input_sizes, len(net_input_sizes))

        self.fc_final = nn.Linear(32, 9)

    def forward(self, grid_embedding, grid_onehot, inventory, goal, instruction_embedding):

        grid_embedding = grid_embedding.view(-1, 25*self.embed_dim)
        grid_embed = F.relu(self.fc_gridembed(grid_embedding))
        grid_onehot = grid_onehot.view(-1, 25*6)
        grid_1hot_embed = F.relu(self.fc_onehot(grid_onehot))
        inv_embed = F.relu(self.fc_inv(inventory))
        goal_embed = F.relu(self.fc_goal(goal))

        net_inputs = ()
        net_inputs += (grid_embed,)
        net_inputs += (grid_1hot_embed,)
        net_inputs += (inv_embed,)
        net_inputs += (goal_embed,)
        net_inputs += (instruction_embedding,)

        key = []
        key.append(grid_embed)
        key.append(grid_1hot_embed)
        key.append(inv_embed)
        key.append(goal_embed)
        key.append(instruction_embedding)
        key = torch.cat(key, 1)

        #batch modules

        output = self.batch_modules((net_inputs, key))

        output = F.relu(self.fc_final(output))

        return output

'''
#Based off of Kenny's code
class AllObsPredict(nn.Module):

    def __init__(self, embed_dim):
        super(AllObsPredict, self).__init__()

        self.embed_dim = embed_dim

        self.hid_sz = 32

        ### INPUTS:
        self.fc_gridembed = nn.Linear(25*embed_dim, 32)
        self.fc_onehot = nn.Linear(25*7, 32)
        self.fc_inv = nn.Linear(embed_dim, 32)
        self.fc_goal = nn.Linear(embed_dim, 32) 
        
        self.fc1 = nn.Linear(32, 32) # not using this right now.

        net_input_sizes = []
        for i in range(4):
            net_input_sizes.append(self.hid_sz)

        self.batch_modules = SwitchModule(net_input_sizes, len(net_input_sizes))

        self.fc_final = nn.Linear(32, 8)

    def forward(self, grid_embedding, grid_onehot, inventory, goal):

        grid_embedding = grid_embedding.view(-1, 25*self.embed_dim)
        grid_embed = F.relu(self.fc_gridembed(grid_embedding))
        grid_onehot = grid_onehot.view(-1, 25*7)
        grid_1hot_embed = F.relu(self.fc_onehot(grid_onehot))
        inv_embed = F.relu(self.fc_inv(inventory))
        goal_embed = F.relu(self.fc_goal(goal))

        net_inputs = ()
        net_inputs += (grid_embed,)
        net_inputs += (grid_1hot_embed,)
        net_inputs += (inv_embed,)
        net_inputs += (goal_embed,)

        key = []
        key.append(grid_embed)
        key.append(grid_1hot_embed)
        key.append(inv_embed)
        key.append(goal_embed)
        key = torch.cat(key, 1)

        #batch modules

        output = self.batch_modules((net_inputs, key))

        output = F.relu(self.fc_final(output))

        return output

class SwitchModule(nn.Module):
    def __init__(self, input_sz, num_inputs):
        super(SwitchModule, self).__init__()

        # Get parameters
        self.input_sz = input_sz
        self.num_inputs = num_inputs
        self.num_modules = 2*num_inputs
        self.hid_sz = 32
        num_layer = 3

        # Make batch modules
        self.batch_modules = []
        for module_in_size in self.input_sz:
            bm_input = BatchMLP(module_in_size, self.hid_sz, num_layer, self.num_modules//num_inputs)
            self.batch_modules.append(bm_input)
        self.batch_modules = ListModule(*self.batch_modules)

        # Make soft attention network components (if applicible)
        self.switch_sz = 32*num_inputs # change this if necessary?
        self.att_in = nn.Linear(self.switch_sz, self.num_modules)
        self.softmax = nn.Softmax(dim=1)

    # Forward (mainly switch between soft and hard modes)
    def forward(self, inputs):
        net_inputs = inputs[0]

        # Compute batch module output
        #assert(len(net_inputs) == self.num_inputs)
        all_module_outs = []
        for i, net_input in enumerate(net_inputs):
            batch_inputs = net_input.unsqueeze(1).expand([-1, self.num_modules//self.num_inputs, -1])
            #print(batch_inputs.size())
            module_outs = self.batch_modules[i](batch_inputs) # module_outs is bs x nm//ni x out_sz
            all_module_outs.append(module_outs)
        module_outs = torch.cat(all_module_outs, 1)

        # Soft attention on output
        switch_input = inputs[1]
        selection = self.softmax(self.att_in(switch_input))
        selection = selection.unsqueeze(2)
        selection = selection.repeat([1, 1, module_outs.size(2)])
        module_outs *= selection
        module_outs = module_outs.sum(1)

        return module_outs


# Batch MLP module
# Basically does a for loop over MLPs, but does this efficiently using bmm
class BatchMLP(nn.Module):
    def __init__(self, input_sz, hid_sz, num_layer, num_modules):
        super(BatchMLP, self).__init__()

        # Fix the init! Shouldn't be done here

        # init_ = lambda m: init(m,
        #       init_normc_,
        #       lambda x: nn.init.constant_(x, 0))

        # Make network values
        self.tanh = nn.Tanh()
        self.in_fc = BatchLinear(input_sz, hid_sz, num_modules)
        assert(num_layer >= 2) # If num_layer is 2, actually no hidden layers technically
        hid_layers = []
        for i in range(0, num_layer-2):
            hid_fc = BatchLinear(hid_sz, hid_sz, num_modules)
            hid_layers.append(hid_fc)
        self.hid_layers = ListModule(*hid_layers)
        self.out_fc = BatchLinear(hid_sz, hid_sz, num_modules)

    # Input batch_size x num_modules x input_sz
    # Output batch_size x num_modules x output_sz
    def forward(self, input):
        x = self.in_fc(input)
        x = self.tanh(x)
        for hid_fc in self.hid_layers:
            x = hid_fc(x)
            x = self.tanh(x)
        x = self.out_fc(x)
        return x

# BatchLinear module
# Same as nn.Linear, but it takes list of inputs x and outputs list of outputs y
# Equivalent to same operation if we did a for loop and did nn.Linear for each
class BatchLinear(nn.Module):
    def __init__(self, in_features, out_features, num_modules, bias=True):
        super(BatchLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modules = num_modules
        self.weight = nn.Parameter(torch.Tensor(num_modules, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_modules, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # Input batch_size x num_module x input_sz
    # Output batch_size x num_module x output_sz
    def forward(self, input):
        # Get sizes
        bs = input.size(0)
        nm = input.size(1)
        assert(input.size(2) == self.in_features)

        # Transpose input to correct shape
        input = input.transpose(0, 1).transpose(1, 2) # nm x in_sz x bs

        # Compute matrix multiply
        output = torch.bmm(self.weight, input)

        # Add bias
        if self.bias is not None:
            output += self.bias.unsqueeze(2).expand([-1, -1, bs]).contiguous()

        # Transpose back to bs x nm x out_sz
        output = output.transpose(1, 2).transpose(0, 1)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# Use ListModule code from fmassa (shouldn't this be in pytorch already?)
class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
'''

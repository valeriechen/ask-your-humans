import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from pytorchppo.a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from pytorchppo.a2c_ppo_acktr.utils import init

import nltk
import pickle
import argparse
from collections import Counter
import torchtext.vocab as vocabtorch

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = MazeBaseBase
            #if len(obs_shape) == 3:
            #    base = CNNBase
            #elif len(obs_shape) == 1:
            #    base = MLPBase
            #else:
            #    raise NotImplementedError
        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class MazeBaseNet(nn.Module):
    def __init__(self, hidden_size):
        super(MazeBaseNet, self).__init__()

        self.embed_dim = 300
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.embed_dim, 150)
        self.fc2 = nn.Linear(7, 20)
        self.fc3 = nn.Linear(170, 90)
        self.fc4 = nn.Linear(self.embed_dim, 150) 
        self.fc5 = nn.Linear(2250+150, 512)
        self.fc_inv = nn.Linear(self.embed_dim, 50) 
        self.fc55 = nn.Linear(512+500, 128)
        self.fc6 = nn.Linear(128, hidden_size)

    def forward(self, x):

        grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        inventory = x[:, (5*5*300)+(5*5*7)+300:]#.reshape((x.shape[0], 10, 300))

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
        #temp_inv = temp_inv.view(-1, 50*10)
        combined_inventory = torch.cat((c6, temp_inv), dim=1)
        features = F.relu(self.fc55(combined_inventory))
        
        c6 = F.relu(self.fc6(features))

        return c6

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

class AllObsPredictAtten(nn.Module):
    def __init__(self, embed_dim, vocab_words, vocab_weights, with_pred=False, init_fn=None, opt=None, env_vocab=None, max_bounds=None, num_stack=1, add_net_pred=False, pred_size=None, **kwargs):
        super(AllObsPredictAtten, self).__init__()

        #self.lang_model = lang_model

        # Set state vars
        self.hid_sz = 32
        self.use_dropout = False
        self.embed_dim = 300
        embed_dim = 300

        self.vocab = vocab_words

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

        self.fc_embed = nn.Linear(300, 32)
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
    def forward(self, x, hiddens):

        grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        inventory = x[:, (5*5*300)+(5*5*7)+300:].reshape((x.shape[0], 10, 300))

        # seqs, hiddens = lang_model.get_hidden_state_new(grid_embedding, grid_onehot, inventory, goal, self.vocab)         

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

        # Get key and net inputs
        key = []
        if 'wids' in self.key_inputs:
            key.append(hiddens)
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
        #if self.use_dropout:
        x = self.drop(x)


        return x



class MazeBaseNetLang(nn.Module):
    def __init__(self, hidden_size, lang_model, vocab_words, vocab_weights):
        super(MazeBaseNetLang, self).__init__()

        self.lang_model = lang_model

        # Set state vars
        self.embed_dim = 300
        self.hid_sz = hidden_size
        self.use_dropout = False

        #self.seq2vec = BOW(vocab_words, vocab_weights, 300)
        self.embedding = nn.Embedding(num_embeddings=len(vocab_words)+1,
                                          embedding_dim=self.embed_dim, padding_idx=len(vocab_words))
        self.embedding.load_state_dict({'weight': vocab_weights})

        self.vocab = vocab_words
        
        #fc layers
        self.fc_gridembed = nn.Linear(25*self.embed_dim, 32)
        self.fc_onehot = nn.Linear(25*7, 32)
        self.fc_inv = nn.Linear(10*self.embed_dim, 32) # 10 inventory slots
        self.fc_goal = nn.Linear(self.embed_dim, 32) 

        self.fc_embed = nn.Linear(300, 32)
        self.fc_onehot_embed = nn.Linear(7, 32)
        self.fc_comb = nn.Linear(25*32, 32)
        self.fc_inv_first = nn.Linear(300, 32)
        self.fc_inv_second = nn.Linear(32*10, 32)

        # Decide what inputs go in net and key

        # key is for attention
        key_sz = 300
        self.key_inputs = ['wids']

        # net is for regular input
        self.net_inputs = ['grid_embed', 'grid_onehot', 'inv', 'goal']
        #self.net_inputs = ['grid_comb', 'inv', 'goal']
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
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        #grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        #grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        #goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        #inventory = x[:, (5*5*300)+(5*5*7)+300:].reshape((x.shape[0], 10, 300))

        k_prev = x[:, 0]
        top_k = x[:,1]

        grid_embedding = x[:,2:2+(5*5*300)].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 2+(5*5*300):2+(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, 2+(5*5*300)+(5*5*7) : 2+(5*5*300)+(5*5*7)+300]
        inventory = x[:, 2+(5*5*300)+(5*5*7)+300:].reshape((x.shape[0], 10, 300))

        #get language here!
        all_sampled_ids = self.lang_model.get_hidden_state(grid_embedding, grid_onehot, inventory, goal, k_prev, top_k, self.vocab)
        bow_ids = [np.array(sent + [len(self.vocab)] * (20 - len(sent))) for sent in all_sampled_ids]
        bow_ids = np.asarray(bow_ids, dtype=int)                                                   
        bow_ids = torch.from_numpy(bow_ids).cuda()     

        # Encode sequence
        output = self.embedding(bow_ids)
        statement = output.sum(1) 
        #statement = self.seq2vec(bow_ids)

        grid_embedding = grid_embedding.view(-1, 25*self.embed_dim)
        grid_embed = F.relu(self.fc_gridembed(grid_embedding))
        grid_onehot = grid_onehot.view(-1, 25*7)
        grid_1hot_embed = F.relu(self.fc_onehot(grid_onehot))
        inventory = inventory.view(-1, 10*self.embed_dim)
        inv_embed = F.relu(self.fc_inv(inventory))
        goal_embed = F.relu(self.fc_goal(goal))


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

        return x

# This module implements a soft network switch
# It takes a set pf network inputs and a soft selection input
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

        # Reshape to proper matrices
        #if self.W is None or self.W.size(0) != bs:
        #    self.W = self.weight.unsqueeze(0).expand([bs, -1, -1, -1]).contiguous().view(nm*bs, self.out_features, self.in_features)
        #input = input.contiguous().view(nm*bs, self.in_features, 1)

        # Compute matrix multiply and add bias (if applicable)
        #output = torch.bmm(self.W, input)

        # Add the bias
        #if self.bias is not None:
        #    if self.b is None or self.b.size(0) != bs:
        #        self.b = self.bias.unsqueeze(0).expand([bs, -1, -1]).contiguous().view(nm*bs, self.out_features, 1)
        #    output += self.b

        # Reshape output
        #output = output.view(bs, nm, self.out_features)

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

    def forward(self, grid_embedding, grid_onehot, inventory, goal, encoded_captions, caption_lengths, max_seq_length=20):

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
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda()
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).cuda()
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

    #def get_hidden_state_new(self, grid_embedding, grid_onehot, inventory, goal, word_map, states=None):
    def get_hidden_state_new(self, x, word_map, states=None):

        grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        inventory = x[:, (5*5*300)+(5*5*7)+300:].reshape((x.shape[0], 10, 300))

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
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).cuda() #.to(device)  # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).cuda() #.to(device)  # (k, 1)

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

    #NEW VERSION!
    def get_hidden_state(self, grid_embedding, grid_onehot, inventory, goal, k_prev, top_k, word_map, states=None):
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
        k_prev_words = k_prev.long().unsqueeze(1)#torch.LongTensor(k_prev)#.to(device)  # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = top_k.unsqueeze(1)#torch.zeros(k, 1) #.to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = [[word_map.word2idx['<start>']] for i in range(k)]
        incomplete_inds = [i for i in range(k)] # used to keep track of original index in complete_seqs
        
        #complete_seqs_scores = list()

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

            values, indices = scores.max(dim=1) 

            assert(indices.size(0) == len(incomplete_inds))

            temp = []
            for i in range(indices.size(0)-1, -1, -1):
                complete_seqs[incomplete_inds[i]].append(indices.data[i])
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

            #incomplete_inds = [i for i in range(indices.size(0)) if indices[i] != word_map.word2idx['<end>']]
            #complete_inds = [i for i in range(indices.size(0)) if indices[i] == word_map.word2idx['<end>']]

            # Add
            #scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            #if step == 1:
            #    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            #else:
            #    # Unroll and find top scores, and their unrolled indices
            #    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            #prev_word_inds = top_k_words / self.vocab_size  # (s)
            #next_word_inds = top_k_words % self.vocab_size  # (s)

            # Add new words to sequences, alphas
            #seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            #incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
            #                   next_word != word_map.word2idx['<end>']]
            #complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            #if len(complete_inds) > 0:
            #    complete_seqs.extend(seqs[complete_inds].tolist())
            #    complete_seqs_scores.extend(top_k_scores[complete_inds])
            #k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            #if k == 0:
            #    break

            #seqs = seqs[incomplete_inds]

            #subset things out...
            #h = h[prev_word_inds[incomplete_inds]]
            #c = c[prev_word_inds[incomplete_inds]]
            #encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            #top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            #k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        return complete_seqs

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


    def get_hidden_state_new(self, x, word_map, states=None):

        grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        inventory = x[:, (5*5*300)+(5*5*7)+300:].reshape((x.shape[0], 1, 300))

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
        k_prev_words = torch.LongTensor([[word_map.word2idx['<start>']]] * k).cuda() # (k, 1)
        
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).cuda()  # (k, 1)

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

class StatePredictorNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_dim, embed_weights, max_seq_length=20, training=True):
        super(StatePredictorNetwork, self).__init__()

        encoder_dim = 128
        self.encoder_dim = encoder_dim
        attention_dim = encoder_dim
        embed_dim = embedding_dim
        decoder_dim = 32

        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size

        self.fc_embed = nn.Linear(embedding_dim, encoder_dim)
        self.fc_onehot = nn.Linear(7, encoder_dim)
        self.fc_inv = nn.Linear(embedding_dim, encoder_dim)
        self.fc_goal = nn.Linear(embedding_dim, encoder_dim) 
    
        self.dropout = 0.5

        # new stuff...
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size+1, embed_dim, vocab_size)  # embedding layer

        if training:
            self.embedding.load_state_dict({'weight': embed_weights})
            self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, embed_dim)  # linear layer to find scores over vocabulary
        self.fc_last = nn.Linear(27*128, 300)


    def get_state_encoding(self, grid_embedding, grid_onehot, inventory, goal):

        #print(grid_embedding.size(), grid_onehot.size(), inventory.size(), goal.size())

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,self.encoder_dim)
        inventory = inventory.view(-1, 1, 300)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.unsqueeze(1)

        encoder_out = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)
        encoder_out = encoder_out.view(-1, 27*128)
        encoder_out = F.relu(self.fc_last(encoder_out))

        return encoder_out

    def forward(self, x):

        batch_size = x.size(0)

        # Initialize to zero
        h = torch.zeros(batch_size, 32).cuda()  # (batch_size, decoder_dim) # 
        c = torch.zeros(batch_size, 32).cuda()

        # Create tensors to hold state prediction
        hiddens = h.clone()

        for t in range(3):

            x = x[:,1:]

            grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
            grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
            goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
            inventory = x[:, (5*5*300)+(5*5*7)+300:(5*5*300)+(5*5*7)+300+300].reshape((x.shape[0], 1, 300))

            #print(grid_embedding.size(), grid_onehot.size(), goal.size(), inventory.size())

            state_encodings = self.get_state_encoding(grid_embedding, grid_onehot, inventory, goal)
            
            h,c = self.decode_step(state_encodings, (h[:], c[:]))
            hiddens[:] = h.clone()
            preds = self.fc(h)

            x = x[:, (5*5*300)+(5*5*7)+300+300:]

        return preds, hiddens


class SimpleNetworkNoState(nn.Module):
  def __init__(self, embed_dim):
    super(SimpleNetworkNoState, self).__init__()

    self.embed_dim = embed_dim

    self.fc1 = nn.Linear(embed_dim, 150)
    self.fc2 = nn.Linear(7, 20)
    self.fc3 = nn.Linear(170, 90)
    self.fc4 = nn.Linear(embed_dim, 150) 
    self.fc5 = nn.Linear(2250+150, 512)
    self.fc_inv = nn.Linear(embed_dim, 50) 
    self.fc55 = nn.Linear(512+50, 48)

    self.fc6 = nn.Linear(32, 48)
    self.fc7 = nn.Linear(48, 8)


  def forward(self, x, hidden):

        grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        inventory = x[:, (5*5*300)+(5*5*7)+300:]#.reshape((x.shape[0], 10, 300))

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
        
        #c6 = F.relu(self.fc6(all_comb)) # updated with new embedding size.
        #c7 = self.fc7(c6)
        #c8 = F.relu(self.fc8(c7))

        c6 = F.relu(self.fc6(hidden)) # updated with new embedding size.

        return c6

class SimpleNetworkOnlyGoal(nn.Module):
  def __init__(self, embed_dim):
    super(SimpleNetworkOnlyGoal, self).__init__()

    self.embed_dim = embed_dim

    self.fc1 = nn.Linear(embed_dim, 150)
    self.fc2 = nn.Linear(7, 20)
    self.fc3 = nn.Linear(170, 90)
    self.fc4 = nn.Linear(embed_dim, 150) 
    self.fc5 = nn.Linear(2250+150, 512)
    self.fc_inv = nn.Linear(embed_dim, 50) 
    self.fc55 = nn.Linear(512+50, 48)

    self.fc6 = nn.Linear(embed_dim, 48)
    self.fc7 = nn.Linear(48, 8)


  def forward(self, x):

        grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        inventory = x[:, (5*5*300)+(5*5*7)+300:]#.reshape((x.shape[0], 10, 300))

        #encode features
        # c1 = F.relu(self.fc1(grid_embedding))
        # c2 = F.relu(self.fc2(grid_onehot))
        # c1 = c1.view(-1, 25,150)
        # c2 = c2.view(-1, 25,20)
        # combined_grids = torch.cat((c1, c2), dim=2)
        # c3 = F.relu(self.fc3(combined_grids)) 
        # c3 = c3.view(-1, 25*90)
        # c4 = F.relu(self.fc4(goal))
        # combined_grid_goal = torch.cat((c3, c4), dim=1)
        # c6 = F.relu(self.fc5(combined_grid_goal))
        # temp_inv = F.relu(self.fc_inv(inventory))
        # combined_inventory = torch.cat((c6, temp_inv), dim=1)
        # features = F.relu(self.fc55(combined_inventory))

        # all_comb = torch.cat((features, hidden), dim=1)
        
        c6 = F.relu(self.fc6(goal)) # updated with new embedding size.
        #c6 = F.relu(self.fc6(all_comb)) # updated with new embedding size.
        #c7 = self.fc7(c6)
        #c8 = F.relu(self.fc8(c7))

        return c6

class StateAutoencoder(nn.Module):
    def __init__(self):
        super(StateAutoencoder, self).__init__()

        self.fc1 = nn.Linear(128, 27)
        self.fc2 = nn.Linear(27*27, 32)
        self.fc3 = nn.Linear(32, 27*27)
        self.fc4 = nn.Linear(27, 128)

        embedding_dim = 300
        encoder_dim = 128

        self.fc_embed = nn.Linear(embedding_dim, encoder_dim)
        self.fc_onehot = nn.Linear(7, encoder_dim)
        self.fc_inv = nn.Linear(embedding_dim, encoder_dim)
        self.fc_goal = nn.Linear(embedding_dim, encoder_dim) 

    def forward(self, x):

        grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        inventory = x[:, (5*5*300)+(5*5*7)+300:]#.reshape((x.shape[0], 10, 300))

        c1 = F.relu(self.fc_embed(grid_embedding))
        c2 = F.relu(self.fc_onehot(grid_onehot))
        grid_comb = c1 + c2 
        grid_comb = grid_comb.view(-1,25,128)
        inventory = inventory.view(-1, 1, 300)
        c3 = F.relu(self.fc_inv(inventory)) # maybe change inventory back to sum later..
        c4 = F.relu(self.fc_goal(goal))
        c4 = c4.unsqueeze(1)
        state_encoding = torch.cat((grid_comb, c3, c4), dim=1) # (batch, 25+10+1, encoder_dim)

        # state_encoding is (batch, 25+1+1, encoder_dim)

        c1 = F.relu(self.fc1(state_encoding))
        c1 = c1.view(-1, 27*27)
        c2 = F.relu(self.fc2(c1))
        c3 = F.relu(self.fc3(c2))
        c3 = c3.view(-1, 27, 27)
        c4 = F.relu(self.fc4(c3))

        return state_encoding, c2, c4

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


  def forward(self, x, hidden):

        x = x[:, -8275:]

        grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        inventory = x[:, (5*5*300)+(5*5*7)+300:]#.reshape((x.shape[0], 10, 300))

        #print(grid_embedding.size(), grid_onehot.size(), goal.size(), inventory.size())

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
        #c7 = self.fc7(c6)
        #c8 = F.relu(self.fc8(c7))

        return c6

def build_vocabulary(train_instructions, save_name, embed_dim):

    freqs = {}

    if embed_dim == 300:
        glove = vocabtorch.GloVe(name='840B', dim=300) #maybe switch this out!
    elif embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)

    for instruction in train_instructions:

        for word in instruction:

            try:
                vec = glove.vectors[glove.stoi[word]]
                if word in freqs:
                    freqs[word] = freqs[word] + 1
                else:
                    freqs[word] = 1
            except:
                if 'UNK' in freqs:
                    freqs['UNK'] = freqs['UNK'] + 1
                else:
                    freqs['UNK'] = 1

    vocab_size = 0

    for i, key in enumerate(freqs):
        if 'UNK' != key and freqs[key] > 10:
            vocab_size = vocab_size + 1


    vocab_weights = np.random.uniform(-0.01, 0.01, (vocab_size+4, embed_dim)).astype("float32")
    #vocab_weights = np.zeros((vocab_size+4, embed_dim), dtype=np.float32)
    vocab = Vocabulary()

    count = 0
    for i, key in enumerate(freqs):

        ## ENFORCE THAT IF < THRESHOLD, DON'T INCLUDE!! 
        # NOT USING i  and key correctly... ?? 

        if 'UNK' == key:
            vec_string = '0.22418134 -0.28881392 0.13854356 0.00365387 -0.12870757 0.10243822 0.061626635 0.07318011 -0.061350107 -1.3477012 0.42037755 -0.063593924 -0.09683349 0.18086134 0.23704372 0.014126852 0.170096 -1.1491593 0.31497982 0.06622181 0.024687296 0.076693475 0.13851812 0.021302193 -0.06640582 -0.010336159 0.13523154 -0.042144544 -0.11938788 0.006948221 0.13333307 -0.18276379 0.052385733 0.008943111 -0.23957317 0.08500333 -0.006894406 0.0015864656 0.063391194 0.19177166 -0.13113557 -0.11295479 -0.14276934 0.03413971 -0.034278486 -0.051366422 0.18891625 -0.16673574 -0.057783455 0.036823478 0.08078679 0.022949161 0.033298038 0.011784158 0.05643189 -0.042776518 0.011959623 0.011552498 -0.0007971594 0.11300405 -0.031369694 -0.0061559738 -0.009043574 -0.415336 -0.18870236 0.13708843 0.005911723 -0.113035575 -0.030096142 -0.23908928 -0.05354085 -0.044904727 -0.20228513 0.0065645403 -0.09578946 -0.07391877 -0.06487607 0.111740574 -0.048649278 -0.16565254 -0.052037314 -0.078968436 0.13684988 0.0757494 -0.006275573 0.28693774 0.52017444 -0.0877165 -0.33010918 -0.1359622 0.114895485 -0.09744406 0.06269521 0.12118575 -0.08026362 0.35256687 -0.060017522 -0.04889904 -0.06828978 0.088740796 0.003964443 -0.0766291 0.1263925 0.07809314 -0.023164088 -0.5680669 -0.037892066 -0.1350967 -0.11351585 -0.111434504 -0.0905027 0.25174105 -0.14841858 0.034635577 -0.07334565 0.06320108 -0.038343467 -0.05413284 0.042197507 -0.090380974 -0.070528865 -0.009174437 0.009069661 0.1405178 0.02958134 -0.036431845 -0.08625681 0.042951006 0.08230793 0.0903314 -0.12279937 -0.013899368 0.048119213 0.08678239 -0.14450377 -0.04424887 0.018319942 0.015026873 -0.100526 0.06021201 0.74059093 -0.0016333034 -0.24960588 -0.023739101 0.016396184 0.11928964 0.13950661 -0.031624354 -0.01645025 0.14079992 -0.0002824564 -0.08052984 -0.0021310581 -0.025350995 0.086938225 0.14308536 0.17146006 -0.13943303 0.048792403 0.09274929 -0.053167373 0.031103406 0.012354865 0.21057427 0.32618305 0.18015954 -0.15881181 0.15322933 -0.22558987 -0.04200665 0.0084689725 0.038156632 0.15188617 0.13274793 0.113756925 -0.095273495 -0.049490947 -0.10265804 -0.27064866 -0.034567792 -0.018810693 -0.0010360252 0.10340131 0.13883452 0.21131058 -0.01981019 0.1833468 -0.10751636 -0.03128868 0.02518242 0.23232952 0.042052146 0.11731903 -0.15506615 0.0063580726 -0.15429358 0.1511722 0.12745973 0.2576985 -0.25486213 -0.0709463 0.17983761 0.054027 -0.09884228 -0.24595179 -0.093028545 -0.028203879 0.094398156 0.09233813 0.029291354 0.13110267 0.15682974 -0.016919162 0.23927948 -0.1343307 -0.22422817 0.14634751 -0.064993896 0.4703685 -0.027190214 0.06224946 -0.091360025 0.21490277 -0.19562101 -0.10032754 -0.09056772 -0.06203493 -0.18876675 -0.10963594 -0.27734384 0.12616494 -0.02217992 -0.16058226 -0.080475815 0.026953284 0.110732645 0.014894041 0.09416802 0.14299914 -0.1594008 -0.066080004 -0.007995227 -0.11668856 -0.13081996 -0.09237365 0.14741232 0.09180138 0.081735 0.3211204 -0.0036552632 -0.047030564 -0.02311798 0.048961394 0.08669574 -0.06766279 -0.50028914 -0.048515294 0.14144728 -0.032994404 -0.11954345 -0.14929578 -0.2388355 -0.019883996 -0.15917352 -0.052084364 0.2801028 -0.0029121689 -0.054581646 -0.47385484 0.17112483 -0.12066923 -0.042173345 0.1395337 0.26115036 0.012869649 0.009291686 -0.0026459037 -0.075331464 0.017840583 -0.26869613 -0.21820338 -0.17084768 -0.1022808 -0.055290595 0.13513643 0.12362477 -0.10980586 0.13980341 -0.20233242 0.08813751 0.3849736 -0.10653763 -0.06199595 0.028849555 0.03230154 0.023856193 0.069950655 0.19310954 -0.077677034 -0.144811'
            average_glove_vector = np.array(vec_string.split(" "))

        else:
            if freqs[key] > 10:
                vocab_weights[count] = glove.vectors[glove.stoi[key]]
                count = count + 1
                vocab.add_word(key)

    print("Total vocabulary size: {}".format(len(vocab)))

    return vocab, vocab_weights

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


  def forward(self, x):

        grid_embedding = x[:,:5*5*300].reshape((x.shape[0], 5,5,300))
        grid_onehot = x[:, 5*5*300:(5*5*300)+(5*5*7)].reshape((x.shape[0], 5,5,7))
        goal = x[:, (5*5*300)+(5*5*7) : (5*5*300)+(5*5*7)+300]
        inventory = x[:, (5*5*300)+(5*5*7)+300:]#.reshape((x.shape[0], 10, 300))

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
        #c7 = self.fc7(c6)
        #c8 = F.relu(self.fc8(c7))

        return c6

class MazeBaseBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=48, vocab=None, vocab_weights=None):
        super(MazeBaseBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        #no language:
        #self.actor = MazeBaseNet(hidden_size)
        #self.critic = MazeBaseNet(hidden_size)


        #pre-trained model:
        #hidden size 48, summed inventory

        # self.actor = SimpleNetworkOnlyGoal(300)
        # self.actor.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/AllObsPredictAttenGLOVE_onlygoal.pt")) # trained with embeddings

        # self.critic = SimpleNetworkOnlyGoal(300)
        # self.critic.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/AllObsPredictAttenGLOVE_onlygoal.pt")) # trained with embeddings


        #self.actor = StateGoalNetv1(300)
        #self.actor.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/StateGoalNetv1_300_05per.pt")) # trained with embeddings

        #self.critic = StateGoalNetv1(300)
        #self.critic.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/StateGoalNetv1_300_05per.pt")) # trained with embeddings

        ### Start State Predictor ###

        name = "compiled_dataset_08131950" #add 50 back in
        embed_dim = 300 # switch this later!!
        embed_size = embed_dim

        with open('/home/kdmarino/Mazebase_gym_environ/mazebaseenv/data/'+name+'_all_instructions', 'rb') as f:
            all_instructions = pickle.load(f)

        vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        self.vocab = vocab
        temp = np.zeros((1,300), dtype=np.float32)
        vocab_weights = np.concatenate((vocab_weights, temp), axis=0)
        vocab_weights = torch.Tensor(vocab_weights).cuda()

        self.lang_model = StatePredictorNetwork(len(vocab), embed_dim, vocab_weights)
        self.lang_model.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/StatePredictor_both.pt")) # trained with embeddings , map_location=lambda storage, loc: storage        
        self.lang_model.eval()
        for params in self.lang_model.parameters():
            params.requires_grad = False

        self.actor = SimpleNetwork(300)
        self.actor.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_statepred.pt")) # trained with embeddings , map_location=lambda storage, loc: storage

        self.critic = SimpleNetwork(300)
        self.critic.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_statepred.pt")) # trained with embeddings , map_location=lambda storage, loc: storage

        ### Start State Predictor  ###


        ### Start Autoencoder ###
        # self.lang_model = StateAutoencoder()
        # self.lang_model.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/StateAutoencoder_both1.pt")) # trained with embeddings , map_location=lambda storage, loc: storage        
        # self.lang_model.eval()
        # for params in self.lang_model.parameters():
        #     params.requires_grad = False

        # self.actor = SimpleNetwork(300)
        # self.actor.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_autoencoder1.pt")) # trained with embeddings , map_location=lambda storage, loc: storage

        # self.critic = SimpleNetwork(300)
        # self.critic.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_autoencoder1.pt")) # trained with embeddings , map_location=lambda storage, loc: storage

        ### Start CORRECT language ###

        # hidden size 32
        
        # name = "compiled_dataset_08131950" #add 50 back in
        # embed_dim = 300 # switch this later!!
        # embed_size = embed_dim

        # with open('/home/kdmarino/Mazebase_gym_environ/mazebaseenv/data/'+name+'_all_instructions', 'rb') as f:
        #     all_instructions = pickle.load(f)

        # vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

        # vocab.add_word('<pad>')
        # vocab.add_word('<start>')
        # vocab.add_word('<end>')
        # vocab.add_word('<unk>')

        # self.vocab = vocab
        # temp = np.zeros((1,300), dtype=np.float32)
        # vocab_weights = np.concatenate((vocab_weights, temp), axis=0)
        # vocab_weights = torch.Tensor(vocab_weights).cuda()

        # self.lang_model = LanguageWithAttentionSUM(len(vocab), embed_dim, vocab_weights, training=False)
        # self.lang_model.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/LanguageWithAttentionSUM_nostate.pt")) # trained with embeddings , map_location=lambda storage, loc: storage        
        # #self.lang_model.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/LanguageWithAttentionSUM_adam.pt")) # trained with embeddings , map_location=lambda storage, loc: storage        
        # #self.lang_model.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/LanguageWithAttentionSUM_75per.pt")) # trained with embeddings , map_location=lambda storage, loc: storage        
        # #self.lang_model.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/LanguageWithAttentionSUM_missing10per.pt")) # trained with embeddings , map_location=lambda storage, loc: storage
        # self.lang_model.eval()
        # for params in self.lang_model.parameters():
        #     params.requires_grad = False

        # self.actor = SimpleNetworkNoState(300)
        # self.actor.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/AllObsPredictAttenGLOVE_nostate.pt")) # trained with embeddings , map_location=lambda storage, loc: storage

        # self.critic = SimpleNetworkNoState(300)
        # self.critic.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/AllObsPredictAttenGLOVE_nostate.pt")) # trained with embeddings , map_location=lambda storage, loc: storage

        # #self.actor = SimpleNetwork(300)
        # #self.actor.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_adam.pt")) # trained with embeddings , map_location=lambda storage, loc: storage
        # #self.actor.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_75per.pt")) # trained with embeddings , map_location=lambda storage, loc: storage
        # #self.actor.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_missing10per.pt")) # trained with embeddings , map_location=lambda storage, loc: storage

        # #self.critic = SimpleNetwork(300)
        # #self.actor.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_adam.pt")) # trained with embeddings , map_location=lambda storage, loc: storage        
        # #self.critic.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_75per.pt")) # trained with embeddings , map_location=lambda storage, loc: storage
        # #self.critic.load_state_dict(torch.load("/scratch/kdmarino/Mazebase_dialog/TRAINED_MODELS/SimpleNetwork_missing10per.pt")) # trained with embeddings , map_location=lambda storage, loc: storage
        
        ### End CORRECT language ###


        ### Start language ###

        # hidden size 32
        
        # name = "compiled_dataset_08131950" #add 50 back in
        # embed_dim = 300 # switch this later!!
        # embed_size = embed_dim

        # with open('/home/kdmarino/Mazebase_gym_environ/mazebaseenv/data/'+name+'_all_instructions', 'rb') as f:
        #     all_instructions = pickle.load(f)

        # vocab, vocab_weights = build_vocabulary(all_instructions, name, embed_dim)

        # vocab.add_word('<pad>')
        # vocab.add_word('<start>')
        # vocab.add_word('<end>')
        # vocab.add_word('<unk>')

        # self.vocab = vocab
        # temp = np.zeros((1,300), dtype=np.float32)
        # vocab_weights = np.concatenate((vocab_weights, temp), axis=0)
        # vocab_weights = torch.Tensor(vocab_weights).cuda()

        # self.lang_model = LanguageWithAttention(len(vocab), embed_dim, vocab_weights)
        # self.lang_model.load_state_dict(torch.load("/home/kdmarino/Mazebase_gym_environ/mazebaseenv/TRAINED_MODELS/LanguageWithAttention_both.pt")) # trained with embeddings , map_location=lambda storage, loc: storage
        # self.lang_model.eval()

        # self.actor = AllObsPredictAtten(hidden_size, vocab, vocab_weights)
        # self.actor.load_state_dict(torch.load("/home/kdmarino/Mazebase_gym_environ/mazebaseenv/TRAINED_MODELS/AllObsPredictAtten_both.pt")) # trained with embeddings , map_location=lambda storage, loc: storage

        # self.critic = AllObsPredictAtten(hidden_size, vocab, vocab_weights)
        # self.critic.load_state_dict(torch.load("/home/kdmarino/Mazebase_gym_environ/mazebaseenv/TRAINED_MODELS/AllObsPredictAtten_both.pt")) # trained with embeddings , map_location=lambda storage, loc: storage
        
        ### End language ###

        #self.actor = nn.Sequential(
        #    init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        #    init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        #self.critic = nn.Sequential(
        #    init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        #    init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        #x = x[:, 1:] #if includes count ... ???

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        ### Language start
        #seqs, lang_hidden = self.lang_model.get_hidden_state_new(x, self.vocab)

        # #for debugging / explaining language
        # for sampled_ids in seqs:
        #     sampled_caption = []
        #     for word_id in sampled_ids:
        #         word = self.vocab.idx2word[word_id]
        #         sampled_caption.append(word)
        #         if word == '<end>':
        #             break
        #     sentence = ' '.join(sampled_caption) 
        #     print(sentence)

        #hidden_critic = self.critic(x, lang_hidden)
        #hidden_actor = self.actor(x, lang_hidden)
        ### Language end


        ### Autoencoder start

        reconstruction, hidden = self.lang_model(x) # state predictor
        #state_encoding, hidden, reconstruction = self.lang_model(x)
        hidden_critic = self.critic(x, hidden)
        hidden_actor = self.actor(x, hidden)

        ### Autoencoder end


        #hidden_critic = self.critic(x)
        #hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

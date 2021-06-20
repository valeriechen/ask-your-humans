import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from envs import make_vec_envs
#from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from pytorchppo.a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from build_vocab import build_vocabulary, load_vocabulary
import pickle

sys.path.append('pytorchppo/a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1,
    help='log interval, one log per n updates (default: 1)')
parser.add_argument(
    '--env-name',
    default='mazebase-v0',
    help='environment to train on (default: mazebase-v0)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det


embed_dim = 300 
embed_size = embed_dim

with open('data/dataset_all_instructions', 'rb') as f:
    all_instructions = pickle.load(f)

vocab, vocab_weights = build_vocabulary(all_instructions, 'blah', embed_dim)

vocab.add_word('<pad>')
vocab.add_word('<start>')
vocab.add_word('<end>')
vocab.add_word('<unk>')

device = torch.device("cuda:0")

env = make_vec_envs(args.env_name, args.seed+101, 1,
                         None, None, device, False, vocabulary=vocab)


actor_critic, ob_rms = torch.load(args.load_dir + ".pt")

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()


count = 0
for i in range(100):
    for step in range(100):

        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

        if done:
            count = count + 1
            break
    if not done:
        obs = env.reset()
print(count)

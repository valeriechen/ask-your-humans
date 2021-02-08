# Written by Anon Author
import sys
import os
import re
from lib.rl.envs import VecNormalize, VecPyTorch, VecPyTorchFrameStack
from lib.rl.model import Policy, DQNPolicy
from lib.rl.storage import RolloutStorage
from lib.rl.utils import get_vec_normalize, update_linear_schedule
from lib.knowledge import TripletInfo
import lib.rl.algo as algo
import numpy as np
import gym
import torch
import pdb

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

class PreconditionRLModule(object):
    ''' This class wraps everything needed to treat precondition satisfaction as an RL problem
    Takes the raw observation from the environment
    Converts it to the representation we want for our specific MDP to solve the precondition satisfaction problem
    Runs the RL algorithm on this MDP. Does what you would expect a main function in RL to day including replay memory
    algorithm updates, environment resets, etc
    '''
    def __init__(self, rl_opt, env_opt, game_bounds, possible_actions):
        super(PreconditionRLModule, self).__init__()

        # Initialize arguments and other values
        self.use_gae = rl_opt['alg']['use_gae']
        self.tau = rl_opt['alg']['gae_tau']
        self.gamma = rl_opt['env']['gamma'] 
        self.num_steps_update = rl_opt['alg']['num_steps']
        self.goal_precond_reward = rl_opt['env']['goal_precond_reward']
        self.irrelevant_precond_reward = rl_opt['env']['irrelevant_precond_reward']
        self.finish_reward = rl_opt['env']['finish_reward']
        self.possible_actions = possible_actions
        self.total_steps = 0
        self.total_episodes = 0

        # Right now, expecting we have just one process
        # Env is outside of this module, essentially
        if rl_opt['alg']['num_processes'] != 1:
            raise Exception("Right now we only have this happening online, so need to do other stuff for this to work for A2C for instance")

        # Get things in state that we put in obs as one-hot
        pattern = re.compile(r'\s+')
        with open(os.path.join(env_opt['knowledge_root'], env_opt['state_rep']['obs_list'])) as f:
            line = f.readline()
        line = re.sub(pattern, '', line)
        self.constant_list = line.split(',')

        # Calculate input size
        # TODO - should we add inventory to state?
        # Input layer for each precondition type (except action)
        num_layers = len(self.constant_list) + len(TripletInfo.PRE_CONDITION_TYPES) - 1
        self.obs_shape = (num_layers, game_bounds[0], game_bounds[1])
        self.action_space = gym.spaces.Discrete(len(possible_actions))
        # TODO - right now allowing all actions here. Might make sense to limit this at some point to non-crafting and non-pass actions

        # Assume observation space is 3D
        assert(len(self.obs_shape) == 3)

        # Make actor_critic policy agent
        if rl_opt['alg']['alg_name'] == 'dqn':
            self.policy = DQNPolicy(self.obs_shape, self.action_space, rl_opt['modle'])
            self.agent = algo.DQN(self.policy, rl_opt['env']['gamma'], batch_size=rl_opt['alg']['batch_size'], target_update=rl_opt['alg']['target_update'],
                                    mem_capacity=rl_opt['alg']['mem_capacity'], lr=rl_opt['optim']['lr'], eps=rl_opt['optim']['eps'], 
                                    max_grad_norm=rl_opt['optim']['max_grad_norm'])
        else:
            self.policy = Policy(self.obs_shape, self.action_space,
                            base_kwargs={'recurrent': rl_opt['model']['recurrent_policy']})
            if rl_opt['alg']['alg_name'] == 'a2c':
                self.agent = algo.A2C_ACKTR(self.policy, rl_opt['alg']['value_loss_coef'],
                                       rl_opt['alg']['entropy_coef'], lr=rl_opt['alg']['lr'],
                                       eps=rl_opt['optim']['eps'], alpha=rl_opt['optim']['alpha'],
                                       max_grad_norm=rl_opt['optim']['max_grad_norm'])
            elif rl_opt['alg']['alg_name'] == 'ppo':
                self.agent = algo.PPO(self.policy, rl_opt['alg']['clip_param'], rl_opt['alg']['ppo_epoch'], 
                                    rl_opt['alg']['num_mini_batch'], rl_opt['alg']['value_loss_coef'], rl_opt['alg']['entropy_coef'], 
                                    lr=rl_opt['optim']['lr'], eps=rl_opt['optim']['eps'],
                                    max_grad_norm=rl_opt['optim']['max_grad_norm'])
            elif rl_opt['alg']['alg_name'] == 'acktr':
                self.agent = algo.A2C_ACKTR(self.policy, rl_opt['alg']['value_loss_coef'],
                                       rl_opt['alg']['entropy_coef'], acktr=True)

        # Make and initialize rollouts
        self.rollouts = RolloutStorage(self.num_steps_update, rl_opt['alg']['num_processes'],
                            self.obs_shape, self.action_space,
                            self.policy.recurrent_hidden_state_size)

        # Initialize (to None) the current goal for the RL agent
        self.goal_preconds = None
        self.last_preconds = None

    # Called when precond MDP has finished, either because it successfuly got to precondition
    # Or there was a timeout event
    def reset(self, success):
        # Set last reward to be the success
        # When get_action is next called, the last reward will set to whether this was successful
        self.last_reward = self.finish_reward * float(success)

        # self.last_done is [True] since last things we recorded ended the episode
        self.last_done = [[True]]

        # Update episode counter
        self.total_episodes += 1

        # Reset goals and preconds to None
        self.goal_preconds = None
        self.last_preconds = None

    # Initialize the RL agent withnew preconditions before we start
    def init_mdp(self, goal_preconditions):
        # Save the goal preconditions and the currently satisfied preconditions
        self.goal_preconds = goal_preconditions
        self.last_preconds = None

    # Get reward (and end condition)
    def get_reward(self, last_preconds, cur_preconds):
        # Init, get finished
        reward = float(0)
        finished = len(self.goal_preconds - cur_preconds) == 0

        # Get set differences
        new_preconds = cur_preconds - last_preconds
        removed_preconds = last_preconds - cur_preconds

        for precond in new_preconds:
            # Ignore action precondition
            if precond[0] == TripletInfo.REQUIRES_ACTION:
                continue

            # If we added a goal precondition, give a small positive reward here
            if precond in self.goal_preconditions:     
                reward += self.goal_precond_reward

            # If we added an irrelevant precondition, give a small negative reward here
            else:
                reward -= self.irrelevant_precond_reward

        for precond in removed_preconds: 
            # Ignore action precondition
            if precond[0] == TripletInfo.REQUIRES_ACTION:
                continue

            # If we removed a goal precondition, give a small negative reward here
            if precond in self.goal_preconditions:
                reward -= self.goal_preconditions

            # If we removed an irrelevant precondition, give a small positive reward here
            else:
                reward += self.irrelevant_precond_reward

        # Return
        return reward, finished

    # Make another step in the precondition satisfaction MDP and update networks  
    def get_action(self, raw_obs_info, cur_satisfied_preconds):
        # Get obs for network input
        obs = self.get_obs_rep(raw_obs_info, cur_satisfied_preconds) 

        # If first step, last reward was set in reset
        if self.last_preconds is not None:
            # Get the reward and end condition from last time step, unless we're at a reset
            self.last_reward, finished = get_reward(self.last_preconds, cur_satisfied_preconds)

            # Finished should never be true here (should have checked this outside)
            assert(not finished)
            self.last_done = [[finished]]
        
        # At first ever step, need to init rollouts, and can't do insert for rollouts yet
        if self.total_steps == 0:
            self.rollouts.obs[0].copy_(obs)
        else:
            # First, do update from last step
            reward = torch.FloatTensor([[self.last_reward]]) # TODO - is this conversion correct?
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                    for done_ in self.last_done])
            rollouts.insert(obs, self.last_recurrent_hidden_states, self.last_action, self.last_action_log_prob, self.last_value, reward, masks)
            self.total_steps += 1
        
            # If we've filled up the rollouts, do an update
            if self.total_steps % self.num_steps_update == 0:
                with torch.no_grad():
                    next_value = self.policy.get_value(rollouts.obs[-1],
                                                        rollouts.recurrent_hidden_states[-1],
                                                        rollouts.masks[-1]).detach()
                rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
                value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
                rollouts.after_update()
                # TODO - log the losses somewhere probably?
       
        # Sample actions
        with torch.no_grad():
            # TODO - I could have an off by one on steps here, the logic is kind of subtle. It seems right
            step = self.total_steps % self.num_steps_update
            value, action, action_log_prob, recurrent_hidden_states = self.policy.act(
                    self.rollouts.obs[step],
                    self.rollouts.recurrent_hidden_states[step],
                    self.rollouts.masks[step])
            self.last_value = value
            self.last_action = action
            self.last_action_log_prob = action_log_prob
            self.last_recurrent_hidden_states = recurrent_hidden_states

        # TODO - here we assume no parallel again
        action_ind = action.item()
        action = self.possible_actions[action_ind]

        # Return action
        return action

    # TODO - Def cuda? Convert everything to cuda???

    # Take in raw observations from environment and put it in format expected by network
    def get_obs_rep(self, obs, cur_preconds):
        grid_obs = obs['grid_obs']
        side_info = obs['side_info']
        one_hot = obs['grid_one_hot']
        one_hot_side = obs['side_one_hot']
        vocab = obs['vocab']

        # Create initial input (all zeros)
        obs = np.zeros(self.obs_shape) 

        # Lets say we're given a list of the items in the world we actually care about always
        # Should include things like Agent or ResourceFont, CraftingItem, CraftingContainer, Block and Water
        obs_onehots = []
        for ind, item in enumerate(self.constant_list):
            item_id = vocab.index(item)
            item_onehot = one_hot[:, :, item_id]
            obs[ind] = item_onehot

        # Go through each precondition we have yet to satisfy
        # Add the concept item locations 
        assert(TripletInfo.PRE_CONDITION_TYPES[0] == TripletInfo.REQUIRES_ACTION)
        unsatisfied_preconds = self.goal_preconds - cur_preconds
        for precond in unsatisfied_preconds:
            # Should never have action here
            precond_type = precond[0]           
            assert(precond_type != TripletInfo.REQUIRES_ACTION)

            # Get the index where we put this precondition
            type_index = TripletInfo.PRE_CONDITION_TYPES.index(precond_type) + len(self.constant_list) - 1

            # Get the location of all the relevant items and add to observation
            concept = precond[1]
            item_id = vocab.index(concept)
            item_onehot = one_hot[:, :, item_id]
            obs[type_index] += item_onehot
               
        # Convert to pytorch
        obs = torch.from_numpy(obs).float()

        # TODO - more processing?
        # Cuda here maybe?
        return obs

    # Load model from checkpoint
    def load_state_dict(self, ckpt):
        self.total_episodes = ckpt['total_episodes']
        self.agent.load_state_dict(ckpt['agent']) 

    # Get state dictionary info
    def state_dict(self):
        ckpt = {}
        ckpt['total_episodes'] = self.total_episodes
        ckpt['agent'] = self.agent.state_dict()
        return ckpt


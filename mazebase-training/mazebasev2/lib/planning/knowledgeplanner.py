from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import abc
import copy
import random
import six
import types
import pdb

from lib.utils import dictutils
from .knowledgemodel import KnowledgeModel
from .preconditionrl import PreconditionRLModule
from lib.knowledge import TripletInfo, Rule

class KnowledgePlanner(object):
    '''This class contains the class which does knowledge planning on the environment
    Takes in observations from the environment and decides actions
    Keeps internal models and past experiences
    Does low-level and high-level planning and action
    '''
    TEST_MODE = 'test'
    PASS_MODE = 'pass'
    PRECOND_MODE = 'precond_satisfy'

    def __init__(self, game, method_opt, env_opt, Kp, item_dict):
        super(KnowledgePlanner, self).__init__()
        self.item_dict = item_dict
        self.Kp = Kp
        self.mode = method_opt['mode']
        self.max_precond_steps = method_opt['max_precond_steps']

        # Probability we try to verify a proposal first (if any left unverified)
        self.proposal_prob = method_opt['proposal_prob']
        self.necessity_prob = method_opt['necessity_prob']
        self.sufficient_prob = method_opt['sufficient_prob']

        # Create KnowledgeModel
        self.K_model = KnowledgeModel(method_opt['knowledge'], env_opt, Kp, item_dict)

        # Create an RL module for the precondition satisfaction part
        self.precond_rl = PreconditionRLModule(method_opt['rl'], env_opt, game.get_max_bounds(), game.actions())

        # Initialize last_obs and last_action as empty
        self.last_obs = None
        self.last_action = None

    # Reset planner to a new reset
    def reset(self, last_obs, first_obs):
        # Resolve the last observation from before the reset (if relevant)
        if last_obs is not None:
            self.check_obs(last_obs, on_reset=True)

        # Reset plans we've tried before
        self.tried_plans = []

        # Call make_plan
        # This makes a new plan for the current observation
        self.make_plan(first_obs)

        # Reset all the variables
        self.last_obs = None
        self.last_action = None

    # Parse info out of observation
    def parse_out_inv(self, obs):
        cur_inv = {}
        side_info = obs['side_info']
        inventory_raw = None
        for row in side_info:
            assert(row[0] == 'INFO')
            if row[1] == 'INVENTORY':
                inventory_raw = row
                break
        assert(inventory_raw is not None)
        for item in inventory_raw:
            # Skip first two and break at empty
            if item in ['INFO', 'INVENTORY']:
                continue
            if item == '':
                break
            
            # Parse item name and count
            item_name = item.split('_count')[0]
            item_count = int(item.split('=')[-1])
            cur_inv[item_name] = item_count
        return cur_inv

    # Check for known post conditions we might have satisfied between observations
    def check_post_conditions(self, last_obs, obs):
        # Right now only effect is one crafting item is created
        # TODO - add more effects
        post_conditions_satisfied = []        
        main_effect = None

        # Get previous and current inventories
        last_inventory = self.parse_out_inv(last_obs)
        cur_inventory = self.parse_out_inv(obs)

        # Find the difference between the inventories
        inventory_diff = dictutils.diff_dicts(cur_inventory, last_inventory) 

        # Find items that were added
        items_added = {}
        for item in inventory_diff:
            if inventory_diff[item] > 0:
                items_added[item] = inventory_diff[item]
        
        # We should never be in a situation where we add two or more new items to inventory
        # Items should no longer spawn in same square
        # Crafting rules should never be satisfiable at the same time
        assert(len(items_added) <= 1)

        # Add item created post-condition
        if len(items_added) == 1:
            item = list(items_added.keys())[0]
            # Create item created postcondition
            assert(items_added[item] <= TripletInfo.MAX_CREATED_ITEMS)
            post_condition_type = TripletInfo.CREATES_ITEM_X % items_added[item]
            created_post_condition = (post_condition_type, item)
            post_conditions_satisfied.append(created_post_condition)
            main_effect = item
            # TODO - maybe this main effect assumption is not necessary. It does make search easier

        # Add destroyed item post-conditions
        for item in inventory_diff:
            if inventory_diff[item] < 0:
                num_destroyed = abs(inventory_diff[item])
                assert(num_destroyed <= TripletInfo.MAX_CONSUMED_ITEMS)
                post_condition_type = TripletInfo.DESTROYS_ITEM_X % num_destroyed
                destroyed_post_condition = (post_condition_type, item)
                post_conditions_satisfied.append(destroyed_post_condition)

        return main_effect, post_conditions_satisfied

    # Get inventory preconditions
    def get_inventory_preconditions(self, inventory):
        inv_preconds = set()
        for item in inventory:
            count = inventory[item]
            for i in range(count):
                item_count = i+1
                assert(item_count <= TripletInfo.MAX_REQUIRED_ITEMS)
                inv_preconds.add((TripletInfo.REQUIRES_ITEM_X % item_count, item))
        return inv_preconds

    # Get all location based preconditions
    def get_location_preconditions(self, flat_environment):
        # Get all the locations that are in flat_env
        locations = set([location for location in flat_environment if location in self.item_dict])
        location_preconds = set((TripletInfo.REQUIRES_LOCATION, location) for location in locations)
        return location_preconds

    # Given observation and action, return all the satisfied preconditions at this state
    def sense_all_satisfied_preconds(self, obs, action=None):
        all_satisfied_precond = set()
        if action is not None:
            all_satisfied_precond.add((TripletInfo.REQUIRES_ACTION, action))

        # Next, add the location based ones
        # Get the grid location of the agent
        grid_obs = obs['grid_obs']
        agent_grid = None
        for row in grid_obs:
            for tile in row:
                if type(tile) != list:
                    pdb.set_trace()
                if 'Agent' in tile:
                    agent_grid = tile 
        assert(agent_grid is not None)

        # Get all the relevant items in your grid and add them as conditions
        all_satisfied_precond.union(self.get_location_preconditions(agent_grid))

        # Next, add the inventory ones
        # Just requires item and depletes item for every item and count we have
        # This gets pruned later based on items expended
        inventory = self.parse_out_inv(obs)
        all_satisfied_precond.union(self.get_inventory_preconditions(inventory))

        return all_satisfied_precond

    # This method takes obs and checks for success conditions
    # Sends rewards to RL agent, updates knowledge, and updates plan and state depending on success
    def check_obs(self, obs, on_reset=False):
        # Check we have satisfied any post conditions
        if self.last_action is not None:
            # Check if any post condition was satisfied and see what preconditions were satisfied
            main_effect_satisfied, post_conditions_satisfied = self.check_post_conditions(self.last_obs, obs) 
            last_preconds = self.sense_all_satisfied_preconds(self.last_obs, self.last_action)

            # TODO TODO - pickup
            # TODO - maybe get rid of this when we deal with multiple rules for same effect
            # TODO - would need to change pick up physics and not allow multiple items in same place first
            # TODO - otherwise we violate the one schema per timestep rule
            # Remove the "pick up" rules (just pick up an item to add it to dictionary)
            #if main_effect_satisfied is not None and any([main_effect_satisfied == precond[1] for precond in last_preconds]) and any([precond[1] == 'grab' for precond in last_preconds]):
            #    main_effect_satisfied = None

            # Update KnowledgeModel
            if self.plan_mode == KnowledgePlanner.TEST_MODE:
                test_intended = self.current_schema.main_effect
            else:
                test_intended = None
            
            # Check if we have anything to update
            # Only update if either some post condition was actually satisfied
            # Or if our preconditions satisfied was something we intented to test
            if main_effect_satisfied is not None or test_intended is not None:
                self.K_model.update_knowledge(main_effect_satisfied, test_intended, post_conditions_satisfied, last_preconds)

            # Possibly update plan if this was the thing we were trying to plan for
            if self.plan_mode == KnowledgePlanner.TEST_MODE:
                self.make_plan(obs)

        # Get the currently satisfied preconditions  
        cur_satisfied_precond = self.sense_all_satisfied_preconds(obs)

        # Check if we've satisfied preconditions or exceded steps, change modes or current goal
        if self.plan_mode == KnowledgePlanner.PRECOND_MODE:
            # Check that thy're all satisfied
            if len(self.goal_preconds - cur_satisfied_precond) == 0:
                # We shouldn't ever be in this situation because we should have checked if we have already
                # Satisfied before we start the RL precond
                assert(self.to_precond_steps > 0)

                # Send reset signal to rl agent, telling it was successful
                self.precond_rl.reset(success=True)

                # Now update current plan given success
                self.update_plan(obs, success=True)

            # If we've exceeded our alloted steps, replan
            elif self.to_precond_steps >= self.max_precond_steps:
                # Sent reset to rl agent, but give failure
                self.precond_rl.reset(success=False)

                # Update current plan given that we have failed to satisfy the current precondition
                self.update_plan(obs, success=False)
            elif on_reset:
                # Sent reset to rl agent, but give failure
                self.precond_rl.reset(success=False)

        # Return the list of currently satisfied preconditions
        return cur_satisfied_precond

    # Plan and decide the next action
    def get_action(self, obs):
        action = None

        # First look at current obs and update states
        satisfied_preconds = self.check_obs(obs)

        # If we're still in precondition mode, try to satisfy
        # This is RL land
        if self.plan_mode == KnowledgePlanner.PRECOND_MODE:
            # Plan to current preconditions
            # Use precondition rl agent policy to decide action
            action = self.precond_rl.get_action(obs, satisfied_preconds)
            self.to_precond_steps += 1
        # If we're in test mode, just do the test action
        elif self.plan_mode == KnowledgePlanner.TEST_MODE:
            # Return the required action    
            action = self.current_schema.get_rule().get_required_action()  
        # If in pass mode, action is pass (no-op)
        elif self.plan_mode == KnowledgePlanner.PASS_MODE:
            action = 'pass'
        else:
            raise Exception('Current mode %s is invalid' % self.plan_mode)

        # Save the current obs and action as last_obs and action
        # Useful for checking post-conditions
        self.last_obs = obs
        self.last_action = action

        # Return action we want to take
        return action

    # Update precondition RL problem
    def update_rl_precond(self, obs):
        # Get the goal preconditions we want to satisfy
        self.goal_preconds = set()
        for precond in self.current_schema.get_preconditions():
            if precond[0] != TripletInfo.REQUIRES_ACTION:
                self.goal_preconds.add(precond)
        already_satisfied = self.sense_all_satisfied_preconds(obs)
       
        # If no non-action preconditions left, now go into testing mode
        if len(self.goal_preconds - already_satisfied) == 0:
            self.plan_mode = KnowledgePlanner.TEST_MODE
        # Otherwise we have a precondition to satisfy, current step in plan is go and do that
        else:
            # Set mode and reset step timer
            self.plan_mode = KnowledgePlanner.PRECOND_MODE
            self.to_precond_steps = 0

            # Init precondition mdp to current goal preconditions 
            self.precond_rl.init_mdp(self.goal_preconds)

    # Update plan after we've satisfied a precondition or we've tested a schema
    # Or if we have failed at some step
    def update_plan(self, obs, success):
        # success=False means a precondition satisfaction step failed
        # May mean we need a new plan
        if not success:
            self.make_plan(obs, on_failure=True)
            return
        
        # If it was successful
        # If plan is through, make a new plan
        if len(self.schema_plan) == 0:
            self.make_plan(obs)
        # Otherwise, update current schema to the next one in the plan and plan the preconditions
        else:
            self.current_schema = self.schema_plan.pop() 
            self.update_rl_precond(obs)

    # Make a plan for the agent to try to follow. Either after when we start, when we execute our last plan
    # Or after we fail at some intermediate step of our last plan
    def make_plan(self, obs, on_failure=False):
        # If we're making a plan after failure, don't try the same thing again (duh)
        if on_failure:
            self.tried_plans.append(self.current_schema) 
        
        # TODO - we're assuming full grounding, perception and no stochasticity right now
        # First, decide what knowledge we want to try to verify
        # 3 options. 
        # 1 - try to verify a Kp subgraph we haven't tried before
        # 2 - try to prune necessity for verified sufficient graphs
        # 3 - try to get a correct sufficient graph given we've tried Kp already and failed
        # For now, do it randomly
       
        # Get all the proposed schemas from Kp
        Kp_proposals = []
        for k in self.K_model.proposed_knowledge:
            for effect in self.K_model.proposed_knowledge:
                Ks_list = self.K_model.proposed_knowledge[effect]
                
                # Only look at Ks we haven't verified yet
                for Ks in Ks_list:
                    if Ks.sufficient is None:
                        Kp_proposals.append(Ks)

        # While loop, try plans until we come up with a realizable plan
        plan_made = False
        failure_count = 0
        while not plan_made:
            failure_count += 1

            # Eventually, we should give up if we can't make a valid plan
            # This might be because we've tried all the plans
            if failure_count > 1000:
                # If we have failed to much, clear our tried plans and be able to try them again
                if len(self.tried_plans) > 0:
                    self.tried_plans = []
                    failure_count = 0
                    continue
                else:
                    # TODO - maybe a better thing to do here, but for now, go to pass only mode
                    self.plan_mode = KnowledgePlanner.PASS_MODE
                    return
                    
            # We're going to start by only doing 1 until we get that working
            # Later this should choose between these three based on chance and what's actually possible
            prob_prop = random.random()
            assert(self.proposal_prob == 1)
            if prob_prop < self.proposal_prob and len(Kp_proposals) > 0:
                # Randomly choose a Ks from Kp
                Ks = random.choice(Kp_proposals)

                # If it's already in tried plans, continue and try again
                for Kt in self.tried_plans:
                    if Kt.precond_set_equal(Ks):
                        continue
            # TODO TODO TODO
            else:
                # TODO - maybe a more graceful way to do this?
                #pdb.set_trace()
                raise Exception("We've verified all the knowledge! Congrats!")

            # Make a schema stack 
            self.schema_plan = []
            self.schema_plan.append(Ks)
            
            # Given Ks we want to test, get the preconditions we need
            final_preconditions = Ks.get_preconditions()
            final_preconditions = [p for p in final_preconditions if p[0] != TripletInfo.REQUIRES_ACTION]       

            # Look at obs and see if we can satisfy all of them in the current environment
            final_unsatisfied = []
            raw_grid = obs['grid_obs']
            flat_list = [item for sublist in raw_grid for item in sublist]
            flat_list = [item for sublist in flat_list for item in sublist]
            inventory = self.parse_out_inv(obs)
            for precond in final_preconditions:               
                # It should be true that the second part of precondition is an item that exists in grid or inventory
                # Check if it's in grid
                if precond[1] in flat_list:
                    continue

                # Check if it's in inventory
                if precond[1] in inventory:
                    continue

                # If not in either, needs another rule to satisfy, or it's just not satisfiable
                final_unsatisfied.append(precond)

            '''# If there are some we cannot satisfy, use crafting graph to see if we can get those recursively
            # TODO - I'm going to save this for later once we get the depth 1 working

            # Get the crafting graph from knowledge model
            crafting_graph = self.K_model.crafting_graph
            
            # TODO - should make new function here that like, given some initial items and things in the world does a forward propogation
            # through the the known schemas in crafting graph and returns items we are able construct (and maybe those satisfy other preconditions)
            # TODO - it needs to somehow deal with some of the end items actually being mutual exlusive somehow. TODO
            possible_items = crafting_graph.forward(available_items)
            # Use crafting graph and initial obs to figure out how/if we can satisfy those preconditions
            # TODO - for now this is just items, but later, this might be switch states and such 
            
            # TODO - also make sure no subset was in tried_plans
            '''

            if len(final_unsatisfied) > 0:
                continue    # Try another schema
            # Will add the other rules we need to do by appending them to self.schema_plan

            # Pop the next schema from self.schema_plan            
            plan_made = True
           
        # If that succeeded, figure out how to satisfy the current precondition we want to satisfy
        self.plan_mode = KnowledgePlanner.PRECOND_MODE

        # Pop current schema and start to execute the precondition satisfaction
        self.current_schema = self.schema_plan.pop() 
        self.update_rl_precond(obs)

    # Given an inventory and flattened grid environment, returns a set of possible preconditions we could satisfy
    #def get_all_possible_preconditions

    # Given an initial inventory and world, forward plan all the possible precondition sets we can get to
    # We can then use these to choose schemas to test
    
    # def forward_plan_preconditions(self, init_inventory, flat_environment, max_depth):
        
    #     # TODO - do one step of this at any given inventory and environment

    #     # First, get all preconditions that can be satisfied in current environment
    #     location_preconds = self.get_location_preconditions(self, flat_environment)
    #     inventory_preconds = self.get_inventory_preconditions(inventory)
        
    #     # ??? TODO?
    #     action_preconds = self.get_

    #     # Get 

    #     # Go through all rules and see if they can be applied
    #     # TODO - no, make this the knowledge graph
    #     # and then check precondition sets!
    #     for rule_graph in self.sufficient_knowledge:
    #         if 
        
    #     #a list of all runable rules from these preconds
    #     self.K_model.crafting_graph



    #     # TODO TODO - need to add pick up schemas now!
    #     # TODO TODO - need like a item not in world anymore post-condition though so we can't infinitely pick up something!!!
    #     # TODO TODO - can also use this to deal with resource fonts if we don't want them to be unlimited anymore

    #     # All items in inventory can be added as requires item preconditions
    #     for item in inventory


    #     # TODO - more preconditions here once we add more to environment    
    
    #         # Look at obs and see if we can satisfy all of them in the current environment
    #         final_unsatisfied = []
    #         raw_grid = obs['grid_obs']
    #         flat_list = [item for sublist in raw_grid for item in sublist]
    #         flat_list = [item for sublist in flat_list for item in sublist]
    #         inventory = self.parse_out_inv(obs)
    #         for precond in final_preconditions:               
    #             # It should be true that the second part of precondition is an item that exists in grid or inventory
    #             # Check if it's in grid
    #             if precond[1] in flat_list:
    #                 continue

    #             # Check if it's in inventory
    #             if precond[1] in inventory:
    #                 continue

    #             # If not in either, needs another rule to satisfy, or it's just not satisfiable
    #             final_unsatisfied.append(precond)

    #         '''# If there are some we cannot satisfy, use crafting graph to see if we can get those recursively
    #         # TODO - I'm going to save this for later once we get the depth 1 working

    #         # Get the crafting graph from knowledge model
    #         crafting_graph = self.K_model.crafting_graph
            
    #         # TODO - should make new function here that like, given some initial items and things in the world does a forward propogation
    #         # through the the known schemas in crafting graph and returns items we are able construct (and maybe those satisfy other preconditions)
    #         # TODO - it needs to somehow deal with some of the end items actually being mutual exlusive somehow. TODO
    #         possible_items = crafting_graph.forward(available_items)
    #         # Use crafting graph and initial obs to figure out how/if we can satisfy those preconditions
    #         # TODO - for now this is just items, but later, this might be switch states and such 
            
    #         # TODO - also make sure no subset was in tried_plans
    #         '''

    #         if len(final_unsatisfied) > 0:
    #             continue    # Try another schema
    #         # Will add the other rules we need to do by appending them to self.schema_plan

    #         # Pop the next schema from self.schema_plan            
    #         plan_made = True
 

    # Load models from a checkpoint
    def load_state_dict(self, ckpt):
        # Kp should not have changed
        assert(ckpt['Kp'] == self.Kp)

        # Load models for knowledge and rl
        self.K_model.load_state_dict(ckpt['K_model'])
        self.precond_rl.load_state_dict(ckpt['precond_rl'])

    # Get checkpoint to save
    def state_dict(self):
        ckpt = {}
        ckpt['Kp'] = self.Kp

        # Get things to save
        ckpt['K_model'] = self.K_model.state_dict()
        ckpt['precond_rl'] = self.precond_rl.state_dict()

        return ckpt

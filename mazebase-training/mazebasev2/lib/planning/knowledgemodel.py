# Written by Anon Author
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.knowledge import CraftingGraph, KnowledgeSubGraph, Rule
from itertools import product
import pdb

class KnowledgeModel(nn.Module):
    '''This class contains the networks, memory and training data for the knowledge networks
    Contains three knowledge networks, that may share parameters

    '''

    # Init function
    def __init__(self, opt, env_opt, Kp, item_dict):
        super(KnowledgeModel, self).__init__()
        self.item_dict = item_dict
        self.max_craft_depth = env_opt['max_craft_depth']
        self.mode = opt['mode']

        # If in no_learning mode, this module just keeps track of confirmed knowledge and traverses through it
        if self.mode == 'no_learning':
            # Confirmed knowledge is every precondition set we've checked bevore
            self.confirmed_knowledge = {}

            # Sufficient knowledge is one precondition graph for each effect. Keeps the largest set possible
            self.sufficient_knowledge = {}

            # Crafting graph is a crafting graph constructed from known sufficient schemas
            self.crafting_graph = CraftingGraph([], self.item_dict, self.max_craft_depth)

            # Now add proposed knowledge
            self.proposed_knowledge = {}
            for proposed_K in Kp:
                # Make proposed knowledge subgraph
                proposed_K = KnowledgeSubGraph(Rule(proposed_K).create_triplet_corresp())

                # Add to proposed_knowledge using effect as key to a list
                if proposed_K.main_effect not in self.proposed_knowledge:
                    self.proposed_knowledge[proposed_K.main_effect] = []
                self.proposed_knowledge[proposed_K.main_effect].append(proposed_K)

        # In this mode, the world is still grounded and deterministic, but we also learn graph networks to propose schemas
        elif self.mode == 'grounded_deterministic_learned':
            # Define all of the network modules here, including shared modules
            # e.g.
            # self.in_fc = nn.Linear(hid_sz, hid_sz)
            # TODO
            # TODO - make this conditional
            # Also maybe make GCN GGN an optional switch?
            pass
        else:
            raise Exception("Mode %s for knowledge not implemented" % self.mode)
        
    # Update our confirmed knowledge
    def update_knowledge(self, main_effect_satisfied, test_intended, post_conditions_satisfied, all_satisfied_precond):
        # TODO - update this when we make the learned version of this
        assert(self.mode == 'no_learning')
        assert(test_intended is not None or main_effect_satisfied is not None)
       
        # Get the effect names so we can find them in our knowledge      
        if main_effect_satisfied  is not None and test_intended is not None and main_effect_satisfied != test_intended:
            effects = [main_effect_satisfied, test_intended]
        elif main_effect_satisfied is not None:
            effects = [main_effect_satisfied]
        else:
            effects = [test_intended]

        # Iterate through all relevant effects
        for effect in effects:
            # Make a temporary knowledge subgraph of what we actually tested and any post conditions we actually satisfied
            precondition_triplets = [("K_tested", precond[0], precond[1]) for precond in all_satisfied_precond]
            postcondition_triplets = [("K_tested", postcond[0], postcond[1]) for postcond in post_conditions_satisfied]
            new_triplets = set(precondition_triplets + postcondition_triplets)
            K_tested = KnowledgeSubGraph(new_triplets)  
            K_tested.update_sufficiency(post_conditions_satisfied)
  
            # If it's exactly in proposed_knowledge, replace it
            # Now the graph it points to will have sufficient not None (thus tested)
            if effect in self.proposed_knowledge:
                for K in self.proposed_knowledge[effect]:
                    # If it's the same graph, set proposed_knowledge
                    if K.precond_set_equal(K_tested):
                        K.update_sufficiency(post_conditions_satisfied)

            # Now try to combine it somehow with already confirmed knowledgemodel
            if effect not in self.confirmed_knowledge:
                self.confirmed_knowledge[effect] = [K_tested]
            else:
                # Check total equality case
                for K in self.confirmed_knowledge[effect]:
                    # Check and eliminate total equality case                 
                    if K.precond_set_equal(K_tested):
                        # If they're identical, in deterministic fully grounded case,
                        # the outcome better be the same. Nothing changes
                        # TODO - this changes in non-deterministic case
                        assert(K.sufficient == K_tested.sufficient)
                        # TODO - need to assert that the postcondition sets are the same 
                        # (this might change if we end up with postconditions that are not caused by the agent)
                        if K.get_postconditions() != K_tested.get_postconditions():
                            pdb.set_trace()
                        assert(K.get_postconditions() == K_tested.get_postconditions())
                        return
                
                    # Update necessity between this graph and all the other graphs
                    K.update_necessity(K_tested)
                    K_tested.update_necessity(K)

                # Add K_tested to the confirmed knowledge
                self.confirmed_knowledge[effect].append(K_tested)

            # Now update sufficient knowledge
            # Only do this when the knowledge is sufficient and update only on the right effect
            if K_tested.sufficient and K_tested.main_effect == effect:
                # This is first successful graph, so add it as the sufficient knowledge
                if effect not in self.sufficient_knowledge:
                    self.sufficient_knowledge[effect] = [K_tested]
                else:
                    # Go through each sufficient schema and try to merge it
                    merged = False
                    for K in self.sufficient_knowledge[effect]:
                        if K.schema_compatible(K_tested):
                            K.merge_other(K_tested)

                    # If the new schema is not compatible with any other, add it as a new sufficient knowledge rule
                    if not merged:
                        self.sufficient_knowledge[effect].append(K_tested) 

        # Go back through the confirmed knowledge list again and update our sufficient knowledge
        if effect in self.sufficient_knowledge:
            for K in self.confirmed_knowledge[effect]:
                for K_suff in self.sufficient_knowledge[effect]:
                    K_suff.update_necessity(K)
        
        # Update crafting graph
        min_suff_rules = []
        ind = 1
        for effect in self.sufficient_knowledge.keys():
            for K in self.sufficient_knowledge[effect]:
                min_rule = K.get_min_suff_rule()
                min_rule.update_rule_name('Rule%d' % ind)
                ind += 1
                min_suff_rules.append(min_rule)
        self.crafting_graph = CraftingGraph(min_suff_rules, self.item_dict, self.max_craft_depth)

    # Not implemented. Knowledge networks invoked individually
    def forward(self, inputs, states, masks):
        raise NotImplementedError

    # Def cuda function?

    # Load data and models from checkpoint
    def load_state_dict(self, ckpt):
        assert(self.mode == 'no_learning')

        # Load the saved/tried knowledge
        self.confirmed_knowledge = ckpt['confirmed_knowledge']
        self.sufficient_knowledge = ckpt['sufficient_knowledge']    
        self.proposed_knowledge = ckpt['proposed_knowledge']
        self.crafting_graph = ckpt['crafting_graph']

    # Get checkpoint to save
    def load_state_dict(self):
        assert(self.mode == 'no_learning')

        # Load the saved/tried knowledge
        ckpt = {}
        ckpt['confirmed_knowledge'] = self.confirmed_knowledge
        ckpt['sufficient_knowledge'] = self.sufficient_knowledge   
        ckpt['proposed_knowledge'] = self.proposed_knowledge 
        ckpt['crafting_graph'] = self.crafting_graph
        return ckpt

class SufficiencyNetwork(nn.Module):
    '''This network, for a given graph/schema, predicts if the entire graph is sufficient
    i.e. does the schema result in the correct effect
    '''
    def __init__(self, opt, common_modules=None):
        super(SufficiencyNetwork, self).__init__()
        # TODO
        # Should either create or load all the modules from common_modules
        # TODO

    def forward(self, inputs):
        # Takes in a graph-structured input which is the schema we want to test
        # Outputs a sigmoid 0/1 sufficiency prediction
        # Should be able to do this in batch
        raise NotImplementedError

class NecessityNetwork(nn.Module):
    '''This network, for a sufficient graph/schema
    predicts for each edge if that precondition is necessary
    '''
    def __init__(self, opt, common_modules=None):
        super(NecessityNetwork, self).__init__()
        # TODO
        # Should either create or load all the modules from common_modules
        # TODO

    def forward(self, inputs):
        # Takes in a graph-structured input which is the schema we want to test
        # Outputs sigmoid 0/1 for each node and/or edge for whether that precondition is necessary
        raise NotImplementedError

class MakeSufficientNetwork(nn.Module):
    '''This network, given an insufficient graph/shema
    predicts for each edge if the precondition is sufficient
    and predicts a hidden state we use to decide what edges to add
    '''
    def __init__(self, opt, common_modules=None):
        super(MakeSufficientNetwork, self).__init__()
        # TODO
        # Should either create or load all the modules from common_modules
        # TODO

    def forward(self, inputs):
        # Takes in a graph-structured input which is the schema we want to make right
        # Outputs a sigmoid 0/1 on each edge for preconditions we want to eliminate
        # And a hidden state used by the edge comparitor network
        raise NotImplementedError

class EdgeComparitorNetwork(nn.Module):
    '''This network takes in the hidden state output from the MakeSufficientNetwork
    and an edge and predicts how likely that edge is to be needed
    Can compare in batch
    '''
    def __init__(self, opt, common_modules=None):
        super(EdgeComparitorNetwork, self).__init__()
        # TODO
        # Should either create or load all the modules from common_modules
        # TODO

    def forward(self, inputs):
        # Takes in an edge we want to maybe add and the hidden state of the make sufficient network
        # Outputs a sigmoid 0/1 for whether we want to add that edge
        raise NotImplementedError

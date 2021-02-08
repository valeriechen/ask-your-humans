from mazebasev2.lib.knowledge import TripletInfo, Rule, create_rule_from_triplets
from mazebasev2.lib.knowledge import Graph
import pdb

# A knowledge graph class
class KnowledgeSubGraph(Graph):
    def __init__(self, triplets):
        super(KnowledgeSubGraph, self).__init__()
        # Initialize all truth values with unknowns (None)
        # If the user of this class wishes, it can use these values to track
        # the truth prposisitions of the knowledge graph
        self.sufficient = None

        # First get triplets based on the input rule
        self.triplets = triplets

        # Get the rule
        self.rule = Rule(create_rule_from_triplets(self.triplets))

        # Get the rule name and make it the root of the knowledge graph
        rule_name = list(self.triplets)[0][0]
        assert(all([triplet[0] == rule_name for triplet in self.triplets]))
        self.rule_name = rule_name
        data = {}
        data['type'] = 'root'
        self.add_node(rule_name, data)

        # Add pre- and post-conditions
        self.precondition_triplets = set()
        self.postcondition_triplets = set()
        for triplet in self.triplets:
            # Is it a pre- or post- condition
            condition_type = triplet[1]
            if condition_type in TripletInfo.PRE_CONDITION_TYPES:
                self.precondition_triplets.add(triplet)
                concept_node_type = 'precondition_concept'
                condition_node_type = 'precondition'
            else:
                assert(condition_type in TripletInfo.POST_CONDITION_TYPES)
                self.postcondition_triplets.add(triplet)
                concept_node_type = 'postcondition_concept'
                condition_node_type = 'postcondition'

            # Add the concepts as nodes if not already there
            concept = triplet[2]
            if concept not in self.nodes:
                data = {}
                data['type'] = [concept_node_type]
                self.add_node(concept, data)
            elif concept_node_type not in self.nodes[concept].data['type']:
                self.nodes[concept].data['type'].append(concept_node_type)

            # Add the precondition or postcondition node itself
            # It really acts like an edge, but this lets us put additional info in the node
            # And also easily search these
            condition_key = str(triplet)
            data = {}
            data['type'] = condition_node_type
            data['precond_type'] = condition_type
            data['necessary'] = None    # Don't know if it's necessary yet
            self.add_node(condition_key, data)

            # Add all the edges from the triplets
            # Preconditions go from concepts to rule name
            if condition_node_type == 'precondition':
                self.add_edge(concept, condition_key)
                self.add_edge(condition_key, self.rule_name)
            # Postconditions go from rule name to concepts
            else:
                self.add_edge(self.rule_name, condition_key)
                self.add_edge(condition_key, concept)

        # Set what the "main effect" is for this schema 
        self.main_effect = None
        for pc_triplet in self.postcondition_triplets:
            cond_type = pc_triplet[1]
            if cond_type in TripletInfo.MAIN_POST_TYPES:
                assert(self.main_effect is None)    # Main effect must be unique in post condition set
                self.main_effect = pc_triplet[2]

    # Return the rule associated with this knowledge graph
    def get_rule(self):
        return self.rule

    # Returns list of preconditions
    def get_preconditions(self):
        assert(type(self.precondition_triplets) == set)
        preconditions = set([(triplet[1], triplet[2]) for triplet in self.precondition_triplets])
        return preconditions

    # Get post conditions
    def get_postconditions(self):
        assert(type(self.postcondition_triplets) == set)
        postconditions = set([(triplet[1], triplet[2]) for triplet in self.postcondition_triplets])
        return postconditions

    # Return a rule object that is the simplest possible rule
    # (i.e. prune any sufficient but unnecessary components if possible)
    def get_min_suff_rule(self):
        # First get min sufficient subgraph
        min_suff_sub = self.get_min_suff_subgraph()

        # Now just return rule for min sufficient subgraph
        return min_suff_sub.get_rule()

    # Return a new graph object that does not contain any edges known to be not necessary
    def get_min_suff_subgraph(self):
        assert(self.sufficient)

        # Find preconditions that actually are necessary
        min_suff_triplets = []
        for triplet in self.precondition_triplets:
            precond_node = self.nodes[str(triplet)]
            if precond_node.data['necessary'] is None or precond_node.data['necessary']:
                min_suff_triplets.append(triplet)

        # Only include postconditions we know are true (marked as necessary here)
        for triplet in self.postcondition_triplets:
            postcond_node = self.nodes[str(triplet)]
            if postcond_node.data['necessary']:
                min_suff_triplets.append(triplet)

        # Return new KnowledgeGraph with just those triplets now
        assert(len(min_suff_triplets) > 0)
        min_suff_subgraph = KnowledgeSubGraph(min_suff_triplets)
        for triplet in min_suff_triplets:
            min_suff_subgraph.nodes[str(triplet)].data['necessary'] = self.nodes[str(triplet)].data['necessary']
        min_suff_subgraph.sufficient = self.sufficient
        return min_suff_subgraph

    # Check total equality case for preconditions
    def precond_set_equal(self, K_other):
        # Get the precondition sets for each
        return self.get_preconditions() == K_other.get_preconditions()

    # Given another knowledge graph, return whether it is a subset of this graph
    def precond_is_subset(self, K_other):
        # Find set differences between edge lists for self and K_other
        # other - self
        sd_other_self = K_other.get_preconditions() - self.get_preconditions()

        # Only condition where it is true is if other-self is empty
        # Other may not be a proper subset (i.e. set equality)
        if len(sd_other_self) == 0:
            return True
        else:
            return False

    # TODO - maybe this is right?
    # Check if two schemas are "compatible" meaining they are truly the same underlying rule
    def schema_compatible(self, K_other):
        # Check if one is a subset of another
        if self.precond_is_subset(K_other) or K_other.precond_is_subset(self):
            return True

        # Check for mutually incompatible preconditions
        our_unique_preconds = [precond for precond in self.get_preconditions() if precond[0] in TripletInfo.SINGLETON_PRECONDITIONS]  
        other_unique_preconds = [precond for precond in K_other.get_preconditions() if precond[0] in TripletInfo.SINGLETON_PRECONDITIONS]  

        # See if any of those unique preconditions are incompatible
        for our_precond in our_unique_preconds:
            for other_precond in other_unique_preconds:
                if our_precond[0] == other_precond[0] and our_precond[1] != other_precond[1]:
                    return False

        # TODO - make sure there's no possible way that we have mutally incompatible necessary preconditions
        # Easy example is we could require iron pickaxe but could also use gold pickaxe
        # TODO TODO - for now, ignore this. Deal with this when we actually get to this

        # TODO - also maybe keep seperate schemas if they have slightly different effects?

        # If neither, return True
        return True

    # Update whether the graph is sufficient
    # Takes in list of actually satisfied postconditions and updates them in graph or adds them
    # Updates sufficient - sufficient if main condition is satisfied 
    def update_sufficiency(self, postconditions):
        # Add or confirm all the postcondition triplets
        self.sufficient = False
        for postcond in postconditions:
            cond_type = postcond[0]
            concept = postcond[1]
            if self.main_effect == concept:
                self.sufficient = True
            triplet = (self.rule_name, cond_type, concept)
            condition_key = str(triplet)
            # Confirm "necessity" of the post_condition
            if condition_key in self.nodes:
                self.nodes[condition_key].data['necessary'] = True 
            # If not in it already, add it, and possibly connect it to a new/existing effect node
            else:
                # Add postcondition node
                data = {}
                data['type'] = 'postcondition'   
                data['precond_type'] = postcond[0]
                data['necessary'] = True
                self.add_node(condition_key, data)

                # Add postcondition concept node if necessary
                if concept not in self.nodes:
                    data = {}
                    data['type'] = ['postcondition_concept']
                    self.add_node(concept, data)
                elif 'postcondition_concept' not in self.nodes[concept].data['type']:
                    self.nodes[concept].data['type'].append('postcondition_concept')

                # Update connections 
                self.add_edge(self.rule_name, condition_key)
                self.add_edge(condition_key, concept)

    # Given another knowledge graph, updates necessity of this graph
    def update_necessity(self, K_other):
        # If not compatible, don't do anything
        if not self.schema_compatible(K_other):
            return

        # Transfer necessity 
        # Should only transfer these between compatible subgraphs
        self.transfer_necessity(K_other)

        # If this graph is sufficient, also update necessity of dropped preconditions
        if not self.sufficient:
            return

        # Get success
        assert(K_other.sufficient is not None)
        success = K_other.sufficient

        # Next, need to check whether it's actually a subset of this graph
        if self.precond_is_subset(K_other):
            # Get set difference between two graphs preconditions
            preconditions_dropped = self.get_preconditions() - K_other.get_preconditions()
            triplets_dropped = [(self.rule_name, precond[0], precond[1]) for precond in preconditions_dropped]

            # TODO - Perhaps a major assumption here
            # Assume that there are only first order necessity effects
            # i.e. necessity of one edge is independent of the other ones
            # So any edges that we removed
            if success:
                # If it still succeded, any edges in set difference are unnecessary by definition
                for triplet in triplets_dropped:
                    self.nodes[str(triplet)].data['necessary'] = False
            else:
                # We can only check necessary true when the difference is one edge,
                # or we already know the other edges are unnecessary
                
                # If we only dropped one and now it fails
                if len(triplets_dropped) == 1:
                    triplet = triplets_dropped[0]
                    self.nodes[str(triplet)].data['necessary'] = True

    # Copy necessity information from another knowledge graph to this one
    def transfer_necessity(self, K_other):
        # Transfer necessity from K_other if we have them to this graph
        for node_name in K_other.nodes:
            other_node = K_other.nodes[node_name]
            if node_name in self.nodes and other_node.data['type'] == 'precondition' and other_node.data['necessary'] is not None:
                self.nodes[node_name].data['necessary'] = other_node.data['necessary']

    # Merge two graphs that we know are compatible
    def merge_other(self, K_other):
        # Assert postconditions are the same
        assert(self.get_postconditions() == K_other.get_postconditions())

        # Update necessity between the two
        K_other.update_necessity(self)
        self.update_necessity(self)

        # Add any preconditions in K_other that are not in self
        missing_preconds = K_other.get_preconditions() - self.get_preconditions()
        for precond in missing_preconds:
            # Add the triplet
            condition_type = precond[0]
            concept = precond[1]
            triplet = (self.rule_name, condition_type, concept)
            other_triplet = (K_other.rule_name, condition_type, concept)
            assert(condition_type in TripletInfo.PRE_CONDITION_TYPES)
            self.precondition_triplets.add(triplet)
            
            # Add the concepts as nodes if not already there
            concept_node_type = 'precondition_concept'
            condition_node_type = 'precondition'
            if concept not in self.nodes:
                data = {}
                data['type'] = [concept_node_type]
                self.add_node(concept, data)
            elif concept_node_type not in self.nodes[concept].data['type']:
                self.nodes[concept].data['type'].append(concept_node_type)

            # Add the precondition node itself
            # It really acts like an edge, but this lets us put additional info in the node
            # And also easily search these
            condition_key = str(triplet)
            data = {}
            data['type'] = condition_node_type
            data['precond_type'] = condition_type
            data['necessary'] = K_other.nodes[condition_key].data['necessary']
            self.add_node(condition_key, data)

            # Add all the edges from the triplets
            # Preconditions go from concepts to rule name
            self.add_edge(concept, condition_key)
            self.add_edge(condition_key, self.rule_name)


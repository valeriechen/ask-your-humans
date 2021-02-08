from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import shuffle, randint
from six.moves import range
from itertools import product
from . import (
    BaseMazeGame,
    WithWaterAndBlocksMixin,
    RewardOnEndMixin,
    BaseVocabulary,
    AbsoluteLocationVocabulary,
)
from mazebasev2.lib.mazebase.utils import creationutils
from mazebasev2.lib.utils import dictutils
from mazebasev2.lib.knowledge import Graph, CraftingGraph, HierarchyTree, Rule, get_triplets
from mazebasev2.lib.mazebase.utils.mazeutils import populate_kwargs, MazeException, choice
from mazebasev2.lib.mazebase.items import agents
import mazebasev2.lib.mazebase.items as mi
import numpy as np
import random
import abc
import pdb
import pickle


# CraftingAgent
# class CraftingAgent(agents.SingleTileMovable, agents.Crafting, agents.Toggling):
#     pass

# Class which inherits from base maze game
# Takes in a list of rules or schema that tell it the rules of the game
class BaseRuleGame(BaseVocabulary):
   # Init - takes in a json file containing all the rules
    def __init__(self, rules_json, **kwargs):
        #populate_kwargs(self, self.__class__.__properties, kwargs)
        super(BaseRuleGame, self).__init__(**kwargs)

        # Give names to rules (for later comparisons, no real meaning)
        for i, rule in enumerate(rules_json):
            rules_json[i][Rule.RULE_NAME] = 'Rule%d' % (i+1)

        # Create the rules
        self.rules = [Rule(j) for j in rules_json]

# Basics of a rule-based crafting game
# Doesn't include rewards and such
class CraftingGame(BaseRuleGame, WithWaterAndBlocksMixin):
    __properties = dict(
        max_craft_depth=1,  # Max crafting depth that is theoretically possible
        inventory_chance=1, # How likely required items are in inventory versus in space
    )

    # Max number of a single item you could be carrying
    MAX_INVENTORY_COUNT = 5

    MAX_INVENTORY_TOTAL_COUNT = 10

    # Init
    def __init__(self, rules_json, other_items, load_items, **kwargs):
        populate_kwargs(self, self.__class__.__properties, kwargs)
        super(CraftingGame, self).__init__(rules_json, **kwargs)

        # List of bad blocks we can't double generate on
        self.bad_blocks = [mi.Block, mi.ResourceFont, mi.CraftingContainer, mi.CraftingItem, mi.Switch, mi.Door]

        # Create subclass tree - which items inherit from which items
        self.hierarchy_tree = HierarchyTree(self.rules)

        # Create a dictionary of items and their types for each item that appears in our rules
        if other_items is None:
            self.item_dict = {}
        else:
            # We might want to add distractor items that do not appear in the rules
            self.item_dict = other_items.copy()
        self.update_item_dict() # Goes through rules and updates item dict

        # Creates a graph of items and rules that lets us decide item placements
        self.crafting_graph = CraftingGraph(self.rules, self.item_dict, self.max_craft_depth)

        # Check for rule conflicts
        Rule.check_rules_valid(self.rules)

        self.load_items = load_items

    # Override all_possible_features
    # We need to add the unique strings to the vocab
    def all_possible_features(self):
        # Call super to get features
        all_features = super(CraftingGame, self).all_possible_features() 

        # Get all the possible named items in the world
        all_features += list(self.item_dict.keys())

        # Get all the possible inventories
        crafting_items = [key for key in self.item_dict if self.item_dict[key] == mi.CraftingItem.__name__]
        all_combs = list(product(crafting_items, list(range(1, CraftingGame.MAX_INVENTORY_COUNT))))
        all_features += [x[0] + '_count=' + str(x[1]) for x in all_combs]
        return all_features

    def add_vertical_wall(self):
        size = (5, 5)
        dim = choice([0, 1])
        line = randint(1, size[1 - dim] - 2)
        opening = randint(0, size[dim] - 1)
        for i in range(size[dim]):
            if i != opening:
                loc = [line, line]
                loc[dim] = i
                self._add_item(mi.Block(location=loc))

        loc = [line, line]
        loc[dim] = opening

        return loc, dim

    # Reset - create environment
    def _reset(self):
        #super(CraftingGame, self)._reset()

        # Implimented in inheriting classes
        # Sets spawn environment with
        # self.end_items - items we want to make possible to create
        # self.visited_rules - rules we want our agent to be able to use
        # self.distractor_rules - rules we want to make possible to add that use necessary resources for other rules
        # self.extra_items - extra items to spawn in environment that aren't related
        self._set_active_rule_states()

        self.inventory = {}
        self.switch_states = 2

        hole, dim = self.add_vertical_wall()

        # Add the door
        self.door = mi.Door(location=hole,
                            state=choice(range(1, self.switch_states)))
        self._add_item(self.door)

        # Add additional blocks and waters
        #super(CraftingGame, self)._reset()

        # Add agent and switch
        side = choice([-1, 1])

        def mask_func(x, y):
            return side * ((x, y)[1 - dim] - hole[1 - dim]) > 0

        self.isSwitch = random.choice([True, False])

        if self.isSwitch:
            loc = choice(creationutils.empty_locations(
                self, bad_blocks=[mi.Block, mi.Door], mask=mask_func))
            self.sw = mi.Switch(location=loc, nstates=self.switch_states)
            self._add_item(self.sw)
        else:
            loc = choice(creationutils.empty_locations(
                self, bad_blocks=[mi.Block, mi.Door], mask=mask_func))
            self.sw = mi.Key("key",location=loc)
            self._add_item(self.sw)

        loc = choice(creationutils.empty_locations(
            self, bad_blocks=[mi.Block, mi.Door], mask=mask_func))
        self.agent = agents.CraftingAgent(location=loc)
        self.agent.inventory = self.inventory
        self._add_agent(self.agent, "CraftingAgent")

        visited, _ = creationutils.dijkstra(self, loc,
                                            creationutils.agent_movefunc)
        if self.sw.location not in visited:
            raise MazeException("No path to goal")

        # Figure our what we want to spawn
        inventory_items, ground_items, containers, fonts = self.crafting_graph.calculate_items(self.end_items, self.visited_rules, self.distractor_rules, self.extra_items, self.inventory_chance)

        # Create inventory objects
        for item in inventory_items:
            assert(self.item_dict[item] == mi.CraftingItem.__name__)
            item_count = inventory_items[item]
            self._create_item_in_inventory(self, item, item_count)

        # Add containers
        for container_name in containers:
            assert(self.item_dict[container_name] == mi.CraftingContainer.__name__)
            loc = choice(creationutils.empty_locations(self, bad_blocks=self.bad_blocks))
            self._add_item(mi.CraftingContainer(container_name, location=loc))

        # Add fonts
        for font_name in fonts:
            assert(self.item_dict[font_name] == mi.ResourceFont.__name__)
            font_item_count = fonts[font_name]

            # Randomly split into fonts with no more than 10 resources per font
            while font_item_count > 0:
                # Make font of random size
                cur_sz = randint(1, 10)
                if cur_sz > font_item_count:
                    cur_sz = font_item_count

                # Add font to environment
                loc = choice(creationutils.empty_locations(self, bad_blocks=self.bad_blocks))
                self._add_item(mi.ResourceFont(font_name, resource_count=cur_sz, location=loc))
                font_item_count -= cur_sz

        # Add ground items
        for item in ground_items:
            assert(self.item_dict[item] == mi.CraftingItem.__name__)
            item_count = ground_items[item]
            for ind in range(item_count):
                loc = choice(creationutils.empty_locations(self, bad_blocks=self.bad_blocks))
                self._add_item(mi.CraftingItem(item, location=loc))

    # Create an item and add to inventory
    def _create_item_in_inventory(self, target_item, target_item_count=1):
        # For now, let's just treat inventory items as just a dictionary of items
        assert(self.item_dict[target_item] == mi.CraftingItem.__name__)
        if target_item in self.inventory:
            if self.inventory[target_item] < 5:
                self.inventory[target_item] += target_item_count
        else:
            self.inventory[target_item] = target_item_count

    # Destroy an object in the inventory
    def _destroy_from_inventory(self, depleted_item, depleted_count):
        # Check we actually have the item and correct count
        assert(depleted_item in self.inventory)
        assert(self.inventory[depleted_item] >= depleted_count)

        # Remove from inventory
        self.inventory[depleted_item] -= depleted_count

    # Take item out of environment and put in inventory
    def _move_item_to_inventory(self, item_id):
        # Get name of item
        item = self._items[item_id]
        name = item.str_id

        # Destroy item from world
        self._remove_item(item_id)

        # Add to inventory
        if name in self.inventory:
            self.inventory[name] += 1
        else:
            self.inventory[name] = 1

    # Update the dictionary of items with the type
    # TODO - This might not actually be necessary
    def update_item_dict(self):

        # Find items in rules
        craft_items = set()
        fonts = set()
        craft_containers = set()
        # TODO - get rid of string literals for actions at some point
        for rule in self.rules:
            if Rule.ACTION not in rule.rule_dict:
                continue
            elif rule.rule_dict[Rule.ACTION] == 'craft':
                if Rule.LOCATION in rule.rule_dict:
                    craft_containers.add(rule.rule_dict[Rule.LOCATION])
                if Rule.DEPLETED_ITEMS in rule.rule_dict:
                    for item in rule.rule_dict[Rule.DEPLETED_ITEMS]:
                        craft_items.add(item)
                if Rule.NON_DEPLETED_ITEMS in rule.rule_dict:
                    for item in rule.rule_dict[Rule.NON_DEPLETED_ITEMS]:
                        craft_items.add(item)
                for item in rule.rule_dict[Rule.CREATED_ITEMS]:
                    craft_items.add(item)
            elif rule.rule_dict[Rule.ACTION] in ['grab', 'mine', 'chop']:
                if Rule.LOCATION in rule.rule_dict:
                    fonts.add(rule.rule_dict[Rule.LOCATION])
                if Rule.NON_DEPLETED_ITEMS in rule.rule_dict:
                    for item in rule.rule_dict[Rule.NON_DEPLETED_ITEMS]:
                        craft_items.add(item)
                for item in rule.rule_dict[Rule.CREATED_ITEMS]:
                    craft_items.add(item)


        # Update self.item_dict
        for ci in craft_items:
            if ci in self.item_dict:
                assert(self.item_dict[ci] == mi.CraftingItem.__name__)
            else:
                self.item_dict[ci] = mi.CraftingItem.__name__
        for cc in craft_containers:
            if cc in self.item_dict:
                assert(self.item_dict[cc] == mi.CraftingContainer.__name__)
            else:
                self.item_dict[cc] = mi.CraftingContainer.__name__
        for f in fonts:
            if f in self.item_dict:
                assert(self.item_dict[f] == mi.ResourceFont.__name__)
            else:
                self.item_dict[f] = mi.ResourceFont.__name__
        self.item_dict['key'] = 'Key'

    # Add inventory as side information
    def _side_information(self):
        inventory_features = []
        for item in self.inventory:
            feature = item + '_count=' + str(self.inventory[item])
            assert(self.inventory[item] <= CraftingGame.MAX_INVENTORY_COUNT)
            inventory_features.append(feature)

        return super(CraftingGame, self)._side_information() + \
            [[self.FEATURE.INVENTORY] + inventory_features]

    # Abstract method. Inheriting classes need to define for each rule how it wants things to be generated
    @abc.abstractmethod
    def _set_active_rule_states(self):
        pass


# Just encodes the environment and handles randomization
# Doesn't track progress for rewards or anything similar
class BasicKnowledgeGame(CraftingGame):
    # Init function
    def __init__(self, world_knowledge, proposed_knowledge, options, load_items, **kwargs):
        #populate_kwargs(self, self.__class__.__properties, kwargs)

        super(BasicKnowledgeGame, self).__init__(world_knowledge['rules'], world_knowledge['objects'], load_items, max_craft_depth=options['max_craft_depth'], inventory_chance=options['inventory_chance'], **kwargs)
        self.options = options
        if 'spawns' in world_knowledge: 
            self.spawns = world_knowledge['spawns']

        # Create the triplet version of the knowledge (world and proposed)
        self.proposed_rules = [Rule(d) for d in proposed_knowledge]
        self.proposed_knowledge = get_triplets(self.proposed_rules)
        self.world_rules = self.rules
        self.world_knowledge = get_triplets(self.world_rules)
        self.load_items = load_items

    def save(self):
        return pickle.dumps(self)

    # Set goal states
    # In this simplest environment, it's always just the first rule we care about
    # This is the same every reset, so we just do the logic in init
    def _set_active_rule_states(self):
        # Init
        self.end_items = {}
        self.extra_items = {}
        self.visited_rules = []
        self.distractor_rules = []

        # Randomly init environment, depending on options
        # Randomly choose rules and distractor rules
        if self.options['spawn']['mode'] == 'random_rules':
            # Randomly choose the rules to follow
            rule_idx = np.random.choice(len(self.world_rules), self.options['spawn']['num_rules'], replace=False)
            for idx in rule_idx:
                self.visited_rules.append(self.world_rules[idx].rule_dict['name'])
            # TODO - Distractor rules is not implimented right now. Not actually sure what to do with it
            # TODO - Also, it only spawns itesms for world rules, not the proposed rules

            # Choose a random number of other items and add them to spawn
            for _ in range(self.options['spawn']['num_distractor_items']):
                item = random.choice(list(self.item_dict.keys()))
                self.extra_items = dictutils.add_merge_dicts(self.extra_items, {item: 1})
        elif self.options['spawn']['mode'] == 'fixed_spawns':
            # Choose randomly from a list of possible spawns
            # All items to be spawned are specifically enumerated

            if self.load_items != None:
                self.extra_items = self.load_items["extra_items"]
                self.goal = self.load_items["goal"]
                self.recipe = self.load_items["recipe"]
            else:
                self.extra_items = random.choice(self.spawns)

                self.goal = 'temp'
                self.recipe = 'temp'

                for key, value in self.extra_items.items() :
                    if key == 'goal':
                        self.goal = value
                    elif key == 'recipe':
                        self.recipe = value
                    elif key == 'count':
                        self.count = value
        else:
            raise Exception('Have not implemented spawn mode %s' % self.options['spawn']['num_distractor_items'])

    # Reset
    # Set new active rules/items
    # Call super reset for most of work
    def _reset(self):
        self._set_active_rule_states()
        super(BasicKnowledgeGame, self)._reset()

    # Always return false (only finishes on time limit, which is handled in main
    def _finished(self):

        #modified this part to check if the goal statement is satisfied, assume that making one thing first.

        import re

        s = self.goal
        goal = s[s.find("(")+1:s.find(")")]
        goal_split = goal.split("=")
        item = goal_split[0]
        value = goal_split[1]

        #print(item, value, self.inventory)

        if item in self.inventory:
            if str(self.inventory[item]) == value:
                return True

        return False

    # Return 0 always
    def _get_reward(self, agent_id):
        assert(self.agent.id == agent_id)
        return 0

    # Nothing to do here.
    # World and inventory item management should all be handled in action functions
    def _step(self):

        # print(self.agent.location)
        # print(self.sw.location)
        # print(self.door.location)

        # print(self.inventory)

        # depending on the setting, check the locations? 


        #will be handled by action?? s

        #self.door.close()

        # Hook the door up to the switch
        if self.isSwitch:
            if self.sw.state == self.door.state:
               self.door.open()
            else:
               self.door.close()


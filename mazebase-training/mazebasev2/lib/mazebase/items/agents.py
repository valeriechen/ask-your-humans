from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import choice
from itertools import combinations
import sys
import pdb

from mazebasev2.lib.mazebase.utils import mazeutils
import mazebasev2.lib.mazebase.items as mi
from mazebasev2.lib.knowledge import Rule

class Agent(mi.MazeItem):
    '''
    Agents are special items that can perform actions. We use a mix-ins model
    to specify Agent traits. To combine traits, simply subclass both
    Agent classes:

    # This agent can move and drop bread crumbs
    class SingleGoalAgent(mi.SingleTileMovable, mi.BreadcrumbDropping):
        pass

    To make a new agent trait, create the class, subclass from Agent, create
    the actions, and call self._add_action('id', self.__action)

    IMPORTANT: Any attributes defined outside of this module will not be
    featurized. Agents are featurized as a list of what they can 'do'
    '''
    __properties = dict(
        # Speed allows some agents to move faster than others
        speed=1
    )

    def __init__(self, **kwargs):
        mazeutils.populate_kwargs(self, self.__class__.__properties, kwargs)
        super(Agent, self).__init__(**kwargs)

        self.actions = {'pass': self._pass}
        self.PRIO = 100
        self._all_agents = [x[1] for x in
                            mazeutils.all_classes_of(sys.modules[__name__])]

    def _pass(self):
        pass

    def _add_action(self, id, func):
        assert id not in self.actions, "Duplicate action id"
        self.actions[id] = func

    def featurize(self):
        features = list(set(self.__get_all_superclasses(self.__class__)))
        return features

    def __get_all_superclasses(self, cls):
        all_superclasses = []
        for superclass in cls.__bases__:
            if superclass in self._all_agents:
                all_superclasses.append(superclass.__name__)
            all_superclasses.extend(self.__get_all_superclasses(superclass))
        return all_superclasses

    def _get_display_symbol(self):
        return (u' A ', None, None, None)

class NPC(Agent):
    ''' NPC Agents cannot be controlled by the player and moves randomly '''
    def get_npc_action(self):
        return (self.id, choice(self.actions))

class SingleTileMovable(Agent):
    ''' Can move up, down, left, and right 1 tile per turn '''
    def __init__(self, **kwargs):
        super(SingleTileMovable, self).__init__(**kwargs)
        self._add_action("up", self.__up)
        self._add_action("down", self.__down)
        self._add_action("left", self.__left)
        self._add_action("right", self.__right)

    def __dmove(self, dx, dy):
        x, y = self.location
        nloc = x + dx, y + dy
        # Cannot walk into blocks, agents, or closed doors
        if (self.game._tile_get_block(nloc, mi.Block) is None and
                self.game._tile_get_block(nloc, Agent) is None and
                (not self.game._tile_get_block(nloc, mi.Door) or
                 self.game._tile_get_block(nloc, mi.Door).isopen)):
            self.game._move_item(self.id, location=nloc)

    def __up(self):
        self.__dmove(0, 1)

    def __down(self):
        self.__dmove(0, -1)

    def __left(self):
        self.__dmove(-1, 0)

    def __right(self):
        self.__dmove(1, 0)

class BreadcrumbDropping(Agent):
    ''' Can drop breadcrumbs as an action '''
    def __init__(self, **kwargs):
        super(BreadcrumbDropping, self).__init__(**kwargs)
        self._add_action("breadcrumb", self.__drop_crumb)

    def __drop_crumb(self):
        if self.game._tile_get_block(self.location, mi.Breadcrumb) is None:
            self.game._add_item(mi.Breadcrumb(location=self.location))

class Pushing(Agent):
    '''
    Can push in the 4 cardinal directions. Pushing moves Pushable objects
    in one of four directions if there's no collision.
    '''
    def __init__(self, **kwargs):
        super(Pushing, self).__init__(**kwargs)
        self._add_action("push_up", self.__push_up)
        self._add_action("push_down", self.__push_down)
        self._add_action("push_left", self.__push_left)
        self._add_action("push_right", self.__push_right)

    def __dpush(self, dx, dy):
        x, y = self.location
        tx, ty = x + dx, y + dy
        nx, ny = tx + dx, ty + dy

        # Cannot push into other blocks or agents
        block = self.game._tile_get_block((tx, ty), mi.Pushable)
        if (block is not None and
                self.game._tile_get_block((nx, ny), Agent) is None and
                self.game._tile_get_block((nx, ny), mi.Block) is None):
            self.game._move_item(block.id, location=(nx, ny))

    def __push_up(self):
        self.__dpush(0, 1)

    def __push_down(self):
        self.__dpush(0, -1)

    def __push_left(self):
        self.__dpush(-1, 0)

    def __push_right(self):
        self.__dpush(1, 0)

class Toggling(Agent):
    ''' Can toggle on current space '''
    def __init__(self, **kwargs):
        super(Toggling, self).__init__(**kwargs)
        self._add_action("toggle_switch", self.__toggle)

    def __toggle(self):
        x, y = self.location
        switch = self.game._tile_get_block((x, y), mi.Switch)
        if switch is not None:
            switch.toggle()
        else:
            #CHECK THE LOCATION, also get location of the door. 
            door_loc = self.game.door.location
            if 'key' in self.inventory:
                if (abs(door_loc[0] - x) == 1 and door_loc[1] == y) or (abs(door_loc[1] - y) == 1 and door_loc[0] == x):
                    self.game.door.open()


class Crafting(Agent):
    ''' Can collect resources from fonts and craft items at special crafting locations '''
    def __init__(self, **kwargs):
        super(Crafting, self).__init__(**kwargs)
        self._add_action("grab", self.__grab)
        self._add_action("mine", self.__mine)
        self._add_action("chop", self.__chop)
        self._add_action("craft", self.__craft)
        self.last_action = None
        self.location_item = None

    # Grab items (or from font) on agent's square and add to inventory
    def __grab(self):

        # print("INVENTORY")
        # for item in self.inventory:
        #     feature = item + '_count=' + str(self.inventory[item])
        #     print(feature)

        x, y = self.location
        
        # See if the font lets you do a grab
        font = self.game._tile_get_block((x, y), mi.ResourceFont)
        if font is not None:
            self.__usefont('grab')
        else:
            # Grab all the items on the space
            item = self.game._tile_get_block((x, y), mi.CraftingItem)        
            while item is not None:
                self.game._move_item_to_inventory(item.id)
                item = self.game._tile_get_block((x, y), mi.CraftingItem)    
        self.last_action = 'grab'           

    # Mine font on agent's square and add item to inventory
    def __mine(self):
        self.__usefont('mine')
        self.last_action = 'mine'

    # Chop font on agent's square and add item to inventory
    def __chop(self):
        self.__usefont('chop')
        self.last_action = 'chop'

    # General font depletion action
    # grab, mine and chop actions all call this with argument action='grab' for example
    def __usefont(self, action):

        x, y = self.location
               
        # Next see if the font lets you chop
        font = self.game._tile_get_block((x, y), mi.ResourceFont)

        # Save inputs to rule check for checking in _step()
        self.location_item = font

        ## CHECK INVENTORY SIZE HERE FIRST:
        total_count = 0
        for key in self.inventory:

            if self.inventory[key] == 5:
                return
            total_count = total_count + self.inventory[key]

        if total_count == 10:
            return

        # Check rules
        if font is not None:
            # Find relevant rule(s)
            satisfied_rules = []
            for rule in self.game.rules:
                if rule.check_rule_satisfied(action, font, self.inventory):
                    satisfied_rules.append(rule)

            # Check rules to see if we are using the correct action and have all pre-requisite items
            if len(satisfied_rules) > 1:
                # TODO - this doesn't make sense for resource fonts to have more than one relevant rule
                raise Exception("We shouldn't have more than one satisfied rule right now")
            elif len(satisfied_rules) == 1:
                rule = satisfied_rules[0]

                # Should not be any depleted items in resource gathering
                #assert(Rule.DEPLETED_ITEMS not in rule.rule_dict)
           
                # Deplete font by one and add item to inventory
                font.deplete()

                # If font is empty, remove it
                if font.resource_count == 0:
                    self.game._remove_item(font.id)
               
                # Create target items and add to inventory
                target_items = rule.rule_dict[Rule.CREATED_ITEMS]
                for target_item in target_items:
                    target_item_count = target_items[target_item]
                    self.game._create_item_in_inventory(target_item, target_item_count)

    # Craft item using the crafting container at the agent's location
    def __craft(self):
        x, y = self.location

        # Get crafting container at location
        craft_container = self.game._tile_get_block((x, y), mi.CraftingContainer)

        # Save inputs to rule check for checking in _step()
        self.location_item = craft_container

        ## CHECK INVENTORY SIZE HERE FIRST:
        total_count = 0
        for key in self.inventory:

            total_count = total_count + self.inventory[key]

        if total_count == 10:
            return


        # Check and resolve rules
        if craft_container is not None:
            # Find relevant rule(s)
            satisfied_rules = []
            for rule in self.game.rules:
                if rule.check_rule_satisfied('craft', craft_container, self.inventory):
                    satisfied_rules.append(rule)

            # Check and make sure that rules do not consume common resource
            if len(satisfied_rules) > 1:
                depleted_union = set()
                sum_set_els = 0
                for rule in satisfied_rules:
                    sum_set_els += len(rule.rule_dict[Rule.DEPLETED_ITEMS])
                    depleted_union |= set(rule.rule_dict[Rule.DEPLETED_ITEMS].keys())
                assert(len(depleted_union) == sum_set_els)
                # TODO - this might need to be revisited later

            # Go through each rule and do the crafting
            for rule in satisfied_rules:
                depleted_items = rule.rule_dict[Rule.DEPLETED_ITEMS]
                target_items = rule.rule_dict[Rule.CREATED_ITEMS]

                # Destroy depleted items
                for depleted_item in depleted_items:
                    depleted_count = depleted_items[depleted_item]
                    self.game._destroy_from_inventory(depleted_item, depleted_count)

                # Create target items and add to inventory
                for target_item in target_items:
                    target_item_count = target_items[target_item]
                    self.game._create_item_in_inventory(target_item, target_item_count)

        # Set last action to craft
        self.last_action = 'craft'

class CraftingAgent(Agent):

    def __init__(self, **kwargs):
        super(CraftingAgent, self).__init__(**kwargs)
        self._add_action("up", self.__up)
        self._add_action("down", self.__down)
        self._add_action("left", self.__left)
        self._add_action("right", self.__right)
        self._add_action("grab", self.__grab)
        self._add_action("mine", self.__mine)
        self._add_action("chop", self.__chop)
        self._add_action("craft", self.__craft)
        self._add_action("toggle_switch", self.__toggle)
        self.last_action = None
        self.location_item = None

    def __dmove(self, dx, dy):
        x, y = self.location
        nloc = x + dx, y + dy
        # Cannot walk into blocks, agents, or closed doors
        if (self.game._tile_get_block(nloc, mi.Block) is None and
                self.game._tile_get_block(nloc, Agent) is None and
                (not self.game._tile_get_block(nloc, mi.Door) or
                 self.game._tile_get_block(nloc, mi.Door).isopen)):
            self.game._move_item(self.id, location=nloc)

    def __up(self):
        self.__dmove(0, 1)

    def __down(self):
        self.__dmove(0, -1)

    def __left(self):
        self.__dmove(-1, 0)

    def __right(self):
        self.__dmove(1, 0)

    # Grab items (or from font) on agent's square and add to inventory
    def __grab(self):

        # print("INVENTORY")
        # for item in self.inventory:
        #     feature = item + '_count=' + str(self.inventory[item])
        #     print(feature)

        x, y = self.location
        
        # See if the font lets you do a grab
        font = self.game._tile_get_block((x, y), mi.ResourceFont)
        if font is not None:
            self.__usefont('grab')
        else:
            # Grab all the items on the space
            item = self.game._tile_get_block((x, y), mi.CraftingItem)        
            while item is not None:
                self.game._move_item_to_inventory(item.id)
                item = self.game._tile_get_block((x, y), mi.CraftingItem)    
        self.last_action = 'grab'           

    # Mine font on agent's square and add item to inventory
    def __mine(self):
        self.__usefont('mine')
        self.last_action = 'mine'

    # Chop font on agent's square and add item to inventory
    def __chop(self):
        self.__usefont('chop')
        self.last_action = 'chop'

    # General font depletion action
    # grab, mine and chop actions all call this with argument action='grab' for example
    def __usefont(self, action):

        x, y = self.location
               
        # Next see if the font lets you chop
        font = self.game._tile_get_block((x, y), mi.ResourceFont)

        # Save inputs to rule check for checking in _step()
        self.location_item = font

        ## CHECK INVENTORY SIZE HERE FIRST:
        total_count = 0
        for key in self.inventory:

            if self.inventory[key] == 5:
                return
            total_count = total_count + self.inventory[key]

        if total_count == 10:
            return

        # Check rules
        if font is not None:
            # Find relevant rule(s)
            satisfied_rules = []
            for rule in self.game.rules:
                if rule.check_rule_satisfied(action, font, self.inventory):
                    satisfied_rules.append(rule)

            # Check rules to see if we are using the correct action and have all pre-requisite items
            if len(satisfied_rules) > 1:
                # TODO - this doesn't make sense for resource fonts to have more than one relevant rule
                raise Exception("We shouldn't have more than one satisfied rule right now")
            elif len(satisfied_rules) == 1:
                rule = satisfied_rules[0]

                # Should not be any depleted items in resource gathering
                #assert(Rule.DEPLETED_ITEMS not in rule.rule_dict)
           
                # Deplete font by one and add item to inventory
                font.deplete()

                # If font is empty, remove it
                if font.resource_count == 0:
                    self.game._remove_item(font.id)
               
                # Create target items and add to inventory
                target_items = rule.rule_dict[Rule.CREATED_ITEMS]
                for target_item in target_items:
                    target_item_count = target_items[target_item]
                    self.game._create_item_in_inventory(target_item, target_item_count)

    # Craft item using the crafting container at the agent's location
    def __craft(self):
        x, y = self.location

        # Get crafting container at location
        craft_container = self.game._tile_get_block((x, y), mi.CraftingContainer)

        # Save inputs to rule check for checking in _step()
        self.location_item = craft_container

        ## CHECK INVENTORY SIZE HERE FIRST:
        total_count = 0
        for key in self.inventory:

            total_count = total_count + self.inventory[key]

        if total_count == 10:
            return


        # Check and resolve rules
        if craft_container is not None:
            # Find relevant rule(s)
            satisfied_rules = []
            for rule in self.game.rules:
                if rule.check_rule_satisfied('craft', craft_container, self.inventory):
                    satisfied_rules.append(rule)

            # Check and make sure that rules do not consume common resource
            if len(satisfied_rules) > 1:
                depleted_union = set()
                sum_set_els = 0
                for rule in satisfied_rules:
                    sum_set_els += len(rule.rule_dict[Rule.DEPLETED_ITEMS])
                    depleted_union |= set(rule.rule_dict[Rule.DEPLETED_ITEMS].keys())
                assert(len(depleted_union) == sum_set_els)
                # TODO - this might need to be revisited later

            # Go through each rule and do the crafting
            for rule in satisfied_rules:
                depleted_items = rule.rule_dict[Rule.DEPLETED_ITEMS]
                target_items = rule.rule_dict[Rule.CREATED_ITEMS]

                # Destroy depleted items
                for depleted_item in depleted_items:
                    depleted_count = depleted_items[depleted_item]
                    self.game._destroy_from_inventory(depleted_item, depleted_count)

                # Create target items and add to inventory
                for target_item in target_items:
                    target_item_count = target_items[target_item]
                    self.game._create_item_in_inventory(target_item, target_item_count)

        # Set last action to craft
        self.last_action = 'craft'

    def __toggle(self):
        x, y = self.location
        switch = self.game._tile_get_block((x, y), mi.Switch)
        if switch is not None:
            switch.toggle()
        else:
            #CHECK THE LOCATION, also get location of the door. 
            door_loc = self.game.door.location
            if 'key' in self.inventory:
                if (abs(door_loc[0] - x) == 1 and door_loc[1] == y) or (abs(door_loc[1] - y) == 1 and door_loc[0] == x):
                    self.game.door.open()

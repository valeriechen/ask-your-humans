from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from mazebasev2.lib.mazebase.items import MazeItem
from numpy import inf


# Class that adds the state feature to all_features representation
class HasStatesMixin(object):
    _MAX_STATES = 10
    STATE_FEATURE = ["state{0}".format(i) for i in range(_MAX_STATES)]

    @classmethod
    def all_features(cls):
        return super(HasStatesMixin, cls).all_features() + cls.STATE_FEATURE

# Item which is an un-passable block that prevent the agent from moving to that space
class Block(MazeItem):
    def __init__(self, **kwargs):
        super(Block, self).__init__(passable=False, **kwargs)

    def _get_display_symbol(self):
        return (None, None, 'on_white', None)

# Item which in many games which gives negative reward for being in
class Water(MazeItem):
    def __init__(self, **kwargs):
        super(Water, self).__init__(**kwargs)
        self.PRIO = -100

    def _get_display_symbol(self):
        return (None, None, 'on_blue', None)

# Item which I think just marks the corner of the environment
# Used for some logic puzzles
class Corner(MazeItem):
    def __init__(self, **kwargs):
        super(Corner, self).__init__(**kwargs)

    def _get_display_symbol(self):
        return (u'   ', None, None, None)

# Special item which is a "goal" that we want the agent to go to in some games
# Some games have multiple of these and rules about which order to go to goals
class Goal(MazeItem):
    __MAX_GOAL_IDS = 10

    def __init__(self, id=0, **kwargs):
        super(Goal, self).__init__(**kwargs)
        self.goal_id = id
        assert self.goal_id < self.__MAX_GOAL_IDS,\
            "cannot create goal with id >{0}".format(
                self.__MAX_GOAL_IDS)

    def _get_display_symbol(self):
        return (u'*{0}*'.format(self.goal_id), 'red', None, None)

    def featurize(self):
        return super(Goal, self).featurize() +\
            ["goal_id" + str(self.goal_id)]

    @classmethod
    def all_features(cls):
        return super(Goal, cls).all_features() +\
            ["goal_id" + str(k) for k in range(cls.__MAX_GOAL_IDS)]

# Item the agend can drop on the ground used for some games
class Breadcrumb(MazeItem):
    def __init__(self, **kwargs):
        super(Breadcrumb, self).__init__(**kwargs)
        self.PRIO = -50

    def _get_display_symbol(self):
        return (u' . ', None, None, None)

# Special type of block that the agent is able to push
class Pushable(Block):
    def __init__(self, **kwargs):
        super(Pushable, self).__init__(**kwargs)

    def _get_display_symbol(self):
        return (None, None, 'on_green', None)

# Type of item which the agent can use the toggle action on to change the state
# Changing the state can effect other items such as doors
class Switch(HasStatesMixin, MazeItem):
    def __init__(self, start_state=0, nstates=2, **kwargs):
        super(Switch, self).__init__(**kwargs)
        self.state = start_state
        self.nstates = nstates
        assert self.nstates < HasStatesMixin._MAX_STATES,\
            "cannot create switches with >{0} states".format(
                self.__MAX_SWITCH_STATES)

    def _get_display_symbol(self):
        return (str(self.state).rjust(3), 'cyan', None, None)

    def toggle(self):
        self.state = (self.state + 1) % self.nstates

    def featurize(self):
        return super(Switch, self).featurize() +\
            [self.STATE_FEATURE[self.state]]

# Item which has states open and closed. When open, is passible, when closed is blocked
class Door(HasStatesMixin, MazeItem):
    def __init__(self, open=False, state=0, **kwargs):
        super(Door, self).__init__(**kwargs)
        self.isopen = open
        self.state = state

    def _get_display_symbol(self):
        return (None if self.isopen else u'\u2588{0}\u2588'.format(self.state),
                None, None, None)

    def open(self):
        self.isopen = True

    def close(self):
        self.isopen = False

    def toggle(self):
        self.isopen = not self.isopen

    def featurize(self):
        return super(Door, self).featurize() + \
            ["open" if self.isopen else "closed", self.STATE_FEATURE[self.state]]

    @classmethod
    def all_features(cls):
        return super(Door, cls).all_features() +\
            ["open", "closed"]

# This abstract class changes featurize to also print the unique str_id
class NamedItem(MazeItem):
    def __init__(self, str_id, **kwargs):
        super(NamedItem, self).__init__(**kwargs)
 
        # String id
        self.str_id = str_id

    def featurize(self):
        return super(NamedItem, self).featurize() + \
            [self.str_id]
        # TODO - this might need to be rethought depending on the actual state representation we end up with

    # TODO - I don't think we need all_features here. Listing all possible str_ids has to be done outside of here

# Item that contains some amount of a resource (is depletable)
class LimitedResourceFont(NamedItem):
    _MAX_COUNT = 10
    STATE_FEATURE = ["count{0}".format(i) for i in range(_MAX_COUNT)]

    def __init__(self, str_id, resource_count=1, **kwargs):
        super(LimitedResourceFont, self).__init__(str_id, **kwargs)

        # Count of how much resource is in the font
        self.resource_count = resource_count

    # TODO - I chose this arbitrarily. Make sure it doesn't overlap with anything
    def _get_display_symbol(self):
        return (str(self.resource_count).rjust(3), 'yellow', None, None)

    # Deplete the resource (use up by one)
    def deplete(self):
        assert(self.resource_count >= 1)
        self.resource_count -= 1

    # Add number left as feature
    def featurize(self):
        return super(LimitedResourceFont, self).featurize() +\
            [self.STATE_FEATURE[self.resource_count]]

    @classmethod
    def all_features(cls):
        return super(LimitedResourceFont, cls).all_features() + cls.STATE_FEATURE

# Item that contains some amount of a resource (is not depletable)
class ResourceFont(NamedItem):
    def __init__(self, str_id, **kwargs):
        super(ResourceFont, self).__init__(str_id, **kwargs)
        self.resource_count = float(inf)

    def _get_display_symbol(self):
        return (" F ", 'yellow', None, None)

    # Deplete the resource (does nothing since it's unlimited)
    def deplete(self):
        pass 

    # Add number left as feature
    def featurize(self):
        return super(ResourceFont, self).featurize() 

    @classmethod
    def all_features(cls):
        return super(ResourceFont, cls).all_features()

# Item that is used for converting items (workbench, furnace, etc)
class CraftingContainer(NamedItem):
    def __init__(self, str_id, **kwargs):
        super(CraftingContainer, self).__init__(str_id, **kwargs) 

    # TODO - I chose this arbitrarily. Make sure it doesn't overlap with anything
    def _get_display_symbol(self):
        return (" C ", 'green', None, None)

# Crafting items, from raw materials to final items
class CraftingItem(NamedItem):
    def __init__(self, str_id, **kwargs):
        super(CraftingItem, self).__init__(str_id, **kwargs)

    # TODO - I chose this arbitrarily. Make sure it doesn't overlap with anything
    # TODO - not sure if it should even have a display. Maybe if we need to drop it
    def _get_display_symbol(self):
        return (" I ", 'yellow', None, None)

class Key(CraftingItem):
    def __init__(self, str_id, **kwargs):
        super(Key, self).__init__(str_id, **kwargs)

    # TODO - I chose this arbitrarily. Make sure it doesn't overlap with anything
    # TODO - not sure if it should even have a display. Maybe if we need to drop it
    def _get_display_symbol(self):
        return (" K ", 'yellow', None, None)


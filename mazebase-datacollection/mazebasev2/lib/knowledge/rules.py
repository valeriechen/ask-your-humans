from mazebasev2.lib.utils import dictutils
import pdb

# This class defines all the constants we use for triplet form of knowledge
# TODO - add more conditions here
class TripletInfo(object):
    # Post condition triplet types
    MAX_CREATED_ITEMS = 16
    CREATES_ITEM_X = 'creates_item_%d'
    CREATES_ITEM_POST_CONDITIONS = ['creates_item_%d' % (i+1) for i in range(MAX_CREATED_ITEMS)]
    MAX_CONSUMED_ITEMS = 9
    DESTROYS_ITEM_X = 'destroys_item_%d'
    DESTROYS_ITEM_POST_CONDITIONS = ['destroys_item_%d' % (i+1) for i in range(MAX_CONSUMED_ITEMS)]
    POST_CONDITION_TYPES = CREATES_ITEM_POST_CONDITIONS + DESTROYS_ITEM_POST_CONDITIONS

    # List the post condition types which are essential to the schemas
    MAIN_POST_TYPES = CREATES_ITEM_POST_CONDITIONS

    # Pre-condition triplet types
    REQUIRES_ACTION = 'requires_action'
    REQUIRES_LOCATION = 'requires_being_at'
    MAX_REQUIRED_ITEMS = 9
    REQUIRES_ITEM_X = 'requires_item_%d'
    REQUIRES_ITEM_PRECONDITIONS = ['requires_item_%d' % (i+1) for i in range(MAX_REQUIRED_ITEMS)]
    PRE_CONDITION_TYPES = [REQUIRES_ACTION, REQUIRES_LOCATION] + REQUIRES_ITEM_PRECONDITIONS

    # List of preconditions that can only appear once, so can mutually exculde each other
    SINGLETON_PRECONDITIONS = [REQUIRES_ACTION, REQUIRES_LOCATION]

    # All triplets
    TRIPLET_TYPES = POST_CONDITION_TYPES + PRE_CONDITION_TYPES

# This is the rule class
# Contains the information needed to verify rules
class Rule(object):
    # TODO - add more conditions here
    # Define a bunch of constants
    # Item pre/post conditions
    NON_DEPLETED_ITEMS = 'non_depleted_items'
    DEPLETED_ITEMS = 'depleted_items'
    CREATED_ITEMS = 'created_items'
    RULE_NAME = 'rule_name'
    ACTION = 'required_action'
    LOCATION = 'required_location'
    SPAWN = 'spawn_ind'
    RULE_DICT_KEYS = [NON_DEPLETED_ITEMS, DEPLETED_ITEMS, CREATED_ITEMS, RULE_NAME, ACTION, LOCATION, SPAWN]

    @classmethod
    # Checks that the crafting rules cannot be satisfied simultanously
    def check_rules_valid(cls, rules):
        # Right now this just checks that the crafting rules can't possibly happen simultanously
        for i, rule1 in enumerate(rules):
            # Ignore non-crafting rules
            if Rule.ACTION not in rule1.rule_dict or rule1.rule_dict[Rule.ACTION] != 'craft':
                continue

            # Go through each pair of rules
            for j, rule2 in enumerate(rules):
                if j <= i:
                    continue
                # Ignore non-crafting rules
                if Rule.ACTION not in rule1.rule_dict or rule1.rule_dict[Rule.ACTION] != 'craft':
                    continue

                # If different location required, no problems
                # Assumes crafting rules all have one required location
                if rule1.rule_dict[Rule.LOCATION] != rule2.rule_dict[Rule.LOCATION]:
                    continue

                # If it's same action at the same location, there's a chance both rules can be satisfied
                # So we will not allow this
                created1 = list(rule1.rule_dict[Rule.CREATED_ITEMS].keys())[0]
                created2 = list(rule2.rule_dict[Rule.CREATED_ITEMS].keys())[0]
                raise Exception("We have conflicting rules for rule %d creating %s and rule %d creating %s" % (i, created1, j, created2))

    def __init__(self, rule_dict):
        self.rule_dict = rule_dict

        # Make sure everthing in rule_dict is known 
        for key in rule_dict:
            assert(key in Rule.RULE_DICT_KEYS)

        # Make sure we have all required elements
        assert(Rule.ACTION in rule_dict)
        assert(Rule.RULE_NAME in rule_dict)

    # Update rule name in rule dictionary
    def update_rule_name(self, new_name):
        self.rule_dict[Rule.RULE_NAME] = new_name

    # Get required depleted items
    def get_required_depleted_items(self, required_items=None):
        if required_items is None:
            required_items = {}
        if Rule.DEPLETED_ITEMS in self.rule_dict:
            required_items = dictutils.add_merge_dicts(required_items, self.rule_dict[Rule.DEPLETED_ITEMS])
        return required_items

    # Get required non-depleted items
    def get_required_nondepleted_items(self, required_items=None):
        if required_items is None:
            required_items = {}
        if Rule.NON_DEPLETED_ITEMS in self.rule_dict:
            required_items = dictutils.add_merge_dicts(required_items, self.rule_dict[Rule.NON_DEPLETED_ITEMS])
        return required_items

    # Get combined required item dict (so we can easily verify inventory satisfies requirements)
    def get_required_items(self):
        required_items = self.get_required_depleted_items()
        required_items = self.get_required_nondepleted_items(required_items)
        return required_items

    # Get the required action
    def get_required_action(self):
        return self.rule_dict[Rule.ACTION]

    # Assumes there's only one relevant item for this action at the location
    def check_rule_satisfied(self, action, location_item, inventory):
        # Make sure action satisfies rule
        if self.rule_dict[Rule.ACTION] != action:
            return False

        # Make sure location satisfies rule
        if Rule.LOCATION in self.rule_dict and self.rule_dict[Rule.LOCATION] != location_item.str_id:
            return False

        # Make sure inventory satisfies rule
        required_items = self.get_required_items()
        for item in required_items:
            if item not in inventory:
                return False
            if inventory[item] < required_items[item]:
                return False

        # All is satisfied, return True
        return True

    # Convert rule into triplets
    # TODO - add more conditions here
    # TODO - add hierarchy here
    def create_triplet_corresp(self):
        triplets = []

        # Get the name of the rule (ie. craft_pickaxe)
        rule_name = self.rule_dict[Rule.RULE_NAME]

        # Get post-conditions
        # Item creation
        if Rule.CREATED_ITEMS in self.rule_dict:
            for created_item in self.rule_dict[Rule.CREATED_ITEMS]:
                item_count = self.rule_dict[Rule.CREATED_ITEMS][created_item]
                assert(item_count <= TripletInfo.MAX_CREATED_ITEMS)
                triplets.append((rule_name, TripletInfo.CREATES_ITEM_X % item_count, created_item))

        # Item destruction
        depleted_items = self.get_required_depleted_items()
        for item in depleted_items:
            depleted_count = depleted_items[item]
            assert(depleted_count <= TripletInfo.MAX_CONSUMED_ITEMS)
            triplets.append((rule_name, TripletInfo.DESTROYS_ITEM_X % depleted_count, item))

        # Add pre-conditions 
        # Add action rule
        assert(Rule.ACTION in self.rule_dict)
        if Rule.ACTION in self.rule_dict:
            action_rule = (rule_name, TripletInfo.REQUIRES_ACTION, self.rule_dict[Rule.ACTION]) 
            triplets.append(action_rule)

        # Add required location rule
        if Rule.LOCATION in self.rule_dict:
            location_rule = (rule_name, TripletInfo.REQUIRES_LOCATION, self.rule_dict[Rule.LOCATION])
            triplets.append(location_rule)

        # Add required item rules
        # For this, depleted and nondepleted are identical actually
        required_items = self.get_required_items()
        for item in required_items:
            item_count = required_items[item]
            assert(item_count < TripletInfo.MAX_REQUIRED_ITEMS)
            item_rule = (rule_name, TripletInfo.REQUIRES_ITEM_X % item_count, item)
            triplets.append(item_rule)

        # Returns a set of triplets (order is irrelevant, want to compare)
        return set(triplets)

    # Tell if it's a crafting rule
    # TODO - remove the string literals
    def is_crafting_rule(self):
        return 'required_action' in self.rule_dict and self.rule_dict[Rule.ACTION] in ['mine', 'grab', 'mine', 'chop']

# Helper method
# Convert triplets into rule
# Basically the inverse of create_triplet_corresp
# TODO - add more conditions here
def create_rule_from_triplets(triplets):
    # Create rule dict that can be passed to Rule init from triplet input
    rule_dict = {}

    # Get the name of the rule
    rule_name = list(triplets)[0][0]
    assert(all([t[0] == rule_name for t in triplets]))
    rule_dict[Rule.RULE_NAME] = rule_name

    # Go through triplets
    for triplet in triplets:
        edge_type = triplet[1]
        assert(edge_type in TripletInfo.TRIPLET_TYPES)

        # Get created item post-condition
        if edge_type in TripletInfo.CREATES_ITEM_POST_CONDITIONS:
            item_count = int(edge_type.split('_')[-1])
            item = triplet[2]

            # Update created item dictionary
            if Rule.CREATED_ITEMS not in rule_dict:
                rule_dict[Rule.CREATED_ITEMS] = {}
            assert(item not in rule_dict[Rule.CREATED_ITEMS])
            rule_dict[Rule.CREATED_ITEMS][item] = item_count

        # Get destroy item post conditions
        elif edge_type in TripletInfo.DESTROYS_ITEM_POST_CONDITIONS:
            depleted_count = int(edge_type.split('_')[-1])
            depleted_item = triplet[2]
            
            # Update depleted item dictionary
            if Rule.DEPLETED_ITEMS not in rule_dict:
                rule_dict[Rule.DEPLETED_ITEMS] = {}
            assert(depleted_item not in rule_dict[Rule.DEPLETED_ITEMS])
            rule_dict[Rule.DEPLETED_ITEMS][depleted_item] = depleted_count    
                    
        # Get item required pre-conditions
        # Add all required items as nondepleted, and later remove the ones that are actually in the depleted item list
        elif edge_type in TripletInfo.REQUIRES_ITEM_PRECONDITIONS:
            item_count = int(edge_type.split('_')[-1])
            item = triplet[2]

            # Update non-depleted item dictionary
            if Rule.NON_DEPLETED_ITEMS not in rule_dict:
                rule_dict[Rule.NON_DEPLETED_ITEMS] = {}
            assert(item not in rule_dict[Rule.NON_DEPLETED_ITEMS])
            rule_dict[Rule.NON_DEPLETED_ITEMS][item] = item_count

        # Update action required
        elif edge_type == TripletInfo.REQUIRES_ACTION:
            rule_dict[Rule.ACTION] = triplet[2]

        # Add required location
        elif edge_type == TripletInfo.REQUIRES_LOCATION:
            rule_dict[Rule.LOCATION] = triplet[2]
 
        # Else shouldn't happen
        else:
            raise Exception("Unkown edge type %s" % edge_type)

    # Make NON-DEPLETED to be REQUIRES - DEPLETED
    if Rule.NON_DEPLETED_ITEMS in rule_dict and Rule.DEPLETED_ITEMS in rule_dict:
        non_depleted_items = dictutils.diff_dicts(rule_dict[Rule.NON_DEPLETED_ITEMS], rule_dict[Rule.DEPLETED_ITEMS], remove_zeros=True)
        assert(all([non_depleted_items[key] > 0 for key in non_depleted_items]))
        rule_dict[Rule.NON_DEPLETED_ITEMS] = non_depleted_items

    return rule_dict

# Creates a set of triplets given a list of rules
def get_triplets(rule_list):
    triplets = set([item for rule_triplet in [rule.create_triplet_corresp() for rule in rule_list] for item in rule_triplet])
    return triplets

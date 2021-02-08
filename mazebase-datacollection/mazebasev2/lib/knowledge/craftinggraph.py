import sys
from mazebasev2.lib.utils import dictutils
from mazebasev2.lib.knowledge import Graph, Rule
import random
import pdb

# TODO - remove the literals
# Maybe will need updating with multiple post-conditions!
# TODO - definitely will
# Class that deals with graph searches through rules
class CraftingGraph(Graph):
    def __init__(self, rules, item_dict, max_craft_depth):
        super(CraftingGraph, self).__init__()
        self.item_dict = item_dict
        self.max_craft_depth = max_craft_depth
        self.rules = rules

        # Create nodes for rules
        for rule in rules:
            # Only use crafting rules
            if not rule.is_crafting_rule():
                continue

            # Create node
            name = rule.rule_dict[Rule.RULE_NAME]
            data = {}
            data['type'] = 'rule'
            data['rule'] = rule
            self.add_node(name, data)

        # Create nodes for items
        for item in item_dict:
            if item_dict[item] != 'CraftingItem':
                continue

            # Create node
            name = item
            data = {}
            data['type'] = item
            self.add_node(name, data)

        # Go through each rule and update connections
        for rule in rules:
            rule_name = rule.rule_dict[Rule.RULE_NAME]
            if rule_name not in self.nodes:
                continue

            # Add edges from required items to rules
            required_items = list(rule.get_required_items().keys())
            for req_item in required_items:
                self.add_edge(req_item, rule_name)

            # Add edge from rule node to created items
            # TODO - again, assuming only one item is created per rule
            assert(len(rule.rule_dict[Rule.CREATED_ITEMS]) == 1)
            assert(list(rule.rule_dict[Rule.CREATED_ITEMS].values())[0] == 1)
            created_item = list(rule.rule_dict[Rule.CREATED_ITEMS].keys())[0]
            self.add_edge(rule_name, created_item)

    # Figure out what needs to spawn given our requirements
    def calculate_items(self, end_items, visited_rules, distractor_rules, extra_items, inventory_chance):
        # Return what should be in inventory, what should be on ground
        inventory_items = {}
        ground_items = {}

        # What containers to spawn
        containers = set()

        # What fonts to spawn (and how much resource it should have)
        fonts = {}

        # Keep track of if we've used rules
        used_rules = set()

        # Seperate required items into depleted and not
        # Need to add depleted but only max non-depleted
        req_depleted = {}
        req_nondepleted = {}

        # First calculate what we need for end items
        nondepleted_created = {}
        for item in end_items:
            item_count = end_items[item]
            node = self.nodes[item]
            new_req_depleted, new_req_nondepleted, new_req_fonts, new_containers, new_used_rules, nondepleted_created, depth_left = self.get_required_recursive(node, nondepleted_created, self.max_craft_depth)

            # Update required items
            for _ in item_count:
                req_depleted = dictutils.add_merge_dicts(new_req_depleted, req_depleted)
                req_nondepleted = dictutils.max_merge_dicts(new_req_nondepleted, req_nondepleted)
                fonts = dictutils.add_merge_dicts(new_req_fonts, fonts)
                containers |= new_containers
                used_rules |= new_used_rules

            # Assert req_nondepleted is a subset of nondepleted_created
            assert(dictutils.is_subset_of(req_nondepleted, nondepleted_created))

        # Now calculate for rules, if we haven't visited them yet
        for rule_name in visited_rules:
            # If we didn't incindentally visit the rule already
            if rule_name not in used_rules:
                node = self.nodes[rule_name]
                new_req_depleted, new_req_nondepleted, new_req_fonts, new_containers, new_used_rules, nondepleted_created, depth_left = self.get_required_recursive(node, nondepleted_created, self.max_craft_depth)

                # Update required items
                req_depleted = dictutils.add_merge_dicts(new_req_depleted, req_depleted)
                req_nondepleted = dictutils.max_merge_dicts(new_req_nondepleted, req_nondepleted)
                fonts = dictutils.add_merge_dicts(new_req_fonts, fonts)
                containers |= new_containers
                used_rules |= new_used_rules

                # Assert req_nondepleted is a subset of nondepleted_created
                assert(dictutils.is_subset_of(req_nondepleted, nondepleted_created))

        # Now get items for our distractor rules
        # Check whether we actually share a common resource
        # Then give rest of ingredients
        for rule_name in distractor_rules:
            # Distractor must not have been visited already
            if rule_name in used_rules:
                continue

            # Distractor must consume a necessary resource
            node = self.nodes[rule_name]
            rule = node.data['rule']
            distractor_consumed = list(rule.get_required_depleted_items().keys())
            common_resources = set(distractor_consumed) & (set(req_depleted.keys()) | set(req_nondepleted.keys()))
            if len(common_resources) == 0:
                continue

            # Get items needed
            new_req_depleted, new_req_nondepleted, new_req_fonts, new_containers, new_used_rules, nondepleted_created, depth_left = self.get_required_recursive(node, nondepleted_created, self.max_craft_depth)

            # Remove common resource items, and only add extra depleted items that aren't common resources
            proc_depleted = {}
            for key in new_req_depleted:
                if key not in common_resources:
                    proc_depleted[key] = new_req_depleted[key]

            # Update required
            req_depleted = dictutils.add_merge_dicts(proc_depleted, req_depleted)
            req_nondepleted = dictutils.max_merge_dicts(new_req_nondepleted, req_nondepleted)
            fonts = dictutils.add_merge_dicts(new_req_fonts, fonts)
            containers |= new_containers
            used_rules |= new_used_rules

            # Assert req_nondepleted is a subset of nondepleted_created
            assert(dictutils.is_subset_of(req_nondepleted, nondepleted_created))

        # Add distractor items
        for item in extra_items:

            if item == 'goal' or item == 'recipe' or item == 'count':
                continue

            item_type = self.item_dict[item]
            item_count = extra_items[item]

            # Add to nondepleted required item list
            if self.item_dict[item] == 'CraftingItem':
                if item not in req_nondepleted:
                    req_nondepleted[item] = item_count
                else:
                    req_nondepleted[item] += item_count

            # Add crafting containers not already in there
            elif self.item_dict[item] == 'CraftingContainer':
                containers.add(item)

            # Add font values
            elif self.item_dict[item] == 'ResourceFont':
                if item not in fonts:
                    fonts[item] = item_count
                else:
                    fonts[item] += item_count

        # Put items in either ground items or inventory items
        all_items = dictutils.add_merge_dicts(req_depleted, req_nondepleted)
        for item in all_items:
            for i in range(all_items[item]):
                if random.random() < inventory_chance:
                    if item in inventory_items:
                        inventory_items[item] += 1
                    else:
                        inventory_items[item] = 1
                else:
                    if item in ground_items:
                        ground_items[item] += 1
                    else:
                        ground_items[item] = 1

        # Return
        return inventory_items, ground_items, containers, fonts

    # Recursively find all the prerequisites to reach a particular node
    def get_required_recursive(self, node, nondepleted_created, depth):
        # Init return values
        req_depleted = {}
        req_nondepleted = {}
        fonts = {}
        containers = set()
        used_rules = set()

        # If node is rule, follow rules for updating
        if node.data['type'] == 'rule':
            # If we're at a rule, decrement depth
            depth -= 1

            # Add rule to used rules
            used_rules.add(node.key)
            rule = node.data['rule']

            # Get what is needed depleted/nondepleted for rule
            rule_req_depleted = rule.get_required_depleted_items()
            rule_req_nondepleted = rule.get_required_nondepleted_items()

            # Add the required location (font or container)
            if 'required_location' in rule.rule_dict:
                location = rule.rule_dict['required_location']
                if self.item_dict[location] == 'CraftingContainer':
                    containers.add(location)
                elif self.item_dict[location] == 'ResourceFont':
                    # Have font count 1. The traceback expects only 1 output item, so it will do the multiplication above in the stack
                    fonts[location] = 1

            # If there are no required items, we're done (should only be fonts)
            if len(node.incoming) == 0:
                pdb.set_tace()
                # Should be font I think
                assert(self.item_dict[location] == 'ResourceFont')
                return req_depleted, req_nondepleted, fonts, containers, used_rules, nondepleted_created, depth
            # If there are required items left
            else:
                # If we're at the last depth, generate everything we need at this point
                if depth == 0:
                    req_depleted = rule_req_depleted
                    req_nondepleted = rule_req_nondepleted
                    nondepleted_created = dictutils.max_merge_dicts(nondepleted_created, req_nondepleted)
                    return req_depleted, req_nondepleted, fonts, containers, used_rules, nondepleted_created, depth
                # Make the recursive call to get all the required items
                else:
                    # Recursive calls
                    depth_left = depth
                    for item_node in node.incoming:
                        # Figure out how many we need, and if its depleted or not
                        item_name = item_node.key

                        # Get how many of the item we need that are depleted
                        depl_count = 0
                        nondepl_count = 0
                        assert(item_name in rule_req_depleted or item in rule_req_nondepleted)
                        if item_name in rule_req_depleted:
                            depl_count = rule_req_depleted[item_name]

                        # Add nondepleted items
                        if item_name in rule_req_nondepleted:
                            # Don't create any more of the item if we've already created enough
                            if item_name in nondepleted_created:
                                nondepl_count = max(0, rule_req_nondepleted[item_name]-nondepleted_created[item_name])
                            else:
                                nondepl_count = rule_req_nondepleted[item_name]

                            # Update nondepleted_created if this rule means we have to create more
                            if nondepl_count > 0:
                                nondepleted_created[item_name] = rule_req_nondepleted[item_name]

                        # If item_count is 0 because we had a nondepleted item already, skip item
                        item_count = depl_count + nondepl_count
                        if item_count == 0:
                            continue

                        # If it's a terminal item, we need to add the item itself
                        if len(item_node['in'] == 0):
                            if item_name in rule_req_depleted:
                                item_req_depleted = {item_name: rule_req_depleted[item_name]}
                                req_depleted = dictutils.add_merge_dicts(item_req_depleted, req_depleted)
                            if item_name in rule_req_nondepleted:
                                item_req_nondepleted = {item_name: rule_req_nondepleted[item_name]}
                                req_nondepleted = dictutils.max_merge_dicts(item_req_nondepleted, req_nondepleted)
                        # Otherwise, we update based on what the item uses
                        else:
                            item_req_depleted, item_req_nondepleted, item_fonts, item_containers, item_used_rules, item_depth_left = get_required_recursive(item_node, depth)

                            # Update items and lists based on return from item
                            for ind in range(item_count):
                                req_depleted = dictutils.add_merge_dicts(item_req_depleted, req_depleted)
                                fonts = dictutils.add_merge_dicts(item_fonts, fonts)
                            req_nondepleted = dictutils.max_merge_dicts(item_req_nondepleted, req_nondepleted)
                            containers = item_containers | containers
                            used_rules = item_used_rules | used_rules
                            depth_left = min(depth_left, item_depth_left)

                    # Return
                    return req_depleted, req_nondepleted, fonts, containers, used_rules, nondepleted_created, depth_left
        # Follow recursive rules for items
        else:
            assert(node.data['type'] == 'item')

            # This should be handled in rule node
            if len(node.incoming) == 0:
                assert(False)

            # If multiple rules, choose one to follow randomly
            # TODO - is this the right behavior actually? Can't think of a better way to do this
            rule_node = random.choice(node.incoming)

            # Follow the rule node
            return get_required_recursive(rule_node, depth)

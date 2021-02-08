from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import shuffle, random
from collections import defaultdict
import itertools
import mazebasev2.lib.mazebase.items as mi
from mazebasev2.lib.mazebase.utils import MazeException
import pdb

# Add and merge elements of two dictionaries
def add_merge_dicts(dict1, dict2):
    merged_dict = {}
    for key in dict1:
        merged_dict[key] = dict1[key]
    for key in dict2:
        if key in merged_dict:
            merged_dict[key] += dict2[key]
        else:
            merged_dict[key] = dict2[key]
    return merged_dict

# Max and merge elements of two dictionaries
def max_merge_dicts(dict1, dict2):
    merged_dict = {}
    for key in dict1:
        merged_dict[key] = dict1[key]
    for key in dict2:
        if key in merged_dict:
            merged_dict[key] = max(merged_dict[key], dict2[key])
        else:
            merged_dict[key] = dict2[key]
    return merged_dict

# Find a difference between the elements in two dictionaries
# Assumes missing entry means 0
def diff_dicts(dict1, dict2, remove_zeros=False):
    # Get all keys and add 0s to missing key locations
    diff_dict = {}
    all_keys = set(dict1.keys()).union(set(dict2.keys()))
    for key in all_keys:
        if key not in dict1:
            dict1[key] = 0
        if key not in dict2:
            dict2[key] = 0
    
    # Add every diff to diff_dict
    for key in all_keys:
        diff_dict[key] = dict1[key] - dict2[key]

    # If remove_zeros, delete keys that are canceled to 0
    if remove_zeros:
        new_diff_dict = {}
        for key in all_keys:
            if diff_dict[key] != 0:
                new_diff_dict[key] = diff_dict[key]
        diff_dict = new_diff_dict    

    return diff_dict

# Check whether the first dictionary of values and counts is a subset of the second
def is_subset_of(count_dict1, count_dict2):
    # Check empty sets
    # Empty sets are subsets of all sets
    if len(list(count_dict2.keys())) == 0:
        return True

    # If first set is empty, second set can't be a subset (unless it's also empty)
    if len(list(count_dict1.keys())) == 0:
        return False

    # Check set subset of just if dict2 has all the keys in dict1
    for key in count_dict1:
        if key not in count_dict2:
            return False

    # Check that the count of each key is greater in dict2
    for key in count_dict1:
        if count_dict1[key] > count_dict2[key]:
            return False

    # If all those checked out, it's a subset
    return True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from random import choice, randint
from time import sleep
from os import system
from pprint import PrettyPrinter
from six.moves import input
import numpy as np

import mazebasev2.lib.mazebase.games as games
from mazebasev2.lib.mazebase.games import featurizers

from mazebasev2.lib.mazebase.items.terrain import CraftingItem, CraftingContainer, ResourceFont, Block, Water, Switch, Door
from mazebasev2.lib.mazebase.items import agents

import copy
import sys
import os
import time
import random
import yaml
import csv
import ast
import pickle
import codecs

#flask imports
from flask import Flask, render_template, request, jsonify, make_response, json
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

#Read links
links = [line.rstrip('\n') for line in open('misc/labeled_links_1')]

#Read item list
items = [line.rstrip('\n') for line in open('misc/labeled_names')]


link_lookup = {}

def look_up_index(name):

  for i in range(len(links)):
    if name in links[i]:
      return i

  return -1

for i in range(len(items)):
  temp = items[i].split('.')
  temp1 = '/'+items[i].replace(" ", "-")
  i = look_up_index(temp1)
  link_lookup[temp[0]] = links[i]


#keep track of all mazebase instances
current_games = {}
game_count = {}
code_dict = {}
users = {}

with open('misc/codes_1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      code_dict[row[0]] = '0'

name = ''

@app.route('/')
def index():
  return render_template('index.html')


def add_to_file(player1, entrance_code, board, action, timestamp):

  with open("log.txt", "a") as myfile:
    myfile.write(player1 + ',' + entrance_code + ',' + str(board) + ',' + action + ',' + timestamp + '\n')


@app.route('/add_message', methods=['POST'])
def add_message():
  
  json_data = request.get_json(force=True) 

  player1 = json_data["a"]
  message = json_data["action"]

  # entrance code here:

  code = json_data["b"]

  print('adding', player1, message)

  add_to_file(player1, code, message, '2', str(time.time()))

  return jsonify(result={"status": 200})

def query_db(player1, entrance_code, move):
 
    player1 = player1 + '.' + entrance_code

    try:
        book=Book.query.filter_by(player1=player1, move=move).first()
        
        if book == None:
          return None
        else:
          return book.action

        #return jsonify(book.serialize())
    except Exception as e:
      return(str(e))

@app.route('/play')
def play():
  global name
  name = request.args.get('username')
  global code
  code = request.args.get('code')
  add_to_file(name, code, 'NONE', '0', str(time.time()))
  return render_template('play_v4.html')

@app.route('/instruction')
def instruction():
  global name
  global code
  name = request.args.get('username')
  code = request.args.get('code')

  if code in code_dict and code_dict[code] == '0':
    code_dict[code] = '1'
    return render_template('instruction.html')
  else:
    return render_template('invalid.html')

@app.route('/demo')
def demo():
  global name
  global code
  name = request.args.get('username')
  code = request.args.get('code')
  return render_template('demo_v1.html')

@app.route('/game_over')
def game_over():
  global name
  name = request.args.get('username')
  return render_template('game_over.html')

@app.route('/game_continue')
def game_continue():
  global name
  name = request.args.get('username')
  return render_template('game_continue.html')

@app.route('/already_completed')
def already_completed():
  return render_template('already_completed.html')

def create_symbol_color_maps(temp):

  count = 0
  symbol_map = {}
  color_map = {}
  agent_loc = 0

  for y in range(4, -1, -1):
    for x in range(5):

      symbs = 'NONE'
      color = 'None'

      for item in temp[x][y]:
        if(isinstance(item, Block)):
          symbs = symbs + ''
          color = 'Grey'
        elif(isinstance(item, Water)):
          symbs = symbs + ''
          color = 'Blue'
        elif(isinstance(item, Switch)):
          symbs = "switch:"+str(item.state)
        elif(isinstance(item, Door)):
          if item.isopen:
            symbs = 'NONE'
          else:
            symbs ='X'
        elif(isinstance(item, ResourceFont)):
          symbs = item.str_id
        elif(isinstance(item, CraftingItem)):
          symbs = item.str_id
        elif(isinstance(item, CraftingContainer)):
          symbs = item.str_id
        elif(isinstance(item, agents.CraftingAgent)):
          agent_loc = count
          #symbs = symbs + '(A)'
      
      symbol_map[count] = symbs
      color_map[count] = color

      count = count + 1
  return symbol_map, color_map, agent_loc


#create a game
@app.route('/start_game', methods=['GET', 'POST'])
def start_game():

  json_data = request.get_json(force=True) 
  player1 = json_data["a"]
  entrance_code = json_data["b"]

  if (player1, entrance_code) in game_count:
    
    if game_count[(player1, entrance_code)] == 2:
      completed = True

    # otherwise, they have already done one. give the short task. 
    yaml_file = 'mazebasev2/options/knowledge_planner/short_task.yaml'
    game_count[(player1, entrance_code)] = 2

  elif player1 not in users:
    users[player1] = 1
    game_count[(player1,entrance_code)] = 1
    yaml_file = 'mazebasev2/options/knowledge_planner/short_task.yaml'

  else:
    #pick a long or short randomly and add to game_count accordingly. 
    if random.uniform(0, 1) > 0.4:
      yaml_file = 'mazebasev2/options/knowledge_planner/long_task.yaml'
      game_count[(player1, entrance_code)] = 2
    else:
      yaml_file = 'mazebasev2/options/knowledge_planner/short_task.yaml'
      game_count[(player1, entrance_code)] = 1

  with open(yaml_file, 'r') as handle:
    options = yaml.load(handle)

  # Get sub opts
  method_opt = options['method']
  env_opt = options['env']
  log_opt = options['logs'] 

  # Set up the mazebase environment
  knowledge_root = env_opt['knowledge_root']
  world_knowledge_file = os.path.join('mazebasev2', knowledge_root, env_opt['world_knowledge']['train'])
  with open(world_knowledge_file) as f:
      world_knowledge = json.load(f)
  
  # Make the world
  map_size = (env_opt['state_rep']['w'], env_opt['state_rep']['w'], env_opt['state_rep']['h'], env_opt['state_rep']['h'])
  all_games = [games.BasicKnowledgeGame(world_knowledge=world_knowledge, proposed_knowledge=[], options=env_opt, load_items=None, map_size=map_size)]

  # Game wrapper
  game = games.MazeGame(
      all_games,
      featurizer=featurizers.GridFeaturizer()
  )

  current_games[(player1, entrance_code)] = game

  game_observe = game.observe()
  game_observe["extra_items"] = game.game.extra_items
  game_observe["inventory"] = game.game.inventory
  game_observe["goal"] = game.game.goal
  game_observe["recipe"] = game.game.recipe

  add_to_file(player1, entrance_code, game_observe, '1', str(time.time()))

  temp = game.display() 
  symbol_map, color_map, agent_loc = create_symbol_color_maps(temp)

  completed = False

  return jsonify(result=symbol_map, color=color_map, goal=game.game.goal, inventory=game.game.inventory, recipe=game.game.recipe, agent=agent_loc, links=link_lookup, already_complete=completed)

#make move and return board..
@app.route('/make_move', methods=['GET', 'POST'])
def make_move():

  #retrieve game given player usernames

  json_data = request.get_json(force=True) 
  player1 = json_data["a"]
  entrance_code = json_data["b"]

  game = current_games[(player1, entrance_code)]

  action = json_data["action"]

  temp_action = action

  if action == "open_door":
    action = "toggle_switch"

  game.act(action)

  temp = game.display()
  symbol_map, color_map, agent_loc = create_symbol_color_maps(temp)

  current_games[(player1, entrance_code)] = game

  game_observe = game.observe()
  game_observe["extra_items"] = game.game.extra_items
  game_observe["inventory"] = game.game.inventory
  game_observe["goal"] = game.game.goal
  game_observe["recipe"] = game.game.recipe

  add_to_file(player1, entrance_code, game_observe, '3', str(time.time()))

  is_complete = False
  rand_num = 2

  if game_count[(player1, entrance_code)] == 1:
    code = "NONE"
  else:
    # GENERATE CODE
    num = random.randrange(1, 10**6)
    # using format of 6 digit number
    num_with_zeros = '{:06}'.format(num)
    code = str(num)

  if game.is_over() or rand_num == 1:
    is_complete = True
    add_to_file(player1, entrance_code, code, '5', str(time.time()))


  return jsonify(result=symbol_map, complete=is_complete, code=code, color=color_map, inventory=game.game.inventory, agent=agent_loc)


#restart
@app.route('/restart_game', methods=['GET', 'POST'])
def restart_game():

  #retrieve game given player usernames

  json_data = request.get_json(force=True) 
  player1 = json_data["a"]
  entrance_code = json_data["b"]

  add_to_file(player1, entrance_code, 'NONE', '4', str(time.time()))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)

name = ''

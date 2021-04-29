import pandas as pd
import numpy as np
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import json
import sys
from PIL import Image

## READ RECIPES:
df = pd.read_csv('../generated_images/recipes.csv')

#just images of the raw items first:
def generate_individual_labeled_images():

	plt.rcParams.update({'font.size': 50})

	#for item in df[df['Type'] == 'Wool']['Item Created']:
	#for item in df['Item Created']:
	
	for item in ['Cobblestone Stash']:
		print(item)

		if item == "NONE":
			continue

		img=mpimg.imread('generated_images/base_images/store.png')

		fig = plt.imshow(img)
		#name = item
		name = ""
		spaced = item.split(' ')
		for s in spaced:
			if name == "":
				name = s
			else:
				name = name + '\n' + s		


		fig.axes.get_yaxis().set_visible(False)

		fig.axes.spines['top'].set_visible(False)
		fig.axes.spines['right'].set_visible(False)
		fig.axes.spines['bottom'].set_visible(False)
		fig.axes.spines['left'].set_visible(False)

		plt.tick_params(
		    axis='x',          # changes apply to the x-axis
		    which='both',      # both major and minor ticks are affected
		    bottom=False,      # ticks along the bottom edge are off
		    top=False,         # ticks along the top edge are off
		    labelbottom=False)

		plt.xlabel(name)


		plt.savefig('generated_images/labeled_images/'+item+'.png', bbox_inches='tight')
		plt.close()


#generate_individual_labeled_images()

#Put them together for recipes
#do this recursively.. 


def recurse_recipe(item):

	# get index of this item..
	idx = df.index[df['Item Created'] == item]
	row = df.iloc[idx[0]]
	# print(row)

	#raw ingredient
	if row['Required Item'] == row['Required Item']:
		if row['Required Item'] == 'NONE':
			#print(row['Location Required'] + " " + row['Action Required'] + " " + item)
				
			## HAS 3 ITEMS, plus 1 filler.

			loc_item_image= 'generated_images/labeled_images/'+row['Location Required']+'.png'

			action_image= 'generated_images/base_images/'+row['Action Required']+'.jpg'

			filler_image = 'generated_images/base_images/blank.png'

			item_image =  'generated_images/labeled_images/'+item+'.png'


			images = map(Image.open, [filler_image, filler_image, loc_item_image, action_image, item_image])
			

			total_width = 500

			max_height = 120

			new_im = Image.new('RGB', (total_width, max_height))

			x_offset = 0
			for im in images:
			  im = im.resize((100,120))
			  new_im.paste(im, (x_offset,0))
			  x_offset += im.size[0]

			# new_im.save('generated_images/recipe_images/'+item+'.png')

			return 1, new_im

		else:

			## HAS 4 ITEMS

			loc_item_image= 'generated_images/labeled_images/'+row['Location Required']+'.png'
			req_item_image= 'generated_images/labeled_images/'+row['Required Item']+'.png'

			action_image= 'generated_images/base_images/'+row['Action Required']+'.jpg'

			item_image =  'generated_images/labeled_images/'+item+'.png'

			filler_image = 'generated_images/base_images/blank.png'


			images = map(Image.open, [filler_image, loc_item_image, req_item_image, action_image, item_image])
			

			total_width = 500

			max_height = 120

			new_im = Image.new('RGB', (total_width, max_height))

			x_offset = 0
			for im in images:
			  im = im.resize((100,120))
			  new_im.paste(im, (x_offset,0))
			  x_offset += im.size[0]

			#new_im.save('generated_images/recipe_images/'+item+'.png')

			#print(row['Required Item'] + ' + ' + row['Location Required'] + " " + row['Action Required'] + " " + item)
			return 1, new_im
	
	#2 ingredients
	elif row['Ingredient 2'] == row['Ingredient 2']:


		#print(row['Ingredient 1'] + ' + ' + row['Ingredient 2'] + " " + row['Action Required'] + " " + item)
		
		## ADD THE LOCATION!

		loc_item_image= 'generated_images/labeled_images/'+item+' Station.png'

		ingred_item1_image = 'generated_images/labeled_images/'+row['Ingredient 1']+'.png'

		ingred_item2_image = 'generated_images/labeled_images/'+row['Ingredient 2']+'.png'

		action_image= 'generated_images/base_images/'+row['Action Required']+'.jpg'

		item_image =  'generated_images/labeled_images/'+item+'.png'


		images = map(Image.open, [loc_item_image, ingred_item1_image, ingred_item2_image, action_image, item_image])
		

		total_width = 500

		max_height = 120

		new_im = Image.new('RGB', (total_width, max_height))

		x_offset = 0
		for im in images:
		  im = im.resize((100,120))
		  new_im.paste(im, (x_offset,0))
		  x_offset += im.size[0]


		count, temp = recurse_recipe(row['Ingredient 1'])
		count1, temp1 = recurse_recipe(row['Ingredient 2'])

		total = count + count1 + 1

		if total > 6:
			return 6, None

		## Stitch images

		new_height = total * 120
		stitched_image = Image.new('RGB', (total_width, new_height))

		stitched_image.paste(new_im, (0,0))
		stitched_image.paste(temp, (0,new_im.size[1]))
		stitched_image.paste(temp1, (0,new_im.size[1] + temp.size[1]))

		return total, stitched_image

	#1 ingredient
	else:
		#print(row['Ingredient 1'] + " " + row['Action Required'] + " " + item)
	
		## ADD THE LOCATION!

		loc_item_image= 'generated_images/labeled_images/'+item+' Station.png'

		ingred_item1_image = 'generated_images/labeled_images/'+row['Ingredient 1']+'.png'

		action_image= 'generated_images/base_images/'+row['Action Required']+'.jpg'

		item_image =  'generated_images/labeled_images/'+item+'.png'

		filler_image = 'generated_images/base_images/blank.png'

		images = map(Image.open, [filler_image, loc_item_image, ingred_item1_image, action_image, item_image])
		
		total_width = 500

		max_height = 120

		new_im = Image.new('RGB', (total_width, max_height))

		x_offset = 0
		for im in images:
		  im = im.resize((100,120))
		  new_im.paste(im, (x_offset,0))
		  x_offset += im.size[0]

		count, temp = recurse_recipe(row['Ingredient 1'])
		total = count + 1

		if total > 6:
			return 6, None

		## stitch images

		new_height = total * 120
		stitched_image = Image.new('RGB', (total_width, new_height))
		stitched_image.paste(new_im, (0,0))
		stitched_image.paste(temp, (0,new_im.size[1]))

		return total, stitched_image


def create_recipes():

	for index, row in df.iterrows():

		item = row['Item Created']	
		
		print(item)

		count, image = recurse_recipe(item)

		filler_image = 'generated_images/base_images/blank.png'

		images = map(Image.open, [filler_image, filler_image, filler_image, filler_image, filler_image])
		
		total_width = 500

		max_height = 120

		new_im = Image.new('RGB', (total_width, max_height))

		x_offset = 0
		for im in images:
		  im = im.resize((100,120))
		  new_im.paste(im, (x_offset,0))
		  x_offset += im.size[0]

		if count < 6:

			## PAD THE BOTTOM OF RECIPES THAT ARE < 5.
			stitched_image = Image.new('RGB', (total_width, 600))
			stitched_image.paste(image, (0,0))
			y_offset = image.size[1]

			for i in range(5-count):
				stitched_image.paste(new_im, (0,y_offset))
				y_offset = y_offset + 120

			# print(count)
			stitched_image.save('generated_images/recipe_images_v2/'+item+'.png')


ingreds = {}

def add_ingred(item):
	if item in ingreds:
		ingreds[item] = ingreds[item] + 1
	else:
		ingreds[item] = 1


def return_spawns(item):

	# get index of this item..
	idx = df.index[df['Item Created'] == item]
	row = df.iloc[idx[0]]
	# print(row)

	#raw ingredient
	if row['Required Item'] == row['Required Item']:

		add_ingred(row['Item Created'])

		## WHAT WOULD HAPPEN TO THIS.. 
		if row['Required Item'] == 'NONE':

			return 1, [row['Location Required']], [row['Item Created']]

		else:

			return 1, [row['Location Required'], row['Required Item']], [row['Item Created']]
	
	#2 ingredients
	elif row['Ingredient 2'] == row['Ingredient 2']:

		add_ingred(row['Ingredient 1'])
		add_ingred(row['Ingredient 2'])

		count, temp, blah = return_spawns(row['Ingredient 1'])
		count1, temp1, blah1 = return_spawns(row['Ingredient 2'])

		total = count + count1 + 1

		return total, [item+' Station'] + temp + temp1, [row['Item Created']] + blah + blah1

	#1 ingredient
	else:

		add_ingred(row['Ingredient 1'])

		count, temp, blah = return_spawns(row['Ingredient 1'])
		total = count + 1

		return total, [item+' Station'] + temp, [row['Item Created']] + blah






## BUILD JSON FILE

#Read links
links = [line.rstrip('\n') for line in open('recipe_links_1')]

#Read item list
items = [line.rstrip('\n') for line in open('recipe_names')]

item_list = [line.rstrip('\n') for line in open('recipe_list')]



def look_up_index(name):

	for i in range(len(links)):
		if name in links[i]:
			return i

	return -1


# need to recursively traverse to figure out what items to spawn??

spawns = []
rules = []
rules_dict = {}
objects = {}

spawns1 = []
rules1= []
rules_dict1 = {}
objects1 = {}

## THIS WILL GENERATE ALL OF THE SPAWNS
#for i in range(len(items)):

## DO A REVERSE LOOKUP OF ALL OF THE ONES THAT WE NEED...

# for item in item_list:
# 	count, lst, blah = return_spawns(item)
# 	print(item, count)



#for i in range(20):

#	item = items[i]
#	s = item.split('.')
#	item = s[0]


for item in item_list:

	#full = item + '.png'
	#i = items.index(full)

	#do a look up.. 
	temp = '/'+item.replace(" ", "-")
	i = look_up_index(temp)


	d2 = {}
	d2["goal"] = "Make "+ item + " ("+item+"=1)"
	
	d2["recipe"] = links[i]

	count, lst, blah = return_spawns(item)
	d2["count"] = count

	print(count)

	for s in lst:
		d2[s] = 1

	if count > 2:
		spawns.append(d2) 
	else:
		spawns1.append(d2)

	lst = lst + blah

	## Add all the rules for the things?	

	for l in lst:

		## FIX THIS SO THAT YOU ALSO GENERATE THE ITEMS THAT ARE CREATED IN THE RULES.

		
		idx = df.index[df['Item Created'] == l]


		if len(idx) > 0:

			row = df.iloc[idx[0]]

			objects[row["Item Created"]] = "CraftingItem"

			if row['Type'] != 'Raw':
				d = {}
				d["created_items"] = {row["Item Created"] : 1}
				d["rule_name"] = "craft_"+row["Item Created"]
				d["required_action"] = "craft"
				d["required_location"] = row["Item Created"] + " Station"

				if count > 2:
					objects[d["required_location"]] = "CraftingContainer"
				else:
					objects1[d["required_location"]] = "CraftingContainer"
				
				if row['Ingredient 2'] == row['Ingredient 2']:
					d["depleted_items"] = {row["Ingredient 1"] : 1, row["Ingredient 2"] : 1}
				else:
					d["depleted_items"] = {row["Ingredient 1"] : 1}
				d["spawn_ind"] = 1

				if count > 2:
					if row["Item Created"] not in rules_dict:
						rules.append(d)
						rules_dict[row["Item Created"]] = d
				else:
					if row["Item Created"] not in rules_dict1:
						rules1.append(d)
						rules_dict1[row["Item Created"]] = d

			else:

				#use the mine once you have the item
			
				d1 = {}
				d1["created_items"] = {row["Item Created"] : 1}
				d1["rule_name"] = "mine_"+row["Item Created"]
				d1["required_action"] = "mine"
				d1["required_location"] = row["Location Required"]

				if count > 2:
					objects[d1["required_location"]] = "ResourceFont"
				else:
					objects1[d1["required_location"]] = "ResourceFont"

				if row['Required Item'] != 'NONE':
					d1["non_depleted_items"] = {row["Required Item"] : 1} ## CHECK IF IT IS NONE?
					objects[row["Required Item"]] = "CraftingItem"

				d1["spawn_ind"] = 1

				if count > 2:
					if row["Item Created"] not in rules_dict:
						rules.append(d1)
						rules_dict[row["Item Created"]] = d1
				else:
					if row["Item Created"] not in rules_dict1:
						rules1.append(d1)
						rules_dict1[row["Item Created"]] = d1
		
		# elif "Station" in l:

		# 	objects[l] = "CraftingContainer"
		# else:
		# 	objects[l] = "CraftingItem"


	#print(item, return_spawns(item))

	#return the locations from the items separately.. 

	# for each one add the link in, and the goal which is the item. 


## THIS WILL GENERATE ALL OF THE RULES

# for index, row in df.iterrows():

# 	if row['Type'] != 'Raw':
# 		#use the Craft

# 		d = {}
# 		d["created_items"] = {row["Item Created"] : 1}
# 		d["rule_name"] = "craft_"+row["Item Created"]
# 		d["required_action"] = "craft"
# 		d["required_location"] = row["Item Created"] + " Station"
# 		if row['Ingredient 2'] == row['Ingredient 2']:
# 			d["depleted_items"] = {row["Ingredient 1"] : 1, row["Ingredient 2"] : 1}
# 		else:
# 			d["depleted_items"] = {row["Ingredient 1"] : 1}
# 		d["spawn_ind"] = 1

# 		rules.append(d)

# 	else:

# 		#use the mine once you have the item
	
# 		d1 = {}
# 		d["created_items"] = {row["Item Created"] : 1}
# 		d["rule_name"] = "mine_"+row["Item Created"]
# 		d["required_action"] = "mine"
# 		d["required_location"] = row["Location Required"]

# 		if row['Required Item'] != 'NONE':
# 			d["non_depleted_items"] = {row["Required Item"] : 1} ## CHECK IF IT IS NONE?
# 		d["spawn_ind"] = 1

# 		rules.append(d1)

dictionary = {}
dictionary["rules"] = rules
dictionary["spawns"] = spawns
dictionary["objects"] = objects

dictionary1 = {}
dictionary1["rules"] = rules1
dictionary1["spawns"] = spawns1
dictionary1["objects"] = objects1

with open('data_long_1.json', 'w') as outfile:
    json.dump(dictionary, outfile, indent=4)

with open('data_short_1.json', 'w') as outfile:
    json.dump(dictionary1, outfile, indent=4)








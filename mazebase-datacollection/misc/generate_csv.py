import csv
import random

with open('codes_1.csv', mode='w') as file:
	writer = csv.writer(file, delimiter=',')

	for i in range(50):

		#generate random numbers
		num = random.randrange(1, 10**6)
		num_with_zeros = '{:06}'.format(num)

		writer.writerow([num_with_zeros])

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import ast 
import scipy.stats


plt.rc('font', size=18)


## Read files

ilrlresults = []

for i in range(3):
	lineList = []
	fileName = "trained_models/ppo/ilrl_" + str(i+1) + '.txt'
	lineList = [line.rstrip('\n') for line in open(fileName)]
	ilrlresults.append(lineList)

rlresults = []
for i in range(3):
	lineList = []
	fileName = "trained_models/ppo/rl_baseline" + str(i+1) + '.txt'
	lineList = [line.rstrip('\n') for line in open(fileName)]
	rlresults.append(lineList)

file = ['ilrl_lang2', 'ilrl_lang3', 'ilrl_lang22']
ilrllangresults = []
for i in range(3):
	lineList = []
	fileName = "trained_models/ppo/"+ file[i]+".txt"
	lineList = [line.rstrip('\n') for line in open(fileName)]
	ilrllangresults.append(lineList)


## Plot

steps = [1.0, 2.0, 3.0, 5.0]
step = 1.0

all_runs = []

for trial in range(3):
	nframes = []
	run_res = []

	for i in range(0, len(ilrlresults[0]), 2):
		index = int(rlresults[trial][i])
		nframes.append(index)
		results = ast.literal_eval(rlresults[trial][i+1])
		run_res.append(float(results[step][0])/float(results[step][1]))

	all_runs.append(run_res)

all_runs = np.asarray(all_runs)
error = scipy.stats.sem(all_runs, axis=0)
mean_run = np.mean(all_runs, axis=0)
plt.plot(nframes, mean_run, '-', color='green', label="RL") #, label=label, color=color, alpha=alpha)
plt.fill_between(nframes, mean_run-error, mean_run+error,
                         alpha=0.2, linewidth=0.0, color='green') #, color=color)
plt.xlabel("Episodes")
plt.ylabel("Accuracy")


all_runs = []

for trial in range(3):
	nframes = []
	run_res = []

	for i in range(0, len(ilrlresults[0]), 2): #0, len(ilrlresults[0]), 2
		index = int(ilrlresults[trial][i])
		nframes.append(index)
		results = ast.literal_eval(ilrlresults[trial][i+1])
		run_res.append(float(results[step][0])/float(results[step][1]))

	all_runs.append(run_res)


# min_length = min([len(run) for run in all_runs])
#all_runs = np.asarray([run[:min_length] for run in all_runs])
all_runs = np.asarray(all_runs)
error = scipy.stats.sem(all_runs, axis=0)
mean_run = np.mean(all_runs, axis=0)
plt.plot(nframes, mean_run, '-', label="IL+RL") #, label=label, color=color, alpha=alpha)
plt.fill_between(nframes, mean_run-error, mean_run+error,
                         alpha=0.2, linewidth=0.0) #, color=color)


# ilresults = [0.15]*len(nframes)
# illangresults = [0.22]*len(nframes)

# plt.plot(nframes, ilresults, ':', color='black', label="IL")
# plt.plot(nframes, illangresults, ':', color='purple', label="IL w/lang")

all_runs = []

for trial in range(3):
	nframes = []
	run_res = []

	for i in range(0, len(ilrlresults[0]), 2):

		index = int(ilrllangresults[trial][i])
		nframes.append(index)
		results = ast.literal_eval(ilrllangresults[trial][i+1])
		run_res.append(float(results[step][0])/float(results[step][1]))

	all_runs.append(run_res)

all_runs = np.asarray(all_runs)
error = scipy.stats.sem(all_runs, axis=0)
mean_run = np.mean(all_runs, axis=0)
plt.plot(nframes, mean_run, '-', color='red', label="Ours") #, label=label, color=color, alpha=alpha)
plt.fill_between(nframes, mean_run-error, mean_run+error,
                         alpha=0.2, linewidth=0.0, color='red') #, color=color)


plt.legend()
plt.title("1 Step Tasks")
plt.savefig('new_graphs/result1.png', bbox_inches = "tight")




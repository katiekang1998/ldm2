import numpy as np
import matplotlib.pyplot as plt 
import csv

data_path = "/home/katie/Desktop/ldm2/data/"
folders = ["expert", "medium_expert", "medium_replay2"]

for folder in folders:
	rollout_lengths = []
	with open(data_path+folder+"/rollout_length.csv", 'r') as csvfile:
	    csvreader = csv.reader(csvfile)
	    fields = next(csvreader)
	    for row in csvreader:
	        rollout_lengths.append(float(row[1]))
	plt.plot([i for i in range(len(rollout_lengths))], rollout_lengths, label = folder)

plt.legend()
plt.xlabel("steps")
plt.ylabel("rollout length")
plt.show()
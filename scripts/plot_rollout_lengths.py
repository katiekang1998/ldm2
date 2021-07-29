import numpy as np
import matplotlib.pyplot as plt 
import csv


data_path = "/home/katie/Desktop/ldm2/data/"
folders = ["flow_ldms/expert", "flow_ldms/expert_cql2_1", "flow_ldms/expert_ground_truth_density_model_update_actor_only", "bc/expert", "bc_ldms/expert", "bc_ldms/expert_cql2_1", "bc_ldms/expert_update_actor_only"]

for folder in folders:
	rollout_lengths = []
	with open(data_path+folder+"/rollout_length.csv", 'r') as csvfile:
	    csvreader = csv.reader(csvfile)
	    fields = next(csvreader)
	    for row in csvreader:
	        rollout_lengths.append(float(row[1]))
	# plt.plot([i for i in range(len(rollout_lengths))][:50], rollout_lengths[:50], label = folder)

	idxs = []
	smooth_rollout_lengths = []
	for i in range(50-3):
		idxs.append(i)
		smooth_rollout_lengths.append(np.mean(rollout_lengths[i:i+3]))
	plt.plot(idxs, smooth_rollout_lengths, label = folder)

plt.legend()
plt.xlabel("steps")
plt.ylabel("rollout length")
plt.show()
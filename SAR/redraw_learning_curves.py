import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import os

def extract_values(correct_path, policy):
    file_name0 = "policy_rewards" + str(policy) + "," + str(0) + ".txt"
    file_name1 = "policy_rewards" + str(policy) + "," + str(1) + ".txt"
    return np.loadtxt(os.path.join(correct_path, file_name0)), np.loadtxt(os.path.join(correct_path, file_name1))

def moving_avarage_smoothing(X,k):
	S = np.zeros(X.shape[0])
	for t in range(X.shape[0]):
		if t < k:
			S[t] = np.mean(X[:t+1])
		else:
			S[t] = np.sum(X[t-k:t])/k
	return S

def plot_and_save(rewards, save_path, policy, lb):
        c = cm.rainbow(np.linspace(0, 1, len(rewards)))
        l = lb
        print(l[0])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

        ax1.set_title('Rewards per episode for agent 0:')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Rewards')

        ax2.set_title('Rewards per episode for agent 1')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Rewards')

        ax1.set_ylim([-20, 30])
        ax2.set_ylim([-20, 30])

        for i in range(0, 3):
            ax1.plot(np.arange(0, len(rewards[i][0])), rewards[i][0], color=c[i])
            ax2.plot(np.arange(0, len(rewards[i][1])), rewards[i][1], color=c[i])
        
        ax1.legend(l)
        ax2.legend(l)

        file_name = "learning_curve" + str(policy) + ".png"
        plt.savefig(os.path.join(save_path, file_name))
        plt.close()

PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'QLearning')
load_path = os.path.join(PATH, 'Saved_data')

policy = [0,1,2]
lr = [0.1, 0.1, 0.1]
dr = [0.1, 0.1, 0.1]
er = [0.05, 0.1, 0.3]
label0 = "α=%s, γ=%s, ϵ=%s, ϵ_min=%s, ϵ_max=%s, ϵ_d=%s" %(lr[0], dr[0], er[0], er[0], er[0], er[0])
label1 = "α=%s, γ=%s, ϵ=%s, ϵ_min=%s, ϵ_max=%s, ϵ_d=%s" %(lr[1], dr[1], er[1], er[1], er[1], er[1])
label2 = "α=%s, γ=%s, ϵ=%s, ϵ_min=%s, ϵ_max=%s, ϵ_d=%s" %(lr[2], dr[2], er[2], er[2], er[2], er[2])

label = []
label.append(label0)
label.append(label1)
label.append(label2)

rewards00, rewards01 = extract_values(load_path, policy[0])
rewards10, rewards11 = extract_values(load_path, policy[1])
rewards20, rewards21 = extract_values(load_path, policy[2])

rewards0 = [rewards00, rewards01]
rewards1 = [rewards10, rewards11]
rewards2 = [rewards20, rewards21]

rewards = np.array([rewards0, rewards1, rewards2])

for i in range(0 ,3):
    rewards[i, 0] = moving_avarage_smoothing(rewards[i, 0], 20)
    rewards[i, 1] = moving_avarage_smoothing(rewards[i, 1], 20)

plot_and_save(rewards, load_path, policy, label)

print("done")

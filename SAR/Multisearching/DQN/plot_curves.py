import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.pyplot import cm
import os
import math

def moving_avarage_smoothing(X,k):
	S = np.zeros(X.shape[0])
	for t in range(X.shape[0]):
		if t < k:
			S[t] = np.mean(X[:t+1])
		else:
			S[t] = np.sum(X[t-k:t])/k
	return S

def plot_learning_curves(scores, filename, policy, c, step, moving_step, ts, pr, nr, per, nsr, ms, lr, dr, er):
    mean_rewards = np.zeros((len(scores), len(scores[0][0])))
    std_rewards = np.zeros((len(scores), len(scores[0][0])))

    if len(scores[0]) > 1:
        for i_ts in range(len(scores)):
            for i_ep in range(len(scores[0][0])):
                s = sum(scores[i_ts][e][i_ep] for e in range(len(scores[i_ts])))
                mean_rewards[i_ts][i_ep] = s / len(scores[0])

    if len(scores[0]) > 1:
        for i_ts in range(len(scores)):
            for i_ep in range(len(scores[0][0])):
                v = sum((scores[i_ts][e][i_ep]-mean_rewards[i_ts][i_ep])**2 for e in range(len(scores[i_ts])))
                std_rewards[i_ts][i_ep] = math.sqrt(v / (len(scores[0])-1))
    
    mov_avg_rewards = np.empty(mean_rewards.shape)

    # mov_avg_rewards[0] = moving_avarage_smoothing(mov_avg_rewards, 10)

    if moving_step > 1:
        mov_avg_rewards = mean_rewards
        mov_avg_rewards[0] = moving_avarage_smoothing(np.array(scores[0][0]), moving_step)
    else:
         mov_avg_rewards = scores[0]
    
    fig=plt.figure()
    ax=fig.add_subplot(111)

    l = []
    l.append("%s:  e=%s, pr=%s, nr=%s, per=%s, nsr=%s, s=%s, \nα=%s, γ=%s, ϵ=%s" %(
            str(policy),
            str(ts),
            str(pr[policy]),
            str(nr[policy]),
            str(per[policy]),
            str(nsr[policy]),
            str(ms[policy]),
            str(lr[policy]), 
            str(dr[policy]), 
            str(er[policy])
            ))

    ax.plot(np.arange(0, len(mov_avg_rewards[0]), int(step)), mov_avg_rewards[0][::int(step)], color=c[policy], label=l[0])
    
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Episodes", color="C0")
    ax.set_ylabel("Rewards", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    np_scores = np.array(scores)
    # np_scores.min()-1
    # ax.set_ylim(np_scores.min()-1, np_scores.max()+0.5)
    ax.set_ylim(25, np_scores.max()+0.5)
    ax.set_title("Learning curve:")
    # plt.xlabel("Training Steps")
    # plt.ylabel("Rewards")

    # plt.legend(loc = "best")

    plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_steps_curves(steps, filename, policy, c, step, moving_step, ts, pr, nr, per, nsr, ms, lr, dr, er):
    mean_steps = np.zeros((len(steps), len(steps[0][0])))
    std_rewards = np.zeros((len(steps), len(steps[0][0])))

    if len(steps[0]) > 1:
        for i_ts in range(len(steps)):
            for i_ep in range(len(steps[0][0])):
                s = sum(steps[i_ts][e][i_ep] for e in range(len(steps[i_ts])))
                mean_steps[i_ts][i_ep] = s / len(steps[0])

    if len(steps[0]) > 1:
        for i_ts in range(len(steps)):
            for i_ep in range(len(steps[0][0])):
                v = sum((steps[i_ts][e][i_ep]-mean_steps[i_ts][i_ep])**2 for e in range(len(steps[i_ts])))
                std_rewards[i_ts][i_ep] = math.sqrt(v / (len(steps[0])-1))
    
    mov_avg_steps = np.empty(mean_steps.shape)

    # mov_avg_rewards[0] = moving_avarage_smoothing(mov_avg_rewards, 10)

    if moving_step > 1:
        mov_avg_steps = mean_steps
        mov_avg_steps[0] = moving_avarage_smoothing(np.array(steps[0][0]), moving_step)
    else:
         mov_avg_steps = steps[0]
    
    fig=plt.figure()
    ax=fig.add_subplot(111)

    l = []
    l.append("%s:  e=%s, pr=%s, nr=%s, per=%s, nsr=%s, s=%s, \nα=%s, γ=%s, ϵ=%s" %(
            str(policy),
            str(ts),
            str(pr[policy]),
            str(nr[policy]),
            str(per[policy]),
            str(nsr[policy]),
            str(ms[policy]),
            str(lr[policy]), 
            str(dr[policy]), 
            str(er[policy])
            ))

    ax.plot(np.arange(0, len(mov_avg_steps[0]), int(step)), mov_avg_steps[0][::int(step)], color=c[policy], label=l[0])

    # for i, val in enumerate(mov_avg_steps[0]):
    #     if val == 191:
    #         ax.plot(i, val, 'r', label=l[0])  # Red if equal to x
    #     else:
    #         ax.plot(i, val, 'b', label=l[0])  # Blue otherwise
    
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Episodes", color="C0")
    ax.set_ylabel("Rewards", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    np_scores = np.array(steps)
    # np_scores.min()-1
    # ax.set_ylim(np_scores.min()-1, np_scores.max()+0.5)
    ax.set_ylim(np_scores.min()-1, 200)
    ax.set_title("Learning curve:")
    # plt.xlabel("Training Steps")
    # plt.ylabel("Rewards")

    # plt.legend(loc = "best")

    plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')

def read_json(path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as f:
        return json.load(f)
    
def read_hp_json(path, file_name):
    lst = []
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def read_hp_jsons(path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as f:
        txt = json.load(f)
    
    ts, _, _, lr, dr, er, pr, nr, per, nsr, ms, _, _, _, _ = txt.split(",")
    return ts, lr, dr, er, pr, nr, per, nsr, ms


PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'DQN')
load_path = os.path.join(PATH, 'Saved_data')
if not os.path.exists(load_path): os.makedirs(load_path)


lc = True
sc = True

step = 10#len(scores[0][0])/10000
moving_step = 1
policies = [0,1,2]


rewards = []
steps = []
tss, lrs, drs, ers, prs, nrs, pers, nsrs, mss = [], [], [], [], [], [], [], [], []
file_name = "hyperparameters.json"
hp = read_hp_json(load_path, file_name)
num_policies =     len(hp["learning rate"]) \
                        * len(hp["discount rate"]) \
                        * len(hp["epsilon"]) \
                        * len(hp["positive rewards"]) \
                        * len(hp["negative rewards"]) \
                        * len(hp["positive exploration rewards"]) \
                        * len(hp["negative step rewards"]) \
                        * len(hp["max steps"])
for policy in range(len(policies)): #num_policies
    file_name = "ts_rewards%s.json" %(str(policy))
    rewards.append(read_json(load_path, file_name))
    file_name = "ts_steps%s.json" %(str(policy))
    steps.append(read_json(load_path, file_name))

pr = np.zeros((num_policies, ))
nr = np.zeros((num_policies, ))
per = np.zeros((num_policies, ))
nsr = np.zeros((num_policies, ))
ms = np.zeros((num_policies, ))
lr = np.zeros((num_policies, ))
dr = np.zeros((num_policies, ))
er = np.zeros((num_policies, 3))

policy = 0

for pr_i in hp["positive rewards"]:
    for nr_i in hp["negative rewards"]:
        for per_i in hp["positive exploration rewards"]:
            for nsr_i in hp["negative step rewards"]:
                for ms_i in hp["max steps"]:
                    for lr_i in hp["learning rate"]:
                        for dr_i in hp["discount rate"]:
                            for er_i in hp["epsilon"]:
                                if policy in policies:
                                      pr[policy] = pr_i
                                      nr[policy] = nr_i
                                      per[policy] = per_i
                                      nsr[policy] = nsr_i
                                      ms[policy] = ms_i
                                      lr[policy] = lr_i
                                      dr[policy] = dr_i
                                      er[policy] = er_i
                                policy += 1

c = cm.rainbow(np.linspace(0, 1, num_policies))

for policy in policies:
    if lc:
        filename = 'drone_0_learning_cruves%s, step=%s, moving=%s.png' %(str(policy), str(step), moving_step)
        filename = os.path.join(load_path, filename)

        plot_learning_curves([rewards[policy]], filename, policy, c, step, moving_step, hp["training sessions"], pr, nr, per, nsr, ms, lr, dr, er)
    if sc:
        filename = 'drone_0_steps_cruves%s, step=%s, moving=%s.png' %(str(policy), str(step), moving_step)
        filename = os.path.join(load_path, filename)

        plot_steps_curves([steps[policy]], filename, policy, c, step, moving_step, hp["training sessions"], pr, nr, per, nsr, ms, lr, dr, er)

# string = ""
# string = [string+","+str(i) for i in policies]
# filename = 'drone_1_learning_cruves%s, step=%s.png' %(string, str(step))
# filename = os.path.join(load_path, filename)
# plot_learning_curves([rewards[0][10:20]], filename, step, tss, prs, nrs, pers, nsrs, mss, lrs, drs, ers)
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.pyplot import cm
import os
import math

def plot_learning_curves(scores, filename, step, ts, pr, nr, per, nsr, ms, lr, dr, er):
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
    
    fig=plt.figure()
    ax=fig.add_subplot(111)

    l = []
    for i in range(len(scores)):
        l.append("%s:  e=%s, pr=%s, nr=%s, per=%s, nsr=%s, s=%s, α=%s, γ=%s, ϵ=%s" %(
                str(i),
                str(ts[i]),
                str(pr[i]),
                str(nr[i]),
                str(per[i]),
                str(nsr[i]),
                str(ms[i]),
                str(lr[i]), 
                str(dr[i]), 
                str(er[i])
                ))

    c = cm.rainbow(np.linspace(0, 1, len(scores)))
    
    if len(scores[0]) == 1:
        for i in range(len(policies)):
            ax.plot(np.arange(0, len(scores[i][0]), int(step+i)), scores[i][0][::int(step+i)], color=c[i], label=l[i])
    else:
        for i in range(len(policies)):
            ax.plot(np.arange(0, len(mean_rewards[i]), int(step+i)), mean_rewards[i][::int(step+i)], color=c[i], label=l[i])
            plt.fill_between(np.arange(0, len(mean_rewards[i]), int(step+i)), mean_rewards[i][::int(step+i)]-std_rewards[i][::int(step+i)], mean_rewards[i][::int(step+i)]+std_rewards[i][::int(step+i)], alpha = 0.1, color=c[i])
    
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Rewards", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    np_scores = np.array(scores)
    # np_scores.min()-1
    ax.set_ylim(np_scores.min()-1, np_scores.max()+1)
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
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as f:
        txt = json.load(f)
    
    ts, lr, dr, er, pr, nr, per, nsr, ms, _, _, _, _ = txt.split(",")
    return ts, lr, dr, er, pr, nr, per, nsr, ms


PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'DQN')
load_path = os.path.join(PATH, 'Saved_data')
if not os.path.exists(load_path): os.makedirs(load_path)

step = 1#len(scores[0][0])/10000
policies = [0,1,2,3,4,5,6,7]
# policies = [0,1,2,3]
# policies = [4,5,6,7]
# policies = [0,1]
# policies = [0,2]
# policies = [0,4]
# policies = [1,3]
# policies = [1,5]
# policies = [2,3]
# policies = [2,6]
# policies = [3,7]
# policies = [4,5]
# policies = [4,6]
# policies = [5,7]
# policies = [6,7]


rewards = []
tss, lrs, drs, ers, prs, nrs, pers, nsrs, mss = [], [], [], [], [], [], [], [], []
for i, policy in enumerate(policies):
    file_name = "rewards%s.json" %(str(policy))
    rewards.append(read_json(load_path, file_name))
    file_name = "hyperparameters%s.json" %(str(policy))
    ts, lr, dr, er, pr, nr, per, nsr, ms = read_hp_json(load_path, file_name)
    tss.append(float(ts))
    lrs.append(float(lr))
    drs.append(float(dr))
    ers.append(float(er))
    prs.append(float(pr))
    nrs.append(float(nr))
    pers.append(float(per))
    nsrs.append(float(nsr))
    mss.append(float(ms))


string = ""
string = [string+","+str(i) for i in policies]
filename = 'learning_cruves%s, step=%s.png' %(string, str(step))
filename = os.path.join(load_path, filename)

plot_learning_curves(rewards, filename, step, tss, prs, nrs, pers, nsrs, mss, lrs, drs, ers)
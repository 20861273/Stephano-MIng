import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.pyplot import cm
import os

def plot_learning_curves(x, scores, filename, ts, pr, nr, ms, lr, dr, er):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")

    l = []
    for i in range(len(scores)):
        l.append("%s:  e=%s, pr=%s, nr=%s, s=%s, α=%s, γ=%s, ϵ=%s" %(
                str(i),
                str(ts[i]),
                str(pr[i]),
                str(nr[i]),
                str(ms[i]),
                str(lr[i]), 
                str(dr[i]), 
                str(er[i])
                ))

    c = cm.rainbow(np.linspace(0, 1, len(scores)))
    step = len(scores[0])/100
    for i in range(len(policies)):
        ax.plot(np.arange(0, len(scores[i]), int(step+i)), scores[i][::int(step+i)], color=c[i])
    ax.legend(l)
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Rewards", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    np_scores = np.array(scores)
    # np_scores.min()-1
    ax.set_ylim(-20, np_scores.max()+1)
    ax.set_title("Learning curve:\nLearning rate: %s\nDiscount rate: %s\nEpsilon: %s" %(str(lr),str(dr), str(er)))

    plt.savefig(filename)

def read_json(path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as f:
        return json.load(f)

def read_hp_json(path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as f:
        txt = json.load(f)
    
    ts, lr, dr, er, pr, nr, ms, _, _ = txt.split(",")
    return ts, lr, dr, er, pr, nr, ms


PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'DQN')
load_path = os.path.join(PATH, 'Saved_data')
if not os.path.exists(load_path): os.makedirs(load_path)


policies = [1,3]
rewards = []
tss, lrs, drs, ers, prs, nrs, mss = [], [], [], [], [], [], []
for i, policy in enumerate(policies):
    file_name = "rewards%s.json" %(str(policy))
    rewards.append(read_json(load_path, file_name))
    file_name = "hyperparameters%s.json" %(str(policy))
    ts, lr, dr, er, pr, nr, ms = read_hp_json(load_path, file_name)
    tss.append(float(ts))
    lrs.append(float(lr))
    drs.append(float(dr))
    ers.append(float(er))
    prs.append(float(pr))
    nrs.append(float(nr))
    mss.append(float(ms))


x = [i+1 for i in range(2000)]


string = ""
string = [string+","+str(i) for i in policies]
filename = 'learning_cruves%s.png' %(string)
filename = os.path.join(load_path, filename)

plot_learning_curves(x, rewards, filename, tss, prs, nrs, mss, lrs, drs, ers)
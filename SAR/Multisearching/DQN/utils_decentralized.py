import matplotlib.pyplot as plt
import numpy as np
# import gym
import json
import os
from matplotlib.pyplot import cm
import math

def save_hp(save_path, nr, training_sessions, episodes,
            positive_rewards, negative_rewards, positive_exploration_rewards, negative_step_rewards,
            max_steps, learning_rate, discount_rate, epsilon, n_actions,
            c_dims, k_size, s_size, fc_dims, prioritized,
            batch_size, mem_size, replace, env_size):
    
    i_exp = 0
    for pr_i in positive_rewards:
            for nr_i in negative_rewards:
                for per_i in positive_exploration_rewards:
                    for nsr_i in negative_step_rewards:
                        for ms_i in max_steps:
                            for lr_i in learning_rate:
                                for dr_i in discount_rate:
                                    for er_i in epsilon:
                                        hp =    {
                                        "training_sessions":training_sessions,
                                        "nr":nr,
                                        "episodes":episodes,
                                        "learning_rate":lr_i,
                                        "discount_rate":dr_i,
                                        "epsilon":er_i,
                                        "positive_goal":pr_i,
                                        "negative_collision":nr_i,
                                        "negative_step":per_i,
                                        "positive_exploration":nsr_i,
                                        "max_steps":ms_i,
                                        "n_actions":n_actions,
                                        "c_dims":c_dims,
                                        "k_size":k_size,
                                        "s_size":s_size,
                                        "fc_dims":fc_dims,
                                        "prioritized":prioritized,
                                        "mem_size":mem_size,
                                        "batch_size":batch_size,
                                        "replace":replace,
                                        "env_size":env_size
                                        }
                                        file_name = "hyperparameters%s.json" %(str(i_exp))
                                        file_name = os.path.join(save_path, file_name)
                                        write_json(hp, file_name)
                                        i_exp += 1


def write_json(lst, file_name):
    with open(file_name, "w") as f:
        json.dump(lst, f)

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)

def read_checkpoint_hp_json(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    
    hps = []
    for key, value in data.items():
        try:
            hp = float(value)
            hps.append(hp)
        except ValueError:
            pass

    ts, ep = hps[:]
    return int(ts), int(ep)

def read_hp_json(path, file_name):
    lst = []
    file_path = os.path.join(path, file_name)
    with open(file_path, "r") as f:
        data = json.load(f)

    for key in data:
        lst.append(data[key])

        # nr, ep, 
#  int(nr), int(ep), 

    ts, nr, ep, lr, dr, er, pr, negr, per, nsr, ms, n_actions, c_dims, k_size, s_size, fc_dims, prioritized, mem_size, batch_size, replace,env_size = lst[:]
    return int(ts),int(nr), int(ep), float(lr), float(dr), er, float(pr), float(negr), float(per), float(nsr), int(ms), int(n_actions), c_dims, k_size, s_size, fc_dims, prioritized, int(mem_size), int(batch_size), int(replace), env_size

def plot_learning_curve(nr, scores, filename, lr, dr, er, pr, negr, per, nsr, ms, totle_time):
    mean_rewards = np.zeros((len(scores[0]),))
    std_rewards = np.zeros((len(scores[0]),))

    if len(scores) > 1:
        for i_ep in range(len(scores[0])):
            s = sum(scores[e][i_ep] for e in range(len(scores)))
            mean_rewards[i_ep] = s / len(scores)
    else:
        mean_rewards = scores[0]

    if len(scores) > 1:
        for i_ep in range(len(scores[0])):
            v = sum((scores[e][i_ep]-mean_rewards[i_ep])**2 for e in range(len(scores)))
            std_rewards[i_ep] = math.sqrt(v / (len(scores)-1))

    fig=plt.figure()
    l = "drone=%s\nα=%s,\nγ=%s,\nϵ=%s,\npositive reward=%s,\nnegative reward=%s,\npositive eexploration reward=%s,\nnegative step reward=%s,\nmax steps=%s" %(str(nr), str(lr), str(dr), str(er), str(pr), str(negr), str(per), str(nsr), str(ms))
    ax=fig.add_subplot(111)

    ax.plot(np.arange(0, len(mean_rewards), 1), mean_rewards[::1], color="C1", label=l)
    if len(scores) > 1:
        plt.fill_between(np.arange(0, len(mean_rewards), int(1)), \
            mean_rewards[::int(1)]-std_rewards[::int(1)], mean_rewards[::int(1)]+std_rewards[::int(1)], alpha = 0.1, color = 'b')
    # plt.fill_between(np.arange(0, len(mean_rewards), int(1)), \
    #     mean_rewards[::int(1)], mean_rewards[::int(1)]+std_rewards[::int(1)], alpha = 0.1, color = 'b')
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend()
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Rewards", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    ax.set_ylim(np.array(scores).min()-1, np.array(scores).max()+1)
    m, s = divmod(totle_time, 60)
    h = 0
    if m >= 60: h, m = divmod(m, 60)
    ax.set_title("Learning curve:\nTime: %sh%sm%ss" %(str(h), str(m), str(s)), fontsize = 10)

    plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_learning_curvess(x, scores, filename, pr, ms, lr, dr, er, total_time):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")

    l = []
    cnt = 0
    for pr_i in pr:
            for ms_i in ms:
                for lr_i in lr:
                    for dr_i in dr:
                        for er_i in er:
                            l.append("%s:  r=%s, s=%s, α=%s, γ=%s, ϵ=%s" %(
                                    str(cnt),
                                    str(pr[pr_i]),
                                    str(ms[ms_i]),
                                    str(lr[lr_i]), 
                                    str(dr[dr_i]), 
                                    str(er[er_i])
                                    ))
                            cnt += 1  

    c = cm.rainbow(np.linspace(0, 1, len(scores)))
    for i in range(3):
        ax.scatter(np.arange(0, len(scores[i]), 50), scores[i][::50], color=c[i])
    ax.legend(l)
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Rewards", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    ax.set_ylim(-1, 1)
    ax.set_title("Learning curve:\nLearning rate: %s\nDiscount rate: %s\nEpsilon: %s" %(str(lr),str(dr), str(er)))

    plt.savefig(filename)

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

# class SkipEnv(gym.Wrapper):
#     def __init__(self, env=None, skip=4):
#         super(SkipEnv, self).__init__(env)
#         self._skip = skip

#     def step(self, action):
#         t_reward = 0.0
#         done = False
#         for _ in range(self._skip):
#             obs, reward, done, info = self.env.step(action)
#             t_reward += reward
#             if done:
#                 break
#         return obs, t_reward, done, info

#     def reset(self):
#         self._obs_buffer = []
#         obs = self.env.reset()
#         self._obs_buffer.append(obs)
#         return obs

# class PreProcessFrame(gym.ObservationWrapper):
#     def __init__(self, env=None):
#         super(PreProcessFrame, self).__init__(env)
#         self.observation_space = gym.spaces.Box(low=0, high=255,
#                                                 shape=(80,80,1), dtype=np.uint8)
#     def observation(self, obs):
#         return PreProcessFrame.process(obs)

#     @staticmethod
#     def process(frame):

#         new_frame = np.reshape(frame, frame.shape).astype(np.float32)

#         new_frame = 0.299*new_frame[:,:,0] + 0.587*new_frame[:,:,1] + \
#                     0.114*new_frame[:,:,2]

#         new_frame = new_frame[35:195:2, ::2].reshape(80,80,1)

#         return new_frame.astype(np.uint8)

# class MoveImgChannel(gym.ObservationWrapper):
#     def __init__(self, env):
#         super(MoveImgChannel, self).__init__(env)
#         self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
#                             shape=(self.observation_space.shape[-1],
#                                 self.observation_space.shape[0],
#                                 self.observation_space.shape[1]),
#                             dtype=np.float32)

#     def observation(self, observation):
#         return np.moveaxis(observation, 2, 0)

# class ScaleFrame(gym.ObservationWrapper):
#     def observation(self, obs):
#         return np.array(obs).astype(np.float32) / 255.0

# class BufferWrapper(gym.ObservationWrapper):
#     def __init__(self, env, n_steps):
#         super(BufferWrapper, self).__init__(env)
#         self.observation_space = gym.spaces.Box(
#                             env.observation_space.low.repeat(n_steps, axis=0),
#                             env.observation_space.high.repeat(n_steps, axis=0),
#                             dtype=np.float32)

#     def reset(self):
#         self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
#         return self.observation(self.env.reset())

#     def observation(self, observation):
#         self.buffer[:-1] = self.buffer[1:]
#         self.buffer[-1] = observation
#         return self.buffer

# def make_env(env_name):
#     env = gym.make(env_name)
#     env = SkipEnv(env)
#     env = PreProcessFrame(env)
#     env = MoveImgChannel(env)
#     env = BufferWrapper(env, 4)
#     return ScaleFrame(env)

import numpy as np
import matplotlib.pyplot as plt

# action_space_size = 4
# state_space_size = 10

# q_table = np.array([[-1.25712730e+00, -5.03993882e+00, -5.26273049e+00, -3.00723174e+00],
#  [-1.25712730e+00, -5.03993882e+00, -5.26273049e+00, -3.00723174e+00],
#  [-4.46543461e+00, -4.18463722e+00, -3.90362485e+00, -1.28027900e+00],
#  [ 3.14287454e+00, -1.22024283e+00, -1.37342380e+00, -7.36638923e-01],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
#  [-1.38995183e-02, -2.12624589e-02,  7.01269566e-04,  1.32844861e-03],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]])



# state = np.array([1, 2, 0, 1, 0, 0, 1, 0, 0, 0])

# #print(q_table[state,:])
# #print(np.argmax(q_table[state]))
# #print(q_table[state,np.argmax(q_table[state,:])])


# a = np.array([[1,2,3],[4,5,6],[7,8,9]])

# #print(a[2,:])
# #print(np.argmax(a[2,:]))

# q_table = np.array([[-14.35498801 ,-14.93186375 ,-13.42812387 ,-14.18855792],
#                 [-28.43328595 ,-37.44187336 ,-28.96565213 ,-30.50002213],
#                 [ -4.63503213 , -9.03421314, -10.37345045 , -8.41597758],
#                 [ -2.782989   , -3.12916642 , -1.1074246 ,  -2.6432362 ],
#                 [ -0.9      ,   -2.90129646 ,  0.      ,     0.        ],
#                 [  0.       ,    0. ,          0.       ,    0.        ],
#                 [  0.        ,   0.,           0.      ,     0.        ],
#                 [  0.         ,  0.           ,0.     ,      0.        ],
#                 [  0.          , 0.          , 0.    ,       0.        ],
#                 [  0. ,          0.         ,  0. ,          0.        ],
#                 [  0.    ,       0.        ,   0.   ,        0.        ],
#                 [  0.   ,        0.       ,    0.,           0.        ],
#                 [  0.   ,        0.      ,     0.           ,0.        ],
#                 [  0.   ,        0.     ,      0.          , 0.        ],
#                 [  0.    ,       0.    ,       0.         ,  0.        ],
#                 [  0.  ,         0.   ,        0.        ,   0.        ],
#                 [  0.   ,        0.  ,         0.       ,    0.        ],
#                 [  0.   ,        0. ,          0.      ,     0.        ],
#                 [  0.    ,       0.,           0.     ,      0.        ],
#                 [  0.    ,       0.           ,0.    ,       0.        ],
#                 [  0.    ,       0.          , 0.   ,        0.        ],
#                 [  0.    ,       0.         ,  0.  ,         0.        ],
#                 [  0.   ,        0.        ,   0. ,          0.        ],
#                 [  0.   ,        0.       ,    0.,           0.        ],
#                 [  0.  ,         0.      ,     0.           ,0.        ],
#                 [  0.  ,         0.     ,      0.          , 0.        ],
#                 [  0.   ,        0.    ,       0.         ,  0.        ],
#                 [  0.  ,         0.   ,        0.        ,   0.        ],
#                 [  0.   ,        0.  ,         0.       ,    0.        ],
#                 [  0.   ,        0. ,          0.      ,     0.        ],
#                 [  0.   ,        0.,           0.     ,      0.        ],
#                 [  0.   ,        0.       ,    0.    ,       0.        ],
#                 [  0.   ,        0.      ,     0.   ,        0.        ],
#                 [  0.    ,       0.     ,      0.  ,         0.        ],
#                 [  0.    ,       0.    ,       0. ,          0.        ],
#                 [  0.   ,        0.   ,        0.,           0.        ]])



# state = np.array([2,3])

# print(q_table[state,0])
# print(np.argmax(q_table[state,:]))
# print(np.argmax(q_table[state]) % 4)


# rewards = [[[1,2,3],[4,5,6]]]
# seq_rewards = []
# cur_rewards = [10,11,12]

# # After every simulation
# tmp_seq_rewards = np.array(seq_rewards)
# new_tmp_seq_rewards = np.array(np.append(tmp_seq_rewards.ravel(),np.array(cur_rewards)))
# if tmp_seq_rewards.shape[0] == 0:
#     new_seq_rewards = new_tmp_seq_rewards.reshape(1,3)
# else:
#     new_seq_rewards = new_tmp_seq_rewards.reshape(tmp_seq_rewards.shape[0]+1,tmp_seq_rewards.shape[1])

# seq_rewards = new_seq_rewards.tolist()

# tmp_seq_rewards = np.array(seq_rewards)
# new_tmp_seq_rewards = np.array(np.append(tmp_seq_rewards.ravel(),np.array(cur_rewards)))
# if tmp_seq_rewards.shape[0] == 0:
#     new_seq_rewards = new_tmp_seq_rewards.reshape(1,3)
# else:
#     new_seq_rewards = new_tmp_seq_rewards.reshape(tmp_seq_rewards.shape[0]+1,tmp_seq_rewards.shape[1])

# seq_rewards = new_seq_rewards.tolist()

# # After every sequence
# tmp_rewards = np.array(rewards)
# new_tmp_rewards = np.array(np.append(tmp_rewards.ravel(),new_seq_rewards.ravel()))
# if tmp_rewards.shape[0] == 0:
#     new_rewards = new_tmp_rewards.reshape(1,1,tmp_rewards.shape[2])
# else:
#     new_rewards = new_tmp_rewards.reshape(tmp_rewards.shape[0]+1,tmp_rewards.shape[1],tmp_rewards.shape[2])

# rewards = new_rewards.tolist()

# print(rewards)

# rewards = [0,1,2,3,4,5,6,7,8,9]

# fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# fig.suptitle("Avg rewards")

# ax[0].plot(np.arange(0, len(rewards)), rewards)
# ax[0].set_title('Rewards per episode')
# ax[0].set_xlabel('Episode')
# ax[0].set_ylabel('Rewards')

# plt.show()


rewards = [ [[0,1],
            [2,3]],

            [[4,5],
            [6,7]] ]

print(rewards)
avg_rewards = np.zeros((4))
print(avg_rewards)
avg_cnt = 0

for seq_i in range(0, 2):
    if avg_cnt % 4 == 0: avg_cnt = 0
    seq_rewards = []
    seq_steps = []
    for lr_i in range(0, 2):
        for dr_i in range(0, 2):
            avg_rewards[avg_cnt] += np.array(rewards)[seq_i, lr_i, dr_i]
            avg_cnt += 1

for i in range(len(avg_rewards)):
    avg_rewards[i] = avg_rewards[i]/2

print(avg_rewards)
line = "hello"
print(line[0:4])
import numpy as np

c = np.zeros((3,2,2,2))
cur_r_pos = np.array([[[1,2],[3,4]],[[6,2],[3,9]]])

print(c)

tmp_r_pos = np.array(c[:1])
new_pos = np.array(np.append(np.append(tmp_r_pos.ravel(),cur_r_pos.ravel()),c[1:3]))
print(c.ravel(), tmp_r_pos.ravel(), cur_r_pos.ravel(),new_pos)


c = new_pos.reshape(c.shape[0],c.shape[1]+1,c.shape[2],c.shape[3])


print(c)


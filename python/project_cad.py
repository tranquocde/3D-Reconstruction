import numpy as np
import matplotlib.pyplot as plt
from submission import estimate_pose , estimate_params
# write your implementation here
data = np.load('/Users/quocdetran/Documents/HUST/CMU-16385-CV-fall-2022/assgn3/data/pnp.npz',allow_pickle=True)
image = data['image']
cad = data['cad']
x = data['x']

X = data['X']
N,_= X.shape

P = estimate_pose(x,X)
K,R,t = estimate_params(P)

xp = P @ np.hstack((X, np.ones((N,1)))).T
xp = xp[:2,:].T / np.vstack((xp[2,:], xp[2,:])).T 

fig ,axs = plt.subplots()
axs.imshow(image)
x_coords , y_coords = zip(*xp)
axs.scatter(x_coords,y_coords,color='red')
plt.savefig('plane.png')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from submission import estimate_pose , estimate_params
# write your implementation here
data = np.load('/Users/quocdetran/Documents/HUST/CMU-16385-CV-fall-2022/assgn3/data/pnp.npz',allow_pickle=True)
image = data['image']
cad = data['cad']
x = data['x']

X = data['X'] #30,3

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


temp = cad[0][0]
vertices = temp[0]
faces = temp[1]
faces = faces - 1

rotated_vertices = vertices@R.T


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], triangles=faces)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points')

plt.show()

from matplotlib.patches import Polygon

N,_ = vertices.shape
projected_vertices = P @ np.hstack((vertices, np.ones((N,1)))).T
projected_vertices = projected_vertices[:2,:].T / np.vstack((projected_vertices[2,:], projected_vertices[2,:])).T 
fig ,axs = plt.subplots()
for face in faces:
    polygon = Polygon(projected_vertices[face, :2], closed=True, fill=True)
    axs.add_patch(polygon)
axs.imshow(image)
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
ax.set_aspect('equal', adjustable='box')

# Set labels and title
axs.set_xlabel('X')
axs.set_ylabel('Y')
axs.set_title('CAD Model Projection')
plt.savefig('plane_projected.png')
plt.show()
import numpy as np
data = np.load('/Users/quocdetran/Documents/HUST/CMU-16385-CV-fall-2022/assgn3/data/pnp.npz',allow_pickle=True)
cad = data['cad']
temp = cad[0][0]
vertices = temp[0]
faces = temp[1]
faces = faces - 1
print(vertices.shape,faces.shape)
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points')

plt.show()



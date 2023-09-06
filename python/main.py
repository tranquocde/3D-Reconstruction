import numpy as np
import matplotlib.pyplot as plt
img1 = plt.imread("/Users/quocdetran/Documents/HUST/CMU-16385-CV-fall-2022/assgn3/data/im1.png")
img2 = plt.imread("/Users/quocdetran/Documents/HUST/CMU-16385-CV-fall-2022/assgn3/data/im1.png")
M = max([img1.shape[0],img1.shape[1]])

data = np.load("/Users/quocdetran/Documents/HUST/CMU-16385-CV-fall-2022/assgn3/data/some_corresp.npz")

from helper import displayEpipolarF , epipolarMatchGUI
from submission import eight_point , epipolar_correspondences
ps1 = data['pts1']
ps2 = data['pts2']
F = eight_point(pts1=ps1,pts2=ps2,M=M)
epipolarMatchGUI(img1,img2,F)

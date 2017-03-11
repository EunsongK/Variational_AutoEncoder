import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


incept_pm_path = "/media/wltjr1007/hdd/personal/brats/output/t_hl_pm_incept_cnn.dat"
orig_pm_path = "/media/wltjr1007/hdd/personal/brats/output/t_hl_pm_orig_cnn.dat"

incept_pm_data = np.memmap(filename=incept_pm_path, mode="r", dtype=np.float32, shape=(110,240,240,155, 5))[44]
orig_pm_data = np.memmap(filename=orig_pm_path, mode="r", dtype=np.float32, shape=(110,240,240,155, 9))[44]


ax1 = plt.subplot(3, 4, 1)
ax2 = plt.subplot(3, 4, 2)
ax3 = plt.subplot(3, 4, 3)
ax4 = plt.subplot(3, 4, 4)

bx1 = plt.subplot(3,6,7)
bx2 = plt.subplot(3,6,8)
bx3 = plt.subplot(3,6,9)
bx4 = plt.subplot(3,6,10)
bx5 = plt.subplot(3,6,11)
bx6 = plt.subplot(3,6,12)

cx1 = plt.subplot(3,6,13)
cx2 = plt.subplot(3,6,14)
cx3 = plt.subplot(3,6,15)
cx4 = plt.subplot(3,6,16)
cx5 = plt.subplot(3,6,17)
cx6 = plt.subplot(3,6,18)

ims = []
fig = plt.figure(facecolor="white", figsize=(22,12))
for i in range(155):
   a1=ax1.imshow(orig_pm_data[...,i,0], cmap="gray", animated=True)
   a2=ax2.imshow(orig_pm_data[...,i,1], cmap="gray", animated=True)
   a3=ax3.imshow(orig_pm_data[...,i,2], cmap="gray", animated=True)
   a4=ax4.imshow(orig_pm_data[...,i,3], cmap="gray", animated=True)

   b1 = bx1.imshow(orig_pm_data[...,i,4], vmin=0, vmax=1, animated=True)
   b2 = bx2.imshow(orig_pm_data[...,i,5], vmin=0, vmax=1, animated=True)
   b3 = bx3.imshow(orig_pm_data[...,i,6], vmin=0, vmax=1, animated=True)
   b4 = bx4.imshow(orig_pm_data[...,i,7], vmin=0, vmax=1, animated=True)
   b5 = bx5.imshow(orig_pm_data[...,i,8], vmin=0, vmax=1, animated=True)
   b6 = bx6.imshow(np.argmax(orig_pm_data[...,i,4:],axis=-1), vmin=0, vmax=4, animated=True)

   c1 = cx1.imshow(incept_pm_data[...,i,0], vmin=0, vmax=1, animated=True)
   c2 = cx2.imshow(incept_pm_data[...,i,1], vmin=0, vmax=1, animated=True)
   c3 = cx3.imshow(incept_pm_data[...,i,2], vmin=0, vmax=1, animated=True)
   c4 = cx4.imshow(incept_pm_data[...,i,3], vmin=0, vmax=1, animated=True)
   c5 = cx5.imshow(incept_pm_data[...,i,4], vmin=0, vmax=1, animated=True)
   c6 = cx6.imshow(np.argmax(incept_pm_data[...,i,:],axis=-1), vmin=0, vmax=4, animated=True)

   ims.append([a1,a2,a3,a4,b1,b2,b3,b4,b5,b6,c1,c2,c3,c4,c5,c6])

ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")

bx1.axis("off")
bx2.axis("off")
bx3.axis("off")
bx4.axis("off")
bx5.axis("off")
bx6.axis("off")

cx1.axis("off")
cx2.axis("off")
cx3.axis("off")
cx4.axis("off")
cx5.axis("off")
cx6.axis("off")
#MOD = {"T1": 0, "T2": 1, "T1c": 2, "Flair": 3, "OT": 4}

ax1.set_title("T1")
ax2.set_title("T2")
ax3.set_title("T1c")
ax4.set_title("T2 Flair")

bx1.set_title("Orig CNN label 1")
bx2.set_title("Orig CNN label 2")
bx3.set_title("Orig CNN label 3")
bx4.set_title("Orig CNN label 4")
bx5.set_title("Orig CNN label 5")
bx6.set_title("Orig CNN prediction")

cx1.set_title("Proposed CNN label 1")
cx2.set_title("Proposed CNN label 2")
cx3.set_title("Proposed CNN label 3")
cx4.set_title("Proposed CNN label 4")
cx5.set_title("Proposed CNN label 5")
cx6.set_title("Proposed CNN prediction")





ani = animation.ArtistAnimation(fig=fig, artists=ims, interval=50, blit=True)
plt.show()
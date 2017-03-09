import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from matplotlib.colors import ListedColormap
from medpy.io import load

fig = plt.figure(facecolor="white", figsize=(22.0, 12.0))
a = fig.add_subplot(231)
b = fig.add_subplot(232)
c = fig.add_subplot(233)
d = fig.add_subplot(245)
e = fig.add_subplot(246)
f = fig.add_subplot(247)
g = fig.add_subplot(248)
t_ax = fig.add_axes([0.35, 0.85, 0.30, 0.15])
cmap = mpl.colors.ListedColormap([[0., 1, 1.], [0., 0.66, 1.], [0., .33, 1.], [0., .0, 1.]])
cmap2 = mpl.colors.ListedColormap([[0., 1, 1.], [0., 0.66, 1.], [0., .33, 1.], [0., .0, 1.]])
ims = []


def make_binary(input, cls):
    result = input.copy()
    result[result != cls] = 0
    return result


imgname = ["/media/wltjr1007/hdd1/personal/brats/data/h.48.VSD.Brain.XX.O.MR_T1c.35621.nii",
           "/media/wltjr1007/hdd1/personal/brats/data/h.48.VSD.Brain.XX.O.MR_Flair.35619.nii",
           "/media/wltjr1007/hdd1/personal/brats/data/h.48.VSD.Brain.XX.O.MR_T1.35622.nii",
           "/media/wltjr1007/hdd1/personal/brats/data/h.48.VSD.Brain.XX.O.MR_T2.35620.nii"]
gt_label = load("/media/wltjr1007/hdd1/personal/brats/data/h.48.VSD.Brain_3more.XX.O.OT.42345.nii")[0]
predict_label = load("/media/wltjr1007/hdd1/personal/brats/output/testoutput/VSD.h.47.35619.nii")[0]
MODS = ["T1c", "T2 Flair", "T1", "T2"]
for j in range(4):
    curimg = load(imgname[j])[0]
    for i in range(curimg.shape[2]):
        font_size = int(mpl.font_manager.FontProperties().get_size())
        tempimg = Image.new('RGBA', (200, 50))
        imgdraw = ImageDraw.Draw(tempimg)
        font = ImageFont.truetype("Times_New_Roman_Normal.ttf", size=font_size)
        imgdraw.text((0, 0), "Prediction masked onto: " + MODS[j], fill=(0, 0, 0, 255), font=font)
        resultimg = np.asarray(tempimg).copy()
        tt = t_ax.imshow(resultimg, animated=True)
        im14 = d.imshow(np.rot90(curimg[:, :, i]), cmap='Greys_r', animated=True)
        im15 = e.imshow(np.rot90(curimg[:, :, i]), cmap='Greys_r', animated=True)
        im16 = f.imshow(np.rot90(curimg[:, :, i]), cmap='Greys_r', animated=True)
        im17 = g.imshow(np.rot90(curimg[:, :, i]), cmap='Greys_r', animated=True)
        im1 = a.imshow(np.rot90(curimg[:, :, i]), cmap='Greys_r', animated=True)
        im2 = b.imshow(np.rot90(curimg[:, :, i]), cmap='Greys_r', animated=True)
        im3 = b.imshow(np.rot90(np.ma.masked_where(gt_label[:, :, i] == 0, gt_label[:, :, i])), cmap=cmap,
                       animated=True, alpha=0.4)
        im4 = c.imshow(np.rot90(curimg[:, :, i]), cmap='Greys_r', animated=True)
        im5 = c.imshow(np.rot90(np.ma.masked_where(predict_label[:, :, i] == 0, predict_label[:, :, i])), cmap=cmap,
                       animated=True, alpha=0.4)
        im6 = d.imshow(np.rot90(np.ma.masked_where(gt_label[:, :, i] != 1, make_binary(gt_label[:, :, i], 1))),
                       animated=True, alpha=0.4)
        im7 = d.imshow(
            np.rot90(np.ma.masked_where(predict_label[:, :, i] != 1, make_binary(predict_label[:, :, i], 1))),
            cmap=cmap, animated=True, alpha=0.4)
        im8 = e.imshow(np.rot90(np.ma.masked_where(gt_label[:, :, i] != 2, make_binary(gt_label[:, :, i], 2))),
                       animated=True, alpha=0.4)
        im9 = e.imshow(
            np.rot90(np.ma.masked_where(predict_label[:, :, i] != 2, make_binary(predict_label[:, :, i], 2))),
            cmap=cmap, animated=True, alpha=0.4)
        im10 = f.imshow(np.rot90(np.ma.masked_where(gt_label[:, :, i] != 3, make_binary(gt_label[:, :, i], 3))),
                        animated=True, alpha=0.4)
        im11 = f.imshow(
            np.rot90(np.ma.masked_where(predict_label[:, :, i] != 3, make_binary(predict_label[:, :, i], 3))),
            cmap=cmap, animated=True, alpha=0.4)
        im12 = g.imshow(np.rot90(np.ma.masked_where(gt_label[:, :, i] != 4, make_binary(gt_label[:, :, i], 4))),
                        animated=True, alpha=0.4)
        im13 = g.imshow(
            np.rot90(np.ma.masked_where(predict_label[:, :, i] != 4, make_binary(predict_label[:, :, i], 4))),
            cmap=cmap, animated=True, alpha=0.4)
        ims.append([tt, im14, im15, im16, im17, im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12, im13])
a.axis("off")
t_ax.axis("off")
b.axis("off")
c.axis("off")
d.axis("off")
e.axis("off")
f.axis("off")
g.axis("off")
a.set_title("Input image")
b.set_title("Manual")
c.set_title("Prediction")
d.set_title("Necrosis")
e.set_title("Edema")
f.set_title("Non-enhancing")
g.set_title("Enhancing")

ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=13, metadata=dict(artist='Yoon, Jee Seok (Korea University MiLab)'), bitrate=1800)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(ims[0][7], cmap=cmap, cax=cbar_ax, ticks=[-0.075, -0.025, 0.025, 0.075])
cbar.ax.set_yticklabels(["Necrosis/Predict", "Edema", "Non-enhancing", "Enhancing/Manual"])
ani.save("test.mov", writer=writer, dpi=300)
plt.show()
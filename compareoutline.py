import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

in_folder = r"C:\temporary work folder gsk35\UROP tests 1\fiji\outputs\output5\GUIDE-0421-051V"
img_cad = cv2.imread(os.path.join(in_folder, "image0.tif"), cv2.IMREAD_GRAYSCALE)
# plt.imshow(img_cad > 200)
# plt.show()
count = 0

# M = np.array([[0.901261629485492, -0.433275288029627, 744.8021128049814], [0.433275288029627, 0.901261629485492, -1509.2875464255246]])
# M = np.array([[0.899603050234375, -0.436708543549366, 1348.6343153960097], [0.436708543549366, 0.899603050234375, -853.3164883280783]])
M = np.array([[0.900982942613293, -0.433854511466565, 1341.1840585028744], [0.433854511466565, 0.900982942613293, -853.4914923855735]])

plt.title("")
for file in os.scandir(in_folder):
    if len(file.name) == 7:
        img = cv2.imread(os.path.join(in_folder, file.name), cv2.IMREAD_COLOR)
        img_warp = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        img_warp[img_cad > 200] = (100,100,100)

        ax = plt.subplot(321 + count)
        ax.set_xlim((2600,3050))
        ax.set_ylim((3100,2750)) # inverted limits to prevent image from being flipped
        ax.set_title(file.name)
        ax.imshow(img_warp)
        count += 1
plt.show()

# img = cv2.imread(os.path.join(in_folder, "im6.tif"), cv2.IMREAD_COLOR)
# img_warp = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# img_warp[img_cad > 200] = (100,100,100)
# plt.imshow(img_warp)
# plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

ref_img = cv2.imread('fruitDataset/Orange/orange_52.jpg')
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

many_fruits_img = cv2.imread('fruitDataset/Banana/orange_51.jpg')
many_fruits_img = cv2.cvtColor(many_fruits_img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(3, 2, figsize=(16, 16))

# Reference Image
axes[0, 0].imshow(ref_img)
axes[0, 0].set_title('Reference Image')
axes[0, 0].axis('off')

# Many Fruits Image
axes[0, 1].imshow(many_fruits_img)
axes[0, 1].set_title('Many Fruits Image')
axes[0, 1].axis('off')

gray_ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)

# Gray Image
axes[1, 0].imshow(gray_ref_img, cmap='gray')
axes[1, 0].set_title('Gray Image')
axes[1, 0].axis('off')

bin_gray_ref_img = cv2.adaptiveThreshold(gray_ref_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

# Binary Image
axes[1, 1].imshow(bin_gray_ref_img, cmap='gray')
axes[1, 1].set_title('Binary Image')
axes[1, 1].axis('off')

inv_bin_gray_ref_img = cv2.bitwise_not(bin_gray_ref_img)

# Inverted Binary Image
axes[2, 0].imshow(inv_bin_gray_ref_img, cmap='gray')
axes[2, 0].set_title('Inverted Binary Image')
axes[2, 0].axis('off')

ref_contours_list, hierarchy = cv2.findContours(inv_bin_gray_ref_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

with_contours = cv2.drawContours(ref_img.copy(), ref_contours_list, -1, (255, 0, 0), 5)

# Detection With Red Boundary
axes[2, 1].imshow(with_contours)
axes[2, 1].set_title('Detection With Red Boundary')
axes[2, 1].axis('off')

plt.tight_layout()
plt.show()

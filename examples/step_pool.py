import torch
import torch.nn as nn
import cv2
import numpy as np

# class stepPool(nn.Module):
# #     def __init__(self,):
# #         super(stepPool, self).__init__()
# #         self.set_pool = nn.AvgPool2d(kernel_size=)

img = cv2.imread('cat.jpg')
img = torch.Tensor(img)
img = np.int32(img)
cv2.imshow('1',img)
cv2.waitKey()
img = img.unsqueeze(0)
print(img.size())
img = img.permute(0,3, 1, 2)
print(img.size())
if 1==2:
    step_pool = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
    img = step_pool(img)
    img = torch.floor(img)
newImg = img[0].permute(1, 2, 0).numpy()
# cv2.imshow('1',newImg)
# cv2.waitKey()
print(img.size())
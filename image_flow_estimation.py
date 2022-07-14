import cv2
import numpy as np
from imread_from_url import imread_from_url

from gmflownet import GMFlowNet

# Initialize model
model_path = 'models/gmflownet_kitti_20_480x640.onnx'
flow_estimator = GMFlowNet(model_path)

# Read inference image
img1 = imread_from_url("https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/frame1.png?raw=true")
img2 = imread_from_url("https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/frame2.png?raw=true")

# Estimate flow and colorize it
flow_map = flow_estimator(img1, img2)
flow_img = flow_estimator.draw_flow()

combined_img = np.vstack((img1, flow_img))

cv2.namedWindow("Estimated GMFlowNet", cv2.WINDOW_NORMAL)
cv2.imshow("Estimated GMFlowNet", combined_img)
cv2.imwrite("kitti.jpg", combined_img)
cv2.waitKey(0)

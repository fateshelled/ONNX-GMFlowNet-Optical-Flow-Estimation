import cv2
import numpy as np
import time

from gmflownet import GMFlowNet

# Initialize model
model_path='models/gmflownet_things_20_480x640.onnx'
flow_estimator = GMFlowNet(model_path)

FLOW_FRAME_OFFSET = 1 # Number of frame difference to estimate the optical flow

# Initialize video
cap = cv2.VideoCapture(0)

cv2.namedWindow("Estimated GMFlowNet", cv2.WINDOW_NORMAL)
frame_list = []
frame_num = 0
while cap.isOpened():

    # Read frame from the video
    ret, prev_frame = cap.read()
    frame_list.append(prev_frame)
    if not ret:
        break

    # Skip the first frames to be able to
    frame_num += 1
    if frame_num <= FLOW_FRAME_OFFSET:
        continue

    t = time.time()
    flow_map = flow_estimator(frame_list[-1], frame_list[0])
    dt = time.time() - t
    print(f"Inference: {1 / dt:.3f} FPS")
    flow_img = flow_estimator.draw_flow()

    combined_img = np.hstack((frame_list[-1], flow_img))
    cv2.imshow("Estimated GMFlowNet", combined_img)

    # Remove the oldest frame
    frame_list.pop(0)

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

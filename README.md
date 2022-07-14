# ONNX-GMFlowNet-Optical-Flow-Estimation
Python scripts for performing optical flow estimation using the GMFlowNet model in ONNX.

- Original Video [https://youtu.be/3wdsE1UgP6k](https://youtu.be/3wdsE1UgP6k)

  ![gmflownet](https://user-images.githubusercontent.com/53618876/179008754-f46cfe1c-38d8-4814-b6b5-0e5b37b1e587.gif)

- Estimation on the KITTI dataset images from OpticalFlowToolkit example [frame1](https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/frame1.png) and [frame2](https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/frame2.png)

  ![kitti](https://user-images.githubusercontent.com/53618876/179009789-c4ad7093-4bbc-4840-80b5-cdddf725313e.jpg)

## Requirements
- opencv-python
- onnxruntime or onnxruntime-gpu

### image_flow_estimation.py
```bash
python3 -m pip install imread-from-url
```

### video_flow_estimation.py
```bash
python3 -m pip install youtube_dl
python3 -m pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

## ONNX model

The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/306_GMFlowNet) and save them into the **[models](https://github.com/fateshelled/ONNX-GMFlowNet-Optical-Flow-Estimation/tree/main/models)** folder.


## References:

- Base Code: https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation
- GMFlowNet model: https://github.com/xiaofeng94/GMFlowNet
- PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
- OpticalFlowToolkit toolkit: https://github.com/liruoteng/OpticalFlowToolkit
- Original paper: https://arxiv.org/abs/2203.11335

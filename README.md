# Comparison experiment between NanoDet and Tiny YOLOv4 & MobileNet YOLOv4 & EfficientNet YOLO3

#### Equipment & Environment

The experiment used CentOS 7.8 64-bit operating system, PyTorch deep learning framework, CUDA 10.2 parallel computing engine for accelerated computing, Python 3.7.13, Torch1.10.1, Opencv 3.4.9 and other deep learning related toolkits, and NVIDIA T4 GPU with 16 GB video memory.

#### Dataset

VOC dataset : The default parameters of train.py are used to train the VOC dataset.

#### Train

When training the model, the resolution of the input image is 416×416 pixels. The entire training process includes 300 epochs, and the stochastic gradient descent (SGD) with momentum is selected as the optimization algorithm. The algorithm adjusts the stability of the learning process through the momentum parameter to adapt to different convergence situations. The momentum parameter is set to 0.9.

#### Model Evaluation

In this experiment, four detection models, Tiny YOLOv4, MobileNet YOLOv4, EfficientNet YOLO3 and NanoDet, were evaluated based on the VOC test set, aiming to measure the model performance through two evaluation indicators: mean Average Precision (mAP) and frames per second (fps).

##### mAP :

###### Tiny YOLOv4

<img width="1739" height="1160" alt="tiny_mAP" src="https://github.com/user-attachments/assets/a7dd935a-eb60-440c-b467-8ed1a6e81fc7" />

###### MobileNet YOLOv4

<img width="1742" height="1187" alt="mobilenet_mAP" src="https://github.com/user-attachments/assets/29e3522c-70ff-4894-8637-c1880fdc4407" />

###### EfficientNet YOLO3

<img width="1745" height="1168" alt="efficientnet_mAP" src="https://github.com/user-attachments/assets/cb7e509e-9f47-46d7-9333-b7f77b591e60" />

###### NanoDet

73.84%

##### fps :

###### Tiny YOLOv4

<img width="1071" height="569" alt="tiny_fps" src="https://github.com/user-attachments/assets/e1a0420c-c899-48cc-ba12-b9a305ec6bd8" />

###### MobileNet YOLOv4

<img width="1056" height="475" alt="mobile_fps" src="https://github.com/user-attachments/assets/a178a08a-79c5-488e-8851-5415b4710280" />

###### EfficientNet YOLO3

<img width="1026" height="536" alt="efficient_fps" src="https://github.com/user-attachments/assets/07acf322-82c8-4a4a-8c06-435a702da73d" />

###### NanoDet

97

#### Conclusion Analysis

Tiny YOLOv4 achieved a mAP of 78.41% on the test set, indicating that it has a high accuracy rate in the object detection task. At the same time, it runs at a high frame rate of 139 fps, showing excellent real-time processing capabilities. This result shows that Tiny YOLOv4 can adapt to application scenarios that require fast processing while ensuring high detection accuracy.

Although MobileNet YOLOv4 has a slightly higher mAP than Tiny YOLOv4, reaching 80.82%, its fps is 72, slightly lower than Tiny YOLOv4. This means that MobileNet YOLOv4 has made some improvements in model structure or parameter optimization to improve detection accuracy, but sacrificed some processing speed.

EfficientNet YOLO3 showed the highest mAP of 83.04% among all evaluated models, showing its excellent performance in object detection tasks. However, EfficientNet YOLO3 has a fps of 43, the lowest among all models, which indicates that while it achieves high accuracy, it has a high demand for computing resources and is not suitable for applications with extremely high real-time requirements.

NanoDet has a mAP of 73.84%, the lowest among all models, but its fps is 97, providing a faster processing speed. Based on this result, it can be shown that NanoDet may have optimized the model size and computing efficiency, which is suitable for resource-constrained environments, but compromises in accuracy.

The selection of the most suitable model needs to be based on the specific application scenario and requirements. For scenarios that require fast detection and do not require extremely high accuracy, Tiny YOLOv4 may be a better choice. If the application scenario can tolerate a lower processing speed but has higher requirements for accuracy, EfficientNet YOLO3 will be a more suitable option. For resource-constrained environments, NanoDet may be a compromise solution.

#### Acknowledgement

This project reproduces and builds upon several existing works :

- [guo-pu/NanoDet-PyTorch](https://github.com/guo-pu/NanoDet-PyTorch)  
  ↳ Author: [guo-pu](https://github.com/guo-pu)


- https://github.com/bubbliiiing/efficientnet-yolo3-pytorch  
  ↳ Author: [bubbliiiing](https://github.com/bubbliiiing)

- https://github.com/bubbliiiing/mobilenet-yolov4-pytorch  
  ↳ Author: [bubbliiiing](https://github.com/bubbliiiing)

- https://github.com/bubbliiiing/yolov4-tiny-pytorch  
  ↳ Author: [bubbliiiing](https://github.com/bubbliiiing)

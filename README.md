# Object-Detection-in-Video-Surveillance-System
This project utilizes deep learning techniques, specifically the YOLO v3 algorithm and convolutional neural networks (CNNs), to perform real-time object detection in video surveillance systems. The implementation includes various functionalities encapsulated within the detector.py file, which is responsible for key tasks such as object detection, non-maxima suppression, and bounding box generation.

In addition to the core detection functionality, a graphical user interface (GUI) has been developed using Tkinter, implemented in the main2.py file. This GUI provides a user-friendly interface for real-time object detection, classification, and tracking within video streams. The GUI offers intuitive controls and visualization tools to enhance the user experience.

The system also incorporates comprehensive evaluation metrics to assess the performance of the object detection model. Upon detection and classification, the system generates three informative bar plot graphs:

#Accuracy: Illustrating the accuracy of the object detection model in identifying and classifying objects within the video stream.
#Mean Squared Error (MSE): Quantifying the disparity between the predicted bounding boxes and ground truth annotations, providing insights into the model's precision.
#Peak Signal-to-Noise Ratio (PSNR): Evaluating the quality of the detected objects by comparing them to reference frames, aiding in the assessment of image fidelity.
This repository serves as a comprehensive solution for real-time object detection in video surveillance scenarios, offering a blend of state-of-the-art deep learning techniques, intuitive user interfaces, and robust performance evaluation metrics. Whether for research, development, or practical applications, this project provides a powerful tool for enhancing video surveillance systems with advanced object detection capabilities.

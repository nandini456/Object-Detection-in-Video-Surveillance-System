import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(20)

class Detector:
    def show_bar_graph(self, values, labels, ylabel, color, x1, y1,width,label):
        plt.clf()  # Clear the previous plot
        bars = plt.bar(labels, values, color=color, alpha=0.7, width=width, label=label)
        plt.title(f'Object Detection {ylabel}')
        plt.xlabel('Class Label')
        plt.ylabel(ylabel)
        plt.ylim([x1, y1])
        plt.xticks(rotation=45, ha='right')  # Display class labels on x-axis
        plt.tight_layout()
        plt.legend()
    

    def compute_psnr_per_class(self, images, class_labels):
        psnr_per_class = []
        non_zero_labels = []

        for label in class_labels:
            indices = [i for i, x in enumerate(class_labels) if x == label]
            # Ensure there are at least two frames for PSNR calculation
            if len(indices) < 2:
                continue
            # Check if the indices are valid and within the range of images
            valid_indices = [i for i in indices if i < len(images)]
            if not valid_indices:
                continue

            class_images = [images[i] for i in valid_indices]
            class_psnr = self.calculate_psnr(class_images)

            # Only append non-zero PSNR values
            if class_psnr > 0.0:
                psnr_per_class.append(class_psnr)
                non_zero_labels.append(label)

        return psnr_per_class, non_zero_labels


    def calculate_psnr(self, images):
        mse = np.mean((images[0] - images[1]) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    def compute_mse_per_class(self, confidences, class_labels):
        mse_per_class = []
        for label in class_labels:
            indices = [i for i, x in enumerate(class_labels) if x == label]
            class_confidences = [confidences[i] for i in indices]
            class_mse = np.mean((np.array(class_confidences) - np.mean(class_confidences))**2)
            mse_per_class.append(class_mse)
        return mse_per_class

    #construction to initialise the parameters,assigning them to the class void variables self 
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        #The __init__ method is the constructor of the class. It is automatically called when an instance of the class is
        #  created.It takes four parameters: videoPath, configPath, modelPath, and classesPath.
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        
        #####################################

        #initialise the network, as self.net and input size is passed which is the frame on which the video is framed,scaling is done between 1.0 and 127,5
        #mean value is subtracted against each channel across all three channels
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        #reading the class labels from the folder coco.names
        
        self.readClasses()

    def readClasses(self):
        # Storing all the entries as a list in the self.classesList
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
    
        # Insert a placeholder '__Background__' at the beginning of the classes list
        self.classesList.insert(0, '__Background__')
    
        # Generate a random color for each class and store in self.colorList
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))


    def onVideo(self):
        #opening video
        cap = cv2.VideoCapture(self.videoPath)
        if not cap.isOpened():
            print("Error opening file....")
            return

        confidences_list = []
        class_labels_list = []
        frame_list = []  # To store frames for PSNR calculation
        (success, image) = cap.read()
        startTime = 0
        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)
            
            bbox = list(bboxs)
            #converting the bounding boxes into a list
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
            #non maxima 
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)
            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]
                    # Check if the detected object is a vehicle (you may need to customize this based on your classes)
                    
                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                    x, y, w, h = bbox

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                    ####################################################
                    lineWidth= min(int(w*0.3),int(h*0.3))
                    cv2.line(image,(x,y),(x+lineWidth,y),classColor,thickness=5)
                    cv2.line(image,(x,y),(x,y+lineWidth),classColor,thickness=5)
                    cv2.line(image,(x+w,y),(x+w -lineWidth,y),classColor,thickness=5)
                    cv2.line(image,(x+w,y),(x+w,y+lineWidth),classColor,thickness=5)
                    ####################################################
                    cv2.line(image,(x,y+h),(x+lineWidth,y+h),classColor,thickness=5)
                    cv2.line(image,(x,y+h),(x,y+h-lineWidth),classColor,thickness=5)
                    cv2.line(image,(x+w,y+h),(x+w -lineWidth,y+h),classColor,thickness=5)
                    cv2.line(image,(x+w,y+h),(x+w,y+h-lineWidth),classColor,thickness=5)

                    # Append confidence score and class label
                    confidences_list.append(classConfidence)
                    class_labels_list.append(classLabel)

            # Store frames for PSNR calculation
            frame_list.append(image.copy())

            cv2.putText(image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()

        # Calculate and print MSE per class
        mse_per_class = self.compute_mse_per_class(confidences_list, class_labels_list)
        # Calculate and print PSNR per class
        psnr_per_class, non_zero_labels = self.compute_psnr_per_class(frame_list, class_labels_list)
        
        

        # Display the bar graph for MSE
        plt.figure(figsize=(7, 5))
        self.show_bar_graph(mse_per_class, class_labels_list, 'MSE', 'red', min(mse_per_class), (max(mse_per_class))+0.001, width=0.6,label='Mean Squared Error')
        plt.show()

        # Display the bar graph for confidence scores
        self.show_bar_graph(confidences_list, class_labels_list, 'Confidence', 'blue', 0, 1, width=0.6,label='confidence')
        plt.show()

        # Display the bar graph for PSNR
        plt.figure(figsize=(7, 5))
        self.show_bar_graph(psnr_per_class, non_zero_labels, 'PSNR', 'green', 0, max(psnr_per_class)+1, width=0.4, label='Peak Signal-Noise Ratio')
        plt.show()

        cap.release()
        cv2.destroyAllWindows()

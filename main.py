
from Detector import *
import os

def main():

    videoPath="test_videos/street6.mp4"
    #path is automatically infered by the os module
    configPath=os.path.join("model_data","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath=os.path.join("model_data","frozen_inference_graph.pb")
    classesPath=os.path.join("model_data","coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

#calling the function
if __name__=='__main__':
    main()  
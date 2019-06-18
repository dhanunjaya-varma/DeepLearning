import random as rand

angles =  [x*15 for x in range(0, 12)]
colors = ['Red', 'Blue']
lengths = [7, 15]
widths = [1, 3]

from PIL import Image, ImageDraw
import math
import os
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc


def createVideo(outputVideoName):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outputVideo = cv2.VideoWriter(outputVideoName, fourcc, 2, (84, 84))

    for i in range(1, 97):

        images = []
        num = 1
        
        dir_path = 'Class' + str(i)

        for img in os.listdir('Class'+str(i)):
            if num > 90:
                break
            num += 1
            images.append(img)
        
        for i in range(0,10):
            frame = Image.new('RGB', (84,84))
            frame.paste(Image.open(dir_path+"/"+images[i*9]), (0,0,28,28))
            frame.paste(Image.open(dir_path+"/"+images[i*9+1]), (28,0,56,28))
            frame.paste(Image.open(dir_path+"/"+images[i*9+2]), (56,0,84,28))
            frame.paste(Image.open(dir_path+"/"+images[i*9+3]), (0,28,28,56))
            frame.paste(Image.open(dir_path+"/"+images[i*9+4]), (28,28,56,56))
            frame.paste(Image.open(dir_path+"/"+images[i*9+5]), (56,28,84,56))
            frame.paste(Image.open(dir_path+"/"+images[i*9+6]), (0,56,28,84))
            frame.paste(Image.open(dir_path+"/"+images[i*9+7]), (28,56,56,84))
            frame.paste(Image.open(dir_path+"/"+images[i*9+8]), (56,56,84,84))

            frame = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            outputVideo.write(frame)
    
    outputVideo.release()
        

createVideo("video.mp4")
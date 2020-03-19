import random as rand

angles =  [x*15 for x in range(0, 12)]
colors = ['Red', 'Blue']
lengths = [7, 15]
widths = [1, 3]

from PIL import Image, ImageDraw
import math
import os

### FOR TESTING
# im = Image.new('RGB', (28, 28), (0, 0, 0)) 
# draw = ImageDraw.Draw(im) 
# draw.line((0, 0, 28, 0), fill='Red', width=3)
# im.show()


def drawline(theta, length, width, color, count):
    
    for x in range(0, 28):
        for y in range(27, -1, -1):
            
            if count >= 1000:
                break
            
            x1 = x + length*math.cos(math.radians(theta))
            y1 = y - length*math.sin(math.radians(theta))

            if(0 <= x1 <= 28 and 0 <= y1 <= 28):  
                count += 1
                im = Image.new('RGB', (28, 28), (0,0,0))
                draw = ImageDraw.Draw(im)
                draw.line((x1, y1, x, y), fill=color, width=width)

                # FOLLOWING THE NAMING CONVENTION
                image_name = ""
                if length == 7: image_name += "0_"
                else: image_name += "1_"
                if width == 1: image_name += "0_"
                else: image_name += "1_"
                image_name += str(angles.index(theta))+"_"
                if color == "Red": image_name += "0_"
                else: image_name += "1_"
                image_name += str(count)
                
                im.save('./Class'+str(num_classes)+'/'+str(image_name)+'.jpeg')
    return count


num_classes = 0

for theta in angles:
    for length in lengths:
        for width in widths:
            for color in colors:
                
                num_classes += 1

                if not os.path.exists('./Class'+str(num_classes)):
                    os.makedirs('./Class'+str(num_classes))

                    count = 0
                    while (count < 1000):
                        retCount = drawline(theta, length, width, color, count)
                        count = retCount



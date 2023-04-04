
import numpy as np
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if (len(np.shape(image)) == 3 and np.shape(image)[2] == 3):
        return image
    else:
        image = image.convert("RGB")
        return image
    
def preprocess_input(image):
    image /= 255.0
    return image
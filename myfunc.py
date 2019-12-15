import time
import random
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
from csvs import *
from mycv import *
from myfunc import *


def nowtime():
    return time.strftime('%c', time.localtime(time.time()))

def now_imgs(batch_size, train_data, train_y, nums):
    now_x = []
    now_y = []

    for j in range(batch_size):
        if len(nums)==0:
            break
        ni = random.randint(0, len(nums)-1)
        num = nums.pop(ni)
        x = read_png(train_data[num][0]).astype('float32')
        x /= 255
        now_x.append(x)
        now_y.append(train_y[num])
    
    return now_x, now_y
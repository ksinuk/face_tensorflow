import numpy as np 
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import matplotlib.pyplot as plt
from csvs import *
from mycv import *


train_data = read_csv("train_vision.csv")
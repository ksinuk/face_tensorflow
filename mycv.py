import numpy as np 
import cv2 as cv

def read_png(name):
    fn = "./faces_images/"
    tmp = cv.imread(fn+name)
    hsv = cv.cvtColor(tmp, cv.COLOR_BGR2HSV)
    hsv = delete_light(hsv)
    out = np.array(hsv)
    return out

def show_img(data):
    cv.imshow('image',data)
    cv.waitKey(0)
    cv.destroyAllWindows()

def delete_light(img):
    h,s,v = cv.split(img)

    hist, bins = np.histogram(h.flatten(), 256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    h =  cdf[h]

    return cv.merge((h,s,v))
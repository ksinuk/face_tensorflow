import numpy as np 
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import matplotlib.pyplot as plt
from csvs import *
from mycv import *
from myfunc import *

batch_size = 50
nb_classes = 6
time_cut = 100
learning_rate = 0.001
layer_cnt = 3
filters = [32,64,128]
filter_sizes = [8,5,3]
last_filter_cnt = 512
img_size = 128

output_name = "output.csv"
save_model_file = './train_model.ckpt'

#------------------------------------------------------------------------

tf.compat.v1.set_random_seed(191)
keep_prob = tf.compat.v1.placeholder(tf.float32)

#--------------------------------------------------------------------------

train_data = read_csv("train_vision.csv")
train_y = [[0 for j in range(nb_classes)] for i in range(len(train_data))]
for i in range(len(train_data)):
    tmp = ord(train_data[i][1][0])-ord('1')
    train_y[i][tmp] = 1

#-----------------------------------------------------------------------
X = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
Y = tf.placeholder(tf.float32, [None, nb_classes])

Ws = [0 for i in range(layer_cnt)]
Ls = [0 for i in range(layer_cnt)]
for i in range(layer_cnt):
    filter_cnt = filters[i]
    filter_cnt_pre = filters[i-1] if i!=0 else 3
    Ws[i] = tf.Variable(tf.random.normal([filter_sizes[i], filter_sizes[i], filter_cnt_pre, filter_cnt], stddev=0.01))

    if i!=0:
        Ls[i] = tf.nn.conv2d(Ls[i-1], Ws[i], strides=[1, 1, 1, 1], padding='SAME')
    else:
        Ls[i] = tf.nn.conv2d(X, Ws[i], strides=[1, 1, 1, 1], padding='SAME')
    Ls[i] = tf.nn.relu(Ls[i])
    Ls[i] = tf.nn.max_pool2d(Ls[i], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    Ls[i] = tf.nn.dropout(Ls[i], rate = (1 - keep_prob))

nnfilter_cnt = filters[layer_cnt-1]*(img_size//(2**layer_cnt))**2
L_flat = tf.reshape(Ls[layer_cnt-1], [-1, nnfilter_cnt])
nnW = tf.compat.v1.get_variable("nnW", shape=[nnfilter_cnt, last_filter_cnt], initializer=tf.contrib.layers.xavier_initializer())
nnb = tf.Variable(tf.random.normal([last_filter_cnt]))
nnL = tf.nn.relu(tf.matmul(L_flat, nnW) + nnb)
nnL = tf.nn.dropout(nnL, rate = (1 - keep_prob))

lastW = tf.compat.v1.get_variable("lastW", shape=[last_filter_cnt, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
lastb = tf.Variable(tf.random.normal([nb_classes]))
logits = tf.matmul(nnL, lastW) + lastb

#---------------------------------------------------------------------
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
#----------------------------------------------------------------------
print("study start!!")
print("\nKeyboardInterrupt is control + c\n")
try:
    for time in range(time_cut):
        cnt = 0
        cost_sum = 0
        nums = [i for i in range(len(train_data))]

        while len(nums):
            now_x, now_y = now_imgs(batch_size, train_data, train_y, nums)

            feed_dict = {X: now_x, Y: now_y, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

        print("time : {}, {}".format(time, nowtime()))
except KeyboardInterrupt:
    print('interrupted!')

print("study end!!")
#----------------------------------------------------------------------
# saver = tf.compat.v1.train.Saver()
# saver.save(sess, save_model_file)
#-----------------------------------------------------------------------
test_data = read_csv("test_vision.csv")
test_y = [0]*len(test_data)
make_y = tf.argmax(logits, 1)

for i in range(len(test_data)):
    testx = [0]
    testx[0] = read_png(test_data[i][0]).astype('float32')
    testx[0] /= 255
    
    testy = sess.run(make_y, feed_dict = {X: testx, keep_prob: 1})
    test_y[i] = testy[0]+1

write_csv(test_y, output_name)


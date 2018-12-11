"""

Car Door Orientation
@author: Hang Wu
@date: 2018.12.05

"""


import os
import random
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import re

# Eigen
import load_image


def load_img(path):
    img = skimage.io.imread(path)[:, :, : 3]  # for RGBA
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img


def get_car_door_pose_from_filename(file_name):

    # car_door_1_125.png ==>> 'car_door', 1, 125
    match = re.match(r'([A-Za-z_]+)(_+)([0-9]+)(_+)([0-9]+)(\.png)', file_name, re.I)
    class_name = match.groups()[0]
    latitude = match.groups()[2]
    longitude = match.groups()[4]
    # print(class_name, latitude, longitude)

    return latitude, longitude


def load_data():

    imgs_car_door = {'car_door': [], }
    latitude_car_door = []    
    longitude_car_door = []
    counter = 0
    for w in imgs_car_door.keys():
        image_paths = load_image.loadim('/home/hangwu/Workspace/car_door_half')
        for image_path in image_paths:
            resized_img = load_img(image_path)
            image_name = image_path.split(os.path.sep)[-1]
            la_cd, lo_cd = get_car_door_pose_from_filename(image_name)

            imgs_car_door[w].append(resized_img)
            latitude_car_door.append(la_cd)
            longitude_car_door.append(lo_cd)
            if counter % 100 == 0:
                print("loading {:.2f}%".format(counter/len(image_paths)*100))
            counter += 1

    return imgs_car_door['car_door'], latitude_car_door, longitude_car_door


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        # self.tfy = tf.placeholder(tf.float32, [None, 1])
        self.tfy_1 = tf.placeholder(tf.float32, [None, 1])
        self.tfy_2 = tf.placeholder(tf.float32, [None, 1])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        # from now on

        # self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        # self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        # self.out = tf.layers.dense(self.fc6, 1, name='out')

        # for latitude
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6_1 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6_1')
        self.out_1 = tf.layers.dense(self.fc6_1, 1, name='out_1')

        # for longitude
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6_2 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6_2')
        self.out_2 = tf.layers.dense(self.fc6_2, 1, name='out_2')

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            # self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.loss_1 = tf.losses.mean_squared_error(labels=self.tfy_1, predictions=self.out_1)
            self.loss_2 = tf.losses.mean_squared_error(labels=self.tfy_2, predictions=self.out_2)
            self.loss = 0.5*self.loss_1 + 0.5*self.loss_2
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y_1, y_2):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy_1: y_1, self.tfy_2: y_2})
        return loss

    def predict(self, paths):
        fig, axs = plt.subplots(1, 4)
        for i, path in enumerate(paths):
            x = load_img(path)
            latitude = self.sess.run(self.out_1, {self.tfx: x})
            longitude = self.sess.run(self.out_2, {self.tfx: x})
            img_name = path.split(os.path.sep)[-1]
            axs[i].imshow(x[0])
            axs[i].set_title('Latitude: %d \nLongitude: %d \nName: %s' % (latitude, longitude, img_name))
            axs[i].set_xticks(()); axs[i].set_yticks(())
        # plt.ion()
        plt.show()

    def save(self, path='/home/hangwu/Mask_RCNN/logs/door_pose_x_y/car_door_pose'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    # tigers_x, cats_x, tigers_y, cats_y = load_data()
    car_door_x, car_door_y_1, car_door_y_2 = load_data()

    xs = np.concatenate(car_door_x, axis=0) # np.concatenate(tigers_x + cats_x, axis=0)

    # latitude
    ys_1 = np.asarray(car_door_y_1)
    ys_1 = ys_1.reshape(len(ys_1), 1)
    # print('=========================================================')
    # print(ys.shape)
    # print('=========================================================')

    # longitude
    ys_2 = np.asarray(car_door_y_2)
    ys_2 = ys_2.reshape(len(ys_2), 1)


    vgg = Vgg16(vgg16_npy_path='/home/hangwu/Mask_RCNN/transferlearning/for_transfer_learning/vgg16.npy')
    print('Net built')

    for i in range(100000):
        b_idx = np.random.randint(0, len(xs), 6)
        # print('=========================================================')
        # print(xs[b_idx], ys[b_idx])
        # print('=========================================================')
        train_loss = vgg.train(xs[b_idx], ys_1[b_idx], ys_2[b_idx])
        if i%100 == 0:
            print(i, 'train loss: ', train_loss)

    vgg.save('/home/hangwu/Mask_RCNN/logs/door_pose_x_y/car_door_pose')      # save learned fc layers


def eval():
    vgg = Vgg16(vgg16_npy_path='/home/hangwu/Mask_RCNN/transferlearning/for_transfer_learning/vgg16.npy',
                restore_from='/home/hangwu/Mask_RCNN/logs/door_pose_x_y/car_door_pose')

    imgs = {'car_door': [], }
    for k in imgs.keys():
        img_dir = '/home/hangwu/CyMePro/data/car_door'
        for file in os.listdir(img_dir):
            if not file.lower().endswith('.png'):
                continue
            try:
                img_path = os.path.join(img_dir, file)
            except OSError:
                continue
            imgs[k].append(img_path)    # [1, height, width, depth] * n
    # print(imgs['car_door'])
    # print(random.choice(imgs['car_door']))
    image_real = '/home/hangwu/Dropbox/Masterarbeit/Photos/door_pure5.jpg'
    vgg.predict(
        [random.choice(imgs['car_door']), random.choice(imgs['car_door']), 
            random.choice(imgs['car_door']), image_real]
    )


if __name__ == '__main__':
    # image_paths = load_image.loadim('/home/hangwu/CyMePro/data/car_door')
    # for image_path in image_paths:
    #     image_name = image_path.split(os.path.sep)[-1]
    #     img_shape = load_img(image_path)
    #     print(img_shape.shape)
    #     # print(image_name)
    #     get_car_door_pose_from_filename(image_name)

    # train()
    eval()

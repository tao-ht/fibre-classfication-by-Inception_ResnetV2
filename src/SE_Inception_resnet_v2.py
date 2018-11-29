import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import data_deal as dl
import numpy as np
import os
import cv2
# from PIL import Image
# import json

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class_num = 3
image_wight = 299
image_hight = 299
img_channels = 3

classes = ["None", "棉", "亚麻"]

weight_decay = 0.001
momentum = 0.2

init_learning_rate = 0.005# 0.0003

reduction_ratio = 4

batch_size = 20
iteration = 3497#2186
#69930

# val_iteration = 500#313# 20

total_epochs = 15

# 训练datasets图片的存放路径
TRAIN_IMAGE_PATH = "../datasets/train_299_299/"
PREDICT_SAVE_PATH = "../datasets/predict_result.csv"

"""图片数据流的入口之1/2"""
# train_name, train_label, val_name, val_label = dl.prepare_data()#后期不在使用
# val_batch_all = len(val_name)
# val_iteration = (val_batch_all//batch_size) + 1
"""
后期data_deal.py中：pic_name_label_load()、prepare_data()
SE_Inception_resnet_V2.py中 data_label_load 不再使用！！！
假设接受数据流为数组(Image_array)形式一组图片（batchsize, weight, height,3)
伪代码处理流程：
1、分别resize（batchsize, 200, 200,3),等比例pad填充
2、label 均默认置0
3、图片数据定位到 """"图片数据流的入口之2/2""""位置
代码实现：
# 添加到demo开头
val_batch_all = len(Image_array[0])
val_iteration = (val_batch_all//50) + 1

# 添加到Evaluate()函数处
img_data = resize_norl(Image_array)
label = np.zeros((val_batch_all,1),int)
img_label = onehot_encoding(label, classes=3)

val_pre_index = 0
add = 50
if val_pre_index + add < val_batch_all:
    val_batch_x = image_data[val_pre_index: val_pre_index + add]
    val_batch_y = img_label[val_pre_index: val_pre_index + add]
else:
    val_batch_x = image_data[val_pre_index:]
    val_batch_y = img_label[val_pre_index:]
"""

def data_label_load(data_name, data_label):
    X_name, Y_label = data_name, data_label
    data_path = TRAIN_IMAGE_PATH
    pic_label = dl.onehot_encoding(Y_label, class_num)
    pic_data = np.zeros([len(X_name), image_wight, image_hight, 3])
    for img_index in range(len(X_name)):
        img_dir = os.path.join(data_path, X_name[img_index][0])
        img_dir = img_dir.replace('\\', '/')
        img = cv2.imread(img_dir)
        # print(img_dir)
        # img = Image.open(img_dir)
        image_array = np.array(img)#.transpose(1, 0, 2)
        # print('image_array.shape = ', image_array.shape)
        pic_data[img_index, :, :, :] = image_array
    return pic_data, pic_label, X_name

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        if activation:
            network = Relu(network)
        return network

def Fully_connected(x, units=class_num, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Dropout(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Evaluate(sess):
    val_acc = 0.0
    val_loss = 0.0
    val_pre_index = 0
    add = batch_size
    # labels = []

    for it in range(val_iteration):
        """此处图片数据的加载，后期要处理成从数组中获得,,具体处理如上面"""
        if val_pre_index + add < val_batch_all:
            val_batch_name = val_name[val_pre_index: val_pre_index + add]
            val_batch_label = val_label[val_pre_index: val_pre_index + add]
        else:
            val_batch_name = val_name[val_pre_index:]
            val_batch_label = val_label[val_pre_index:]
        """图片数据流的入口之2/2"""
        val_batch_x, val_batch_y, val_batch_name = data_label_load(val_batch_name, val_batch_label)
        """此处图片数据的加载，后期要处理成从数组中获得,,具体处理如上面"""


        val_batch_x = dl.color_preprocessing(val_batch_x)
        # val_batch_x = data_augmentation(val_batch_x)
        val_pre_index = val_pre_index + add
        val_feed_dict = {
            x: val_batch_x,
            label: val_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=val_feed_dict)
        # pre_labels = labels_max_idx.eval(feed_dict=val_feed_dict)
        # labels.extend(pre_labels)
        # writeTojson(val_batch_name, pre_labels_)
        # print("idx =%d, val_loss =%.4f, val_acc =%.4f\n" % (it*add, loss_, acc_))
        val_loss += loss_
        val_acc += acc_

    val_loss /= val_iteration
    val_acc /= val_iteration
    #
    summary = tf.Summary(value=[tf.Summary.Value(tag='val_loss', simple_value=val_loss),
                                tf.Summary.Value(tag='val_accuracy', simple_value=val_acc)])

    return val_acc, val_loss, summary #labels #

class SE_Inception_resnet_v2():
    def __init__(self,x , training):
        self.training = training
        self.model = self.Build_SEnet(x)

    def Stem(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=32, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_conv1')
            x = conv_layer(x, filter=32, kernel=[3,3], padding='VALID', layer_name=scope+'_conv2')
            block_1 = conv_layer(x, filter=64, kernel=[3,3], layer_name=scope+'_conv3')

            split_max_x = Max_pooling(block_1)
            split_conv_x = conv_layer(block_1, filter=96, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')
            x = Concatenation([split_max_x,split_conv_x])

            split_conv_x1 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv3')

            split_conv_x2 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[7,1], layer_name=scope+'_split_conv5')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[1,7], layer_name=scope+'_split_conv6')
            split_conv_x2 = conv_layer(split_conv_x2, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv7')

            x = Concatenation([split_conv_x1,split_conv_x2])

            split_conv_x = conv_layer(x, filter=192, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv8')
            split_max_x = Max_pooling(x)

            x = Concatenation([split_conv_x, split_max_x])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_A(self, x, scope):
        with tf.name_scope(scope) :
            init = x

            split_conv_x1 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=32, kernel=[3,3], layer_name=scope+'_split_conv3')

            split_conv_x3 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = conv_layer(split_conv_x3, filter=48, kernel=[3,3], layer_name=scope+'_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[3,3], layer_name=scope+'_split_conv6')

            x = Concatenation([split_conv_x1,split_conv_x2,split_conv_x3])
            x = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_final_conv1', activation=False)

            x = x*0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_B(self, x, scope):
        with tf.name_scope(scope) :
            init = x

            split_conv_x1 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=128, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=160, kernel=[1,7], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=192, kernel=[7,1], layer_name=scope+'_split_conv4')

            x = Concatenation([split_conv_x1, split_conv_x2])
            x = conv_layer(x, filter=1152, kernel=[1,1], layer_name=scope+'_final_conv1', activation=False)
            # 1154
            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_resnet_C(self, x, scope):
        with tf.name_scope(scope) :
            init = x

            split_conv_x1 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=224, kernel=[1, 3], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=256, kernel=[3, 1], layer_name=scope + '_split_conv4')

            x = Concatenation([split_conv_x1,split_conv_x2])
            x = conv_layer(x, filter=2144, kernel=[1,1], layer_name=scope+'_final_conv2', activation=False)
            # 2048
            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Reduction_A(self, x, scope):
        with tf.name_scope(scope) :
            k = 256
            l = 256
            m = 384
            n = 384

            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=n, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=k, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=l, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=m, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Reduction_B(self, x, scope):
        with tf.name_scope(scope):
            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope+'_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=384, kernel=[3, 3], stride=2, padding='VALID', layer_name=scope+'_split_conv2')

            split_conv_x2 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3, 3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

            split_conv_x3 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope+'_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=288, kernel=[3, 3], layer_name=scope+'_split_conv6')
            split_conv_x3 = conv_layer(split_conv_x3, filter=320, kernel=[3, 3], stride=2, padding='VALID', layer_name=scope+'_split_conv7')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):


            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

            return scale

    def Build_SEnet(self, input_x):
        input_x = tf.pad(input_x, [[0, 0], [0, 0], [0, 0], [0, 0]])

        x = self.Stem(input_x, scope='stem')

        for i in range(5):
            x = self.Inception_resnet_A(x, scope='Inception_A'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A'+str(i))

        x = self.Reduction_A(x, scope='Reduction_A')
   
        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A')

        for i in range(10):
            x = self.Inception_resnet_B(x, scope='Inception_B'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B'+str(i))

        x = self.Reduction_B(x, scope='Reduction_B')
        
        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B')

        for i in range(5):
            x = self.Inception_resnet_C(x, scope='Inception_C'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C'+str(i))
         
            
        # channel = int(np.shape(x)[-1])
        # x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C')
        
        x = Global_Average_Pooling(x)
        #rate = 0.2 ->0.3,增加0.1的dropout
        x = Dropout(x, rate=0.3, training=self.training)
        x = flatten(x)

        x = Fully_connected(x, layer_name='final_fully_connected')
        return x


# train_name, train_label, val_name, val_label = prepare_data()

train_name, train_label, val_name, val_label = dl.prepare_data()#后期不在使用
val_batch_all = len(val_name)
val_iteration = (val_batch_all//batch_size) + 1
# image_size = 200*200, img_channels = 3, class_num = 3
x = tf.placeholder(tf.float32, shape=[None, image_wight, image_hight, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_Inception_resnet_v2(x, training=training_flag).model
labels_max_idx = tf.argmax(logits, axis=1, name='labels_max_idx')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))# tf.nn.softmax_cross_entropy_with_logits_v2
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, use_locking=False, name='GradientDescent')
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())


with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('../model_299_3/')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('../logs', sess.graph)

    epoch_learning_rate = init_learning_rate
    
    for epoch in range(0, total_epochs):
        if epoch % 4 == 0:
            epoch_learning_rate = epoch_learning_rate / 2

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0

        for step in range(1, iteration + 1):
            # 这里batch循环读取
            if pre_index + batch_size < 69930:
                batch_train_name = train_name[pre_index: pre_index + batch_size]

                batch_train_label = train_label[pre_index: pre_index + batch_size]
            else:
                batch_train_name = train_name[pre_index:]
                batch_train_label = train_label[pre_index:]

            batch_x, batch_y, _ = data_label_load(batch_train_name, batch_train_label)
            batch_x = dl.color_preprocessing(batch_x)

            # batch_x = data_augmentation(batch_x)
            # for i in batch_x:
            #     cv2.imshow("111", i)
            #     cv2.waitKey(800)

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)
            if pre_index % 32 == 0:
                print("index =%d ,train_loss =%.4f, train_acc =%.4f" % (pre_index, batch_loss, batch_acc))

            train_loss += batch_loss
            train_acc += batch_acc
            pre_index += batch_size

            if pre_index % 10000 == 0:
                print('--------------------save--------------------------')
                saver.save(sess=sess, save_path='../model_299_3/model.ckpt')

            # if pre_index % 100 == 0:# 6000
            #     print('**************************************************')
            #     val_acc, val_loss, val_summary = Evaluate(sess)
            #     print('val_loss =%.4f, val_acc=%.4f\n' % (val_loss, val_acc))

        train_loss /= iteration
        train_acc /= iteration

        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
        
        val_acc, val_loss, val_summary = Evaluate(sess)
        # print('val_loss =%.4f, val_acc=%.4f \n' % (val_loss, val_acc))
        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=val_summary, global_step=epoch)
        summary_writer.flush()

        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, val_loss: %.4f, val_acc: %.4f " % (
            epoch, total_epochs, train_loss, train_acc, val_loss, val_acc)
        print(line)
        with open('logs.txt', 'a') as f:
            f.write(line)

        saver.save(sess=sess, save_path='../model_299_3/model.ckpt')
        # print('*** --- Over ! --- ***')
        # break


# if __name__ == "__main__":
#     """模型调用，使用Evaluate得到预测结果示例"""
#     with tf.Session() as sess:
#         ckpt = tf.train.get_checkpoint_state('../model_200/')
#         if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#             saver.restore(sess, ckpt.model_checkpoint_path)
#         else:
#             sess.run(tf.global_variables_initializer())
#
#         epoch_learning_rate = init_learning_rate
#         predict = Evaluate(sess)
#         print(predict)


    # num = 0
    # step = 0
    #
    # val_name, val_label = prepare_data()
    # val_num = len(val_name)
    # iteration = (val_num // 30) + 1
    #
    # for step in range(iteration):
    #     if step*30 <= iteration:
    #         val_x, pic_label = data_label_load(val_name[step*30:step*30+30], val_label[step*30:step*30+30])
    #     else:
    #         val_x, pic_label = data_label_load(val_name[step * 30:], val_label[step * 30:])
    #     val_batch_all = len(val_x)
    #     val_iteration = (val_batch_all // 50) + 1
    #
    #     """模型调用，使用Evaluate得到预测结果示例"""
    #     with tf.Session() as sess:
    #         ckpt = tf.train.get_checkpoint_state('../model_val/')
    #         if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    #             saver.restore(sess, ckpt.model_checkpoint_path)
    #         else:
    #             sess.run(tf.global_variables_initializer())
    #         predict = Evaluate(sess, val_x)
    #
    #         for i in range(len(predict)):
    #             if predict[i] == pic_label[i]:
    #                 num += 1
    #             else:
    #                 print(predict[i], pic_label[i])
    #         print(num, step*30)
    # print("val_num = ", val_num)
    # print("pre_acc = %.4f" % (num // val_num))
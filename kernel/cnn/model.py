# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from kernel.cnn import helper
from util import doubanutil
from config import filepath

CAPTCHA_HEIGHT = 40
CAPTCHA_WIDTH = 250

# None代表不限条数的输入，紧跟其后的参数代表向量维数
X = tf.placeholder(tf.float32, [None, CAPTCHA_HEIGHT * CAPTCHA_WIDTH])
Y = tf.placeholder(tf.float32, [None, helper.MAX_LENGTH * helper.ALPHA_LENGTH])
keep_prob = tf.placeholder(tf.float32)


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, CAPTCHA_HEIGHT * CAPTCHA_WIDTH])
    batch_y = np.zeros([batch_size, helper.MAX_LENGTH * helper.ALPHA_LENGTH])

    for i in range(batch_size):
        image, word = helper.get_captcha_word_and_image()
        if image == "" or word == "404":
            doubanutil.logger.warning("image empty or word 404")
            break
        image = helper.compute_gray(filepath.image_path + image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = helper.word2vec(word)

    return batch_x, batch_y


def cnn_model(w_alpha=0.01, b_alpha=0.1):
    # 定义CNN模型

    # 输入层
    input_layer = tf.reshape(X, shape=[-1, CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 1])

    # 卷积层
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_layer, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 全连接层
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 32 * 40, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, helper.MAX_LENGTH * helper.ALPHA_LENGTH]))
    b_out = tf.Variable(b_alpha * tf.random_normal([helper.MAX_LENGTH * helper.ALPHA_LENGTH]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def train_model():
    output = cnn_model()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, helper.MAX_LENGTH, helper.ALPHA_LENGTH])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, helper.MAX_LENGTH, helper.ALPHA_LENGTH]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            doubanutil.logger.info("step = " + str(step) + ";loss_ = " + str(loss_))

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                doubanutil.logger.info("step = " + str(step) + ";acc = " + str(acc))

                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.8:
                    saver.save(sess, "./crack_capcha.model", global_step=step)
                    break

            step += 1


if __name__ == "__main__":
    doubanutil.init_func()
    train_model()

"""
# Create a dataset tensor from the images and the labels
dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels))
# Automatically refill the data queue when empty
dataset = dataset.repeat()
# Create batches of data
dataset = dataset.batch(batch_size)
# Prefetch data for faster consumption
dataset = dataset.prefetch(batch_size)

# Create an iterator over the dataset
iterator = dataset.make_initializable_iterator()
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# IMPORTS
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MODE = 'TRAIN'

MNIST = input_data.read_data_sets("/data/mnist/", one_hot=True)
save_path = "../Saving/model.ckpt"

# Training Parameters
learning_rate = 0.001
batch_size = 128
epochs = 200

# HyperParameters
inputs = 784
classes = 10
dropout = 0.75

# CARICO I DATI DI TRAINING E VALUTAZIONE
# mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# CARICO I DATI DI TRAINING E VALUTAZIONE
# train_data = mnist.train.images
# train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

dataset = tf.data.Dataset.from_tensor_slices(
    (MNIST.train.images, MNIST.train.labels))
# Automatically refill the data queue when empty
dataset = dataset.repeat()
# Create batches of data
dataset = dataset.batch(batch_size)
# Prefetch data for faster consumption
dataset = dataset.prefetch(batch_size)

# Create an iterator over the dataset
iterator = dataset.make_initializable_iterator()


def cnn_model_fn(x, MODE):
    if MODE == 'TRAIN':
        mode = tf.estimator.ModeKeys.TRAIN
    elif MODE == 'TEST':
        mode = tf.estimator.ModeKeys.EVAL

    # INPUT LAYER
    input_layer = tf.reshape(x, [-1, 28, 28, 1])

    # CONVOLUTIONAL LAYER #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
    )

    # APPLICO LA FUNZIONE RELU
    conv1_relu = tf.nn.relu(conv1)

    # POOLING LAYER #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1_relu,
        pool_size=[2, 2],
        strides=2
    )

    # CONVOLUTIONAL LAYER #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
    )

    # APPLICO LA FUNZIONE RELU
    conv2_relu = tf.nn.relu(conv2)

    # POOLING LAYER #2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2_relu,
        pool_size=[2, 2],
        strides=2
    )

    # RIDIMENSIONO POOL2 PER RIDURRE IL CARICO COMPUTAZIONALE
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # DENSE LAYER
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
    )

    # APPLICO LA FUNZIONE RELU
    dense_relu = tf.nn.relu(dense)

    # AGGIUNGO L'OPERAZIONE DI DROPOUT
    dropout = tf.layers.dropout(
        inputs=dense_relu,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # LOGIT LAYER
    logits = tf.layers.dense(
        inputs=dropout,
        units=10
    )

    return logits


X, Y = iterator.get_next()

logits = cnn_model_fn(X, MODE)
prediction = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

correct_predict = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

init = tf.global_variables_initializer()
best_acc = 0

with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(init)
    saver = tf.train.Saver()

    if MODE == 'TRAIN':

        print("TRAINING MODE")

        for step in range(1, epochs + 1):
            check = '[ ]'

            sess.run(train_op)
            los, acc = sess.run([loss, accuracy])
            # if step % 10 == 0 or step == 1:
            #     print(step)
            # if acc > 0.95:
            #     if acc>best_acc:
            #         best_acc=acc
            #         print("Best Accuracy:"+str(acc))
            #         saver.save(sess,save_path)

            if step % 20 == 0 or acc >= best_acc:
                if acc >= best_acc:
                    check = '[X]'
                    best_acc = acc
                    print(str(step) + '\t' + '%.4f' % acc + '\t\t' + check)
                else:
                    print(str(step) + '\t' + '%.4f' % acc + '\t\t' + check)

    elif MODE == 'TEST':

        print("TESTING MODE")
        saver.restore(sess, save_path)
        print("Initialization Complete")

        # test the model
        print("Testing Accuracy:" + str(sess.run(accuracy)))
        print("Testing finished")

    else:
        os.system('clear')
        print('\nLa modalità inserita non è corretta.\n')

sess.close()

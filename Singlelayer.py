import tensorflow as tf
import numpy as np

# Dataset
x_data = np.array([
[0.,0.], [0.,1.], [1.,0.], [1.,1.]
])
y_data = np.array([
[0.], [1.], [1.], [0.]
])

# Hyperparamters
n_input = 2
n_hidden = 100
n_output = 1
lr = 0.1
epochs = 10000
display_step = 1000

# Placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Weights
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))#Random uniforme tra -1 e 1
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

# Bias
b1 = tf.Variable(tf.zeros([n_hidden]))
b2 = tf.Variable(tf.zeros([n_output]))

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)
cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y) * tf.log(1-hy))
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #file = open("./testfile.txt", "w")
    #file.write("STAMPO I PESI\n")
    for step in range(epochs):
        _, c = sess.run([optimizer, cost], feed_dict = {X: x_data, Y: y_data})


        if step % display_step == 0:
            print("Cost: ", c)
            #file.write("\n----------------------------\n")
            #file.write(str(W1.eval(session=sess)))
            #file.write("\n")
            #file.write(str(W2.eval(session=sess)))
            #file.write("\n----------------------------\n")

    #file.write("FINE PESI")

    answer = tf.equal(tf.floor(hy + 0.1), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))
    #file.close()
    print(sess.run([hy], feed_dict = {X: x_data, Y: y_data}))
    print("Accuracy: ", accuracy.eval({X: x_data, Y: y_data}))
    sess.close()
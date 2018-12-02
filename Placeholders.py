import tensorflow as tf

x1 = tf.placeholder(tf.int32)
x2 = tf.placeholder(tf.int32)

product = x1 * x2


sess = tf.Session()

d = {x1: 20, x2: 3}

result = sess.run(product,feed_dict= d)

print(result)


import tensorflow as tf

x1 = tf.placeholder(tf.int32,name="x1")
x2 = tf.placeholder(tf.int32,name="x2")

sum = tf.add(x1,x2,name="addition")


with tf.Session() as session:
    d = {x1: 20, x2: 3}

    result = session.run(sum,feed_dict= d)
    writer=tf.summary.FileWriter("C:\\Users\\Giacomo\\Documents\\Tensorboard",session.graph)
    print(result)
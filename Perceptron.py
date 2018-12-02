import tensorflow as tf

inp = tf.placeholder(tf.float32)
weights = tf.get_variable(name="weights",dtype=tf.float32, initializer=[[3.],[2.],[-1.]])
bias =tf.placeholder(tf.float32)

sess=tf.Session()

sess.run(weights.initializer) #Inizializzo la variabile v1
dot = tf.tensordot(inp, weights,axes=2)
#print(sess.run(dot))
sum=tf.add(dot,bias)


#sigpresigass=presig.assign(sum)



d = {inp: [[1.],[2.],[3.]],  bias: 5.}
#print(sess.run(rel,feed_dict=d))
result= sess.run(sum,feed_dict=d)

if(result>=0):
    print(str(1))
else:
    print(str(-1))


sess.close()

#print(M.eval(session=sess))
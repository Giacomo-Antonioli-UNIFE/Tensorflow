import tensorflow as tf

v1=tf.get_variable(name="v1", initializer=1)

sess=tf.Session()

sess.run(v1.initializer) #Inizializzo la variabile v1
print(v1.eval(session=sess))
assignment= v1.assign_add(3)
sess.run(assignment)
print(v1.eval(session=sess))

M=tf.get_variable(name="MATRIX", initializer=[[1,2,3],[4,5,6],[7,8,9]])

G=tf.get_variable(name="PERMUTATION", initializer= [[0,0,1],[0,1,0],[1,0,0]] )

permutation= tf.matmul(M,G)

matrixoverwrite= M.assign(permutation)

tf.global_variables_initializer().run(session=sess)



print(M.eval(session=sess))
print(G.eval(session=sess))
sess.run(matrixoverwrite)
print(M.eval(session=sess))


writer=tf.summary.FileWriter("C:\\Users\\Giacomo\\Documents\\Tensorboard")
writer.add_graph(sess.graph)
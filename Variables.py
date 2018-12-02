import tensorflow as tf

v1=tf.get_variable(name="v1", shape=[3,3,1])
#shape definisce la forma vettoriale, in questo caso una matrice 3x3x1
v2=tf.get_variable(name="v2", initializer=[[1,2,3],[4,5,6],[7,8,9]])

sess=tf.Session()

sess.run(v1.initializer) #Inizializzo la variabile v1

tf.global_variables_initializer().run(session=sess)
#Inizializzo tutte le variabili in una volta sola.

print(v2.eval(session=sess))
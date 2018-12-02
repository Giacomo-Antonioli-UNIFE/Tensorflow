import tensorflow as tf

x1=tf.constant(1)
x2=tf.constant(2)
#Definiamo un operazione tra costanti
y=x1+x2
#Definiamo la sessione
sess=tf.Session()
#Eseguiamo l'operazione nella nostra sessione
result=sess.run(y)
print(result)

print(x2.eval(session=sess))
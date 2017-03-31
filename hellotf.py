import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) #float32 implicitly
node3 = tf.add(node1, node2)

sess = tf.Session()
sess.run(node3)

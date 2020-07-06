import tensorflow as tf

with tf.Session() as sess:
    # Phase 1: constructing the graph
    a = tf.constant(15, name = "a")
    b = tf.constant(5, name = "b")
    prod = tf.multiply(a,b, name ="Multiply")
    sum = tf.add(a,b, name = "Add")
    res = tf.divide(prod, sum, name= "Divide")

    #phase 2: running the sess
    out = sess.run(res)
    print(out)

    

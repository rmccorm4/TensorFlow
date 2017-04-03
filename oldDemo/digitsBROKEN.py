#### This is demo code gotten from Siraj Raval ##########
#### This is NOT my work ############

import input_data
#don't know what one_hot is supposed to be
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

import tensorflow as tf

#Set parameters
learning_rate = 0.01 #known good learning_rate for this set
training_iteration = 30
batch_size = 100
display_step = 2

#Input for TF graph
x = tf.placeholder("float", [None, 784]) #784 entries in dataset
y = tf.placeholder("float", [None, 10]) #10 possible outputs 0-9

######Create a Model#######

#Set model weights
weight = tf.Variable(tf.zeros([784, 10])) #probability of paths
bias = tf.Variable(tf.zeros([10])) #shifts regression line to fit data

with tf.name_scope("Wx_b") as scope:
	#construct linear model
	model = tf.nn.softmax(tf.matmul(x, weight) + bias) #soft max

#Add summary ops to collect data
weight_histogram = tf.summary.histogram("weights", weight)
bias_histogram = tf.summary.histogram("biases", bias)

# More name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
	#minimize error using cros entropy
	cost_function = -tf.reduce_sum(y * tf.log(model))
	#create a summary to monitor the cost function
	tf.summary.scalar("cost_function", cost_function)

#makes our model improve during training
with tf.name_scope("train") as scope:
	#Gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

#initializing variables
init = tf.global_variables_initializer()

#merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()


#Launch the graph
with tf.Session() as sess:
	sess.run(init)

	# Set the logs writer to the folder /tmp/tensorflow_logs
	summary_writer = tf.summary.FileWriter("/tmp/tensorflow_logs", sess.graph)

	#Training cycle
	for iteration in range(training_iteration):
		avg_cost = 0.0
		total_batch = int(mnist.train.num_examples(batch_size))
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# Fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys}) 
			# Compute average loss
			avg_cost += sess.run(cost_function, feed_dict={x:batch_xs, y:batch_ys})/total_batch
			#Write logs for each iteration
			summary_str = sess.run(merged_summary_op, feeddict={x:batch_xs, y:batch_ys})
			summary_writer.add_summary(summary_str, itration * total_batch + i)
		#Display logs per iteration step
		if iteration % display_step == 0:
			print("Iteration:", "%04d" % (iteration + i), "cost=", "{:.9f}".format(avg_cost))

	print("Tuning completed!")

	# Test the model
	predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
	print("Accuracy: ", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

import tensorflow as tf
class BaseModel():
    def __init__(self):
        self.weight_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.25, dtype=tf.float32)
        self.bias_initializer = tf.zeros_initializer()

    def weights_bias(self):

        """Intialize weights and biases for network"""

        with tf.name_scope("weights"):
            self.w1=tf.get_variable(name="w1",shape=[29,20],dtype=tf.float32,initializer=self.weight_initializer)
            self.w2=tf.get_variable(name="w2",shape=[20,8], dtype=tf.float32,initializer=self.weight_initializer)
            self.w3=tf.get_variable(name="w3",shape=[8,20],dtype=tf.float32, initializer=self.weight_initializer)
            self.w4=tf.get_variable(name="w4",shape=[20,29], dtype=tf.float32,initializer=self.weight_initializer)
        with tf.name_scope("biases"):
            self.b1 = tf.get_variable(name="b1",shape=[20],dtype=tf.float32,initializer=self.bias_initializer)
            self.b2 = tf.get_variable(name="b2",shape=[8], dtype=tf.float32,initializer=self.bias_initializer)
            self.b3 = tf.get_variable(name="b3",shape=[20], dtype=tf.float32, initializer=self.bias_initializer)
    
    def forward_pass(self, inputs):
        """ Forward propagation , compute the output from neural network.

        @param inputs : (?,29) features from dataset as an input to Auto-encoder '?' defines number of batches
        @returns output : mse produced during training over the batches 

        """
        with tf.name_scope("forward_pass"):
            z1=tf.matmul(inputs,self.w1,name="layer_1")+self.b1
            a1=tf.nn.sigmoid(z1)
            z2=tf.matmul(a1, self.w2, name="layer_2")+self.b2
            a2=tf.nn.sigmoid(z2)
            z3=tf.matmul(a2,self.w3, name="layer_")+self.b3
            a3=tf.nn.sigmoid(z3)
            output=tf.matmul(a3,self.w4, name="final_layer")

        return output

    def forward_pass_prod(self, inputs):
        """ Forward propagation , compute the output from neural network.

        @param inputs : (?,29) features from dataset as an input to Auto-encoder '?' defines number of batches
        @returns output : mse produced during training over the batches 

        """
        f1 = lambda: tf.constant("Fraud Transaction",dtype=tf.string)
        f2 = lambda: tf.constant("Normal Transaction",dtype=tf.string)
        with tf.name_scope("forward_pass"):
            z1=tf.matmul(inputs,self.w1,name="layer_1")+self.b1
            a1=tf.nn.sigmoid(z1)
            z2=tf.matmul(a1, self.w2, name="layer_2")+self.b2
            a2=tf.nn.sigmoid(z2)
            z3=tf.matmul(a2,self.w3, name="layer_")+self.b3
            a3=tf.nn.sigmoid(z3)
            output=tf.matmul(a3,self.w4, name="final_layer")
            mse=tf.reduce_mean(tf.square(output - inputs))
            y=tf.constant(0.0001)
            results = tf.case([(tf.math.greater(mse, y), f1)], default=f2)

        return results

            

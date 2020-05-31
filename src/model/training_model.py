from model.base_model import BaseModel
import tensorflow as tf

class Anamoly_Detector(BaseModel):
    def __init__(self,FLAGS):
        super().__init__()
        self.FLAGS = FLAGS
        self.weights_bias()
    
    def compute_loss(self,x_train,x_test):
        """ 
        Compute  mse on batch train and test
        @params x_train : train set batch
        @params x_test  : test set batch

        @returns mse_train & mse_test : the value of reconstruction error between train and test set 
        """
        with tf.name_scope("loss function"):
            mse_train=tf.reduce_mean(tf.square(x_train-self.forward_pass(x_train)))
            mse_test=tf.reduce_mean(tf.square(x_test)-self.forward_pass(x_test))

        return mse_train, mse_test

    def train_network(self, mse_train):
        with tf.name_scope("optimizer_init"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate,name="adams_optimizer").minimize(mse_train)
            
        



    
        

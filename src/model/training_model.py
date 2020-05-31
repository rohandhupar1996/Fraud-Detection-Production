from model.base_model import BaseModel
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class Anamoly_Detector(BaseModel):
    def __init__(self):
        super().__init__()
        self.weights_bias()
    
    def compute_loss(self,predictions,x_train):
        """ 
        Compute  mse on batch train and test
        @params predictions : set of prediction values from NN
        @params x_train : set of train batch on which prediction is done

        @returns mse_train : the value of reconstruction error on train batches
        """
        with tf.name_scope("loss_function"):
            mse_train=tf.reduce_mean(tf.square(predictions - x_train))

        return mse_train

    def train_network(self, mse_train):
        with tf.name_scope("optimizer_init"):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001,name="adams_optimizer").minimize(mse_train)
        return optimizer
        



    
        

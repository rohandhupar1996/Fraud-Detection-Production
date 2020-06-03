from model.base_model import BaseModel
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class InferenceModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.weights_bias()
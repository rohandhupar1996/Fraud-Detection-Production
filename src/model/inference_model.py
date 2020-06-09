from model.base_model import BaseModel
import tensorflow as tf
class InferenceModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.weights_bias()
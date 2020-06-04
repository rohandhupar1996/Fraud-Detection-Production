import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os 
from model.inference_model import InferenceModel
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np
from data.preprocess import get_data

checkpoint_dir = "checkpoints/"
model_export_dir="model-export/"
tf_record_test_dir="data/preprocessed/tf_record_test/"
model_name="anamoly_detection/"
model_version="1/"

def main(_):

    # data_dir="" # system input
    # data=get_data(data_dir=data_dir,data_type="test")
    inference_graph=tf.Graph()
    with inference_graph.as_default():
    # inference model 
        model=InferenceModel()
        input_data=tf.placeholder(tf.float32, shape=[1, 29])
        output=model.forward_pass_prod(input_data)
        saver = tf.train.Saver()

        with tf.Session(graph=inference_graph) as sess:
        
        # Restore the learned weights and biases of neural network from the last checkpoint
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)   
            saver.restore(sess, ckpt.model_checkpoint_path)      
        
            # Build the export path. path/to/model/MODELNAME/VERSION/
            export_path = os.path.join(tf.compat.as_bytes(model_export_dir),
                                    tf.compat.as_bytes(model_name),
                                    tf.compat.as_bytes(model_version)
                                    )
            print('Exporting trained model to %s'%export_path)
        
            # Instance of the build for the SavedModel format
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        
        
            # Create tensor info for datainput and output of the neural network
            predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(input_data)
            
            predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(output)

                # Build prediction signature
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'inputs': predict_tensor_inputs_info},
                    outputs={'class name': predict_tensor_scores_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )
            
                
            # Export the model in a SavedModel format
            builder.add_meta_graph_and_variables(
                sess, # session
                [tf.saved_model.tag_constants.SERVING], #tags for the metagraph
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature
                })

            builder.save()
            
if __name__ == "__main__":
    tf.app.run()


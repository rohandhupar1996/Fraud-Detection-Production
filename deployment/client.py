from argparse import ArgumentParser
import re
from preprocess import get_data


# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf

# TensorFlow serving stuff to send messages

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto


def parse_args():
    '''
    Parse input arguments from the terminal.
    
    @--server: The IP adress of the destination server and the port, must be in the format IP:PORT
               Default value is the local host with port 8500
    
    '''
    parser = ArgumentParser(description='Request a TensorFlow server for a prediction on the image')
    parser.add_argument('-s', '--server',
                        dest='server',
                        default='af332fe93f15a4e948333897feaace27-988552581.ap-south-1.elb.amazonaws.com:8501',
                        help='prediction service host:port')
    
    args = parser.parse_args()

    host, port = args.server.split(':')
    
    return host, port


def main():
    # Parse command line arguments
    host, port= parse_args()

    # Create a grpc channel using the IP adress and the port
    channel = implementations.insecure_channel(host, int(port))
    # Create a stub
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Name of the data that should be send to the server

    file='../data/raw/creditcard_dataset_fraud.csv'
    data=get_data(data_type="test",data_dir=file)

    sample_df=data.sample(20)

    for _,row in sample_df.iterrows():

        input_data=[[s for s in row]]
        # print(input_data)

        # Create a request object
        request = predict_pb2.PredictRequest()
        
        # Name of the model running on the tensorflow_model_server (either locally or in Docker container)
        request.model_spec.name = 'anamoly_detection'
        # Name of the defined prediction signature in the SavedModelInstance on the server (either locally or in Docker container)
        request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        
        # Make a request (time-out after 20 seconds)
        request.inputs['inputs'].CopyFrom(make_tensor_proto(input_data, shape=[1,29]))
	
        result = stub.Predict(request, 20.0)  # 60 secs timeout
        print(result)


if __name__ == '__main__':
    main()

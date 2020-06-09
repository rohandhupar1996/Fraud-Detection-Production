# Fraud-Detection-Production
It is basically extension to my previous work where I used tensorflow pipelines (etl) to enhance the efficiency
of cpu-gpu parallel processing 

This repository contains prod-code where we used autoencoder to detect frauds in transaction made on Tensorflow v1.13.1 framework
with tensorflow serving api v1.13.1

structure for the files and code :

#### checkpoints -> model checkpoints while training 
#### data -> both processed and raw
#### deployment -> tensorflow serving api 
#### model-export -> saved model 
#### notebook-> Data exploration and working protoype (training the model with tf pipeline)
#### src-> source code
#### summary-> model summary while training and validation on test set 

#### dependcies : tensorflow v1.13.1

how to run the api 
#### 1> old school way 
#### install tensorflow server model and port to 8500 for grpc request and give model_name and model_path and install 
tensorflow_model_server --port=8500 --model_name=anamoly_detection --model_base_path=$HOME/Desktop/Fraud-Detection-Production-master/model-export/anamoly_detection/
#### tensorflow serving api v1.13.1 with that 
#### run -> python client.py
   
2nd way docker containerization approach though I deployed whole model in aws but cost way up high you can use this way which very easy 
#### 1> download docker image for tensorflow serving api of google which has all dependecies 
#### 2> create it's container 
docker create -p 8500:8500 -e MODEL_NAME=anamoly_detection --mount type=bind , source=$HOME/Desktop/Fraud-Detection-Producton/model-export/anamoly_detection,target=/models/anamoly_detection --name=my_container1 tensorflow/serving
#### 3> start the container once container is started you don't need to run sever again and again
docker start my_container1
#### 4> run python client.py

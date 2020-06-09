# Fraud-Detection-Production
It is basically extension to my previous work where I used tensorflow pipelines (etl) to enhance the efficiency
of cpu-gpu parallel processing

This repository contains prod-code where we used autoencoder to detect frauds in transaction made on TF v1.13 framework
with tensorflow serving api

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
#### tensorflow serving api v1.13.1 with that 
#### run -> python client.py
   
2nd way docker containerization approach though I deployed whole model in aws but cost way up high you can use this way which very easy 
#### 1> download docker image for tensorflow serving api of google which has all dependecies 
#### 2> create it's container 
#### 3> start the container once container is started you don't need to run sever again and again
#### 4> run python client.py

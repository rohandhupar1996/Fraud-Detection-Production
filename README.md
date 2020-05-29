# Fraud-Detection-Production
It is basically extension to my previous work where I used tensorflow pipelines (etl) to enhance the efficiency
of cpu-gpu parallel processing

This repository contains prod-code where we used autoencoder to detect frauds in transaction made on TF v1.13 framework
with tensorflow serving api

structure for the files and code :

checkpoints -> model checkpoints while training 
data -> both processed and raw
deployment -> tensorflow serving api 
model-export -> saved model 
notebook-> Data exploration and working protoype (training the model with tf pipeline)
src-> source code
summary-> model summary while training and validation on test set 

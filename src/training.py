import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
from data.dataset import get_training_data,get_test_data,get_validation_set_1,get_validation_set_2
from model.training_model import Anamoly_Detector
import os 
import sys


# Number of training samples
n_training_samples=250000
# Batch size
batch_size=64
# Learning rate 
learning_rate=0.001
# Number of epochs
num_epoch=50
# number of batches
n_batches=int(n_training_samples/batch_size)
# Evaluate model after number of steps
eval_after=1000
# path for training test and validation set 

tf_record_dir="data/preprocessed/tf_records_dir/"
train_path=os.path.join(tf_record_dir,"tf_records_train")
test_path=os.path.join(tf_record_dir,"tf_records_test")
validation_path_set_1=os.path.join(tf_record_dir,"tf_records_validation_set_1")
validation_path_set_2=os.path.join(tf_record_dir,"tf_records_validation_set_2")



# Data Pipeline initiated 
def main(_):

    training_graph=tf.Graph()

    with training_graph.as_default():
        
        writer = tf.summary.FileWriter("summary/train/1")

        with tf.name_scope("Data_input_pipeline"):
            # Access the tf.dataset instance of the tf.data API for the training, 
            # testing and validation of the model
            training_dataset=get_training_data(train_path)
            test_dataset=get_test_data(test_path)
            validation_dataset_1=get_validation_set_1(validation_path_set_1)
            validation_dataset_2=get_validation_set_2(validation_path_set_2)

            # build an interator for each dataset to access the elements of the dataset
            iterator_train = training_dataset.make_initializable_iterator()
            iterator_test = test_dataset.make_initializable_iterator()
            iterator_val_1 = validation_dataset_1.make_initializable_iterator()
            iterator_val_2 = validation_dataset_2.make_initializable_iterator()

            # get the features (x) and labels (y) from the dataset
            x_train, y_train = iterator_train.get_next()
            x_test, y_test = iterator_test.get_next()
            x_val_1, y_val_1 = iterator_val_1.get_next()
            x_val_2, y_val_2 = iterator_val_2.get_next()


            x_train_copy=tf.identity(x_train,name=None) 
            y_train_copy=tf.identity(y_train,name=None)

        model=Anamoly_Detector()
        output = model.forward_pass(x_train_copy)
        mse_train=model.compute_loss(output,x_train_copy)
        update_op=model.train_network(mse_train)
        mse_test=model.compute_loss(model.forward_pass(x_test),x_test)

        predictions_val_1=model.forward_pass(x_val_1)
        predictions_val_2=model.forward_pass(x_val_2)

        saver=tf.train.Saver(max_to_keep=5)

    with tf.Session(graph=training_graph) as sess:

        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        print('\n\n\nBegin training...')
        for epoch in range(0, num_epoch):  

            sess.run(iterator_train.initializer)
            sess.run(iterator_test.initializer)

            temp_loss=0

            for iter_nr in range(n_batches-1): 

                _, l=sess.run((update_op, mse_train))
                temp_loss+=l

                sys.stdout.write('\r>> Training on batch_nr: %i/%i' % (iter_nr+1, n_batches))
                sys.stdout.flush()

                if iter_nr>1 and iter_nr%eval_after==0:
                    val_loss=sess.run(mse_test)
                    print('\n>> epoch_nr: %i, training_loss: %.5f , validation_loss: %.5f' %(epoch+1, 
                            (temp_loss/eval_after),val_loss))

                    temp_loss=0

        print('\nSaving Checkpoints...')
        # Save a checkpoint after the training
        saver.save(sess, FLAGS.checkpoints_path)

        print('\n\nResult of the evaluation on the test set: \n')

        # Test the model after the training is complete with the test dataset
        sess.run(iterator_val_1.initializer)
        sess.run(iterator_val_2.initializer)

        predictions_val_1,x_val_1,y_val_1=sess.run((predictions_val_1,x_val_1,y_val_1))
        predictions_val_2,x_val_2,y_val_2=sess.run((predictions_val_2,x_val_2,y_val_2))

        mse = np.mean(np.power(x_val_1 - predictions_val_1, 2), axis=1)
        error_df_test = pd.DataFrame({'reconstruction_error': mse,
                                'true_class': y_val_1.ravel()})
        
        error_df_test["predicted_class"]=[1 if x > 0.001 else 0 for x in error_df_test["reconstruction_error"]]

        evaluate_model(error_df_test.true_class,error_df_test.predicted_class,error_df_test.reconstrution_error)


    

if __name__ == "__main__":
    tf.app.run()





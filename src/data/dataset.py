import tensorflow as tf
import os
def get_training_data(filepaths):
    '''Prepares training dataset with the tf-data API for the input pipeline.
    Reads TensorFlow Records files from the harddrive and applies several
    transformations to the files, like mini-batching, shuffling etc.
    
    @return dataset: the training dataset
    '''
    
    filenames=['data/preprocessed/tf_records_dir/tf_records_train/'+f for f in os.listdir(filepaths)]
    
    dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=32)
    dataset = dataset.map(parse,num_parallel_calls=4)
    dataset = dataset.shuffle(buffer_size=205275)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=64)
    dataset = dataset.prefetch(buffer_size=16)
    
    return dataset
 


def get_test_data(filepaths):
    '''Prepares validation dataset with the tf-data API for the input pipeline.
    Reads TensorFlow Records files from the harddrive and applies several
    transformations to the files, like mini-batching, shuffling etc.
    
    @return dataset: the validation dataset
    '''
    
    filenames=['data/preprocessed/tf_records_dir/tf_records_test/'+f for f in os.listdir(filepaths)]
    
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=51319)
    dataset = dataset.prefetch(buffer_size=1)
    
    return dataset

def get_validation_set_1(filepaths):
    '''Prepares test dataset with the tf-data API for the input pipeline.
    Reads TensorFlow Records files from the harddrive and applies several
    transformations to the files, like mini-batching, shuffling etc.
    
    @return dataset: the test dataset
    '''
    
    filenames=['data/preprocessed/tf_records_dir/tf_records_validation_set_1/'+f for f in os.listdir(filepaths)]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=14462)
    dataset = dataset.prefetch(buffer_size=1)
    
    return dataset


def get_validation_set_2(filepaths):
    '''Prepares test dataset with the tf-data API for the input pipeline.
    Reads TensorFlow Records files from the harddrive and applies several
    transformations to the files, like mini-batching, shuffling etc.
    
    @return dataset: the test dataset
    '''
    
    filenames=['data/preprocessed/tf_records_dir/tf_records_validation_set_2/'+f for f in os.listdir(filepaths)]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=13751)
    dataset = dataset.prefetch(buffer_size=1)
    
    return dataset


def parse(serialized):

    features={'features':tf.FixedLenFeature([29], tf.float32),
              'labels':tf.FixedLenFeature([1], tf.float32),
              }
    
    
    parsed_example=tf.parse_single_example(serialized,
                                           features=features,
                                           )
 
    features=parsed_example['features']
    label = tf.cast(parsed_example['labels'], tf.int32)
    
    return features, label
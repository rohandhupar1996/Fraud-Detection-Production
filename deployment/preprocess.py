import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# data_dir=os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','/Deep Learning Production/Fraud-Detection-Production/data/raw/creditcard_dataset.csv'))

def get_data(data_type="train",data_dir=""):
    """ This funtion load data and preprocess it and return train and test numpy values of it """
    if data_type=="train":
        print("train preprocessing")
        data=pd.read_csv(data_dir)
        min_max_scaler=MinMaxScaler()
        df_cred=data.drop("Time",axis=1)
        df_cred_scaled = min_max_scaler.fit_transform(df_cred.iloc[:,:-1])
        df_cred_normalized = pd.DataFrame(df_cred_scaled)
        df_cred_normalized["Class"]=df_cred["Class"]
        
        print("class count : ",df_cred_normalized["Class"].value_counts())

        df_cred_normalized_train=df_cred_normalized[df_cred_normalized["Class"]==0]
        df_cred_normalized_test=df_cred_normalized[df_cred_normalized["Class"]==1]    
        df_cred_normalized_test_part_1=df_cred_normalized_train.sample(frac=0.05)
        df_cred_normalized_train=df_cred_normalized_train.drop(df_cred_normalized_test_part_1.index)
        df_cred_normalized_test_part_2=df_cred_normalized_train.sample(frac=0.05)
        df_cred_normalized_train=df_cred_normalized_train.drop(df_cred_normalized_test_part_2.index)
        
        df_cred_normalized_test_class_1=df_cred_normalized_test.sample(frac=0.5)
        df_cred_normalized_validation_class_1=df_cred_normalized_test.drop(df_cred_normalized_test_class_1.index)
        
        print("fraud cases shape : ",df_cred_normalized_test_class_1.shape)
        
        df_cred_normalized_test_set=df_cred_normalized_test_part_1.append(df_cred_normalized_test_class_1)
        df_cred_normalized_validation_set=df_cred_normalized_test_part_2.append(df_cred_normalized_validation_class_1)
        
        print("train set dimensions :",df_cred_normalized_train.shape)
        print("test set dimensions :",df_cred_normalized_test_set.shape)
        print("validate set dimensions :",df_cred_normalized_validation_set.shape)
        
        print("class counts on validation set")
        print(df_cred_normalized_validation_set["Class"].value_counts())
        
        x_train, x_test = train_test_split(df_cred_normalized_train, test_size=0.2, random_state=2020)
        x_train = x_train[x_train.Class == 0]
        y_train = x_train["Class"]
        x_train = x_train.drop(['Class'], axis=1)
        y_test = x_test['Class']
        x_test = x_test.drop(['Class'], axis=1)
        x_train = x_train.values
        x_test = x_test.values
        print("train data set shape")
        print(x_train.shape)
        print("test data set shape")
        print(x_test.shape)
        y_train = y_train.values
        y_test = y_test.values
        
        x_val_set_1=df_cred_normalized_test_set.iloc[:,:-1]
        y_val_set_1=df_cred_normalized_test_set["Class"]
        x_val_set_1=x_val_set_1.values
        y_val_set_1=y_val_set_1.values
        
        x_val_set_2=df_cred_normalized_validation_set.iloc[:,:-1]
        y_val_set_2=df_cred_normalized_validation_set["Class"]
        x_val_set_2=x_val_set_2.values
        y_val_set_2=y_val_set_2.values
            
        return [x_train, y_train], [x_test, y_test] ,[x_val_set_1,y_val_set_1],[x_val_set_2,y_val_set_2]

    else :
        print("test preprocessing")
        data=pd.read_csv(data_dir)
        min_max_scaler=MinMaxScaler()
        df_cred=data.drop(["Time",'Class'],axis=1)
        df_cred_scaled = min_max_scaler.fit_transform(df_cred.iloc[:,:-1])
        df_cred_normalized = pd.DataFrame(df_cred_scaled)

        return df_cred_normalized
    

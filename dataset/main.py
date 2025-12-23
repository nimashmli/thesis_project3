import torch
import json
import sys
from dataset.extractor import DataExtractor 
from dataset.make_variable import dataset
import random
from  torch.utils.data import DataLoader , TensorDataset
from dataset.dataset_subject_dependet import kfold_dataset as dataset_for_subjet_dependet
import time

import torch.nn as nn 
# در فایل main.py
def prepar_dataset(test_person, over_lap, time_len , device, emotion, label_method , data_type):
    with open('dataset/config.json', 'r') as f:
        config = json.load(f)
    file_id = config['file_id']
    file_path = config['data_paths']

    extract_data = DataExtractor()
    extract_data.extract_data_file(file_id)

    data_manage = dataset(test_person, over_lap, time_len, emotion, label_method)
    data_manage.extract(file_path, data_type)
    data_manage.normalize()
    x_train, x_test, y_train, y_test = data_manage.recieve_data()

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    return x_train, x_test, y_train, y_test
class data : 
    def __init__(self , test_person, overlap, time_len, device, emotion, label_method, batch_size , data_type ) : 
        self.x_train, self.x_test, self.y_train,  self.y_test = prepar_dataset(test_person, overlap, time_len, device, emotion, label_method , data_type)
        test_dataset = TensorDataset(self.x_test , self.y_test)
        self.test_loader = DataLoader(test_dataset ,batch_size , shuffle=False)
        train_dataset = TensorDataset(self.x_train , self.y_train )
        self.train_loader = DataLoader(train_dataset , batch_size,shuffle=True )
    def train_data(self ) :
        return self.train_loader
    def  test_data(self ) : 
        return self.test_loader
    



def data_for_subject_dependet(overlap , time_len ,  emotion , label_method , data_type , device , k=5 ) :
    with open('dataset/config.json', 'r') as f:
        config = json.load(f)
    file_id = config['file_id']
    file_path = config['data_paths']
    extract_data = DataExtractor()
    extract_data.extract_data_file(file_id)
    data = dataset_for_subjet_dependet(data_type, file_path , label_method , overlap , time_len , emotion , device , k)
    return data 




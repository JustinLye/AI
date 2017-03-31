#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import math

def predict(w,x,b):
    return (1/(1+np.exp(-1*np.dot(x,w)+b)))

def update(y,fa,x,alpha):
    u = np.zeros((x.shape[1],y.shape[1])).astype(np.float32)
    for i in range(u.shape[1]):
        u[:,i] = (alpha * np.sum(((y-fa)*fa*(1-fa))[:,i][:,np.newaxis]*x,axis=0))
    return u
    
def save_model(outfile,w,b,label_count,feature_count,uniquetags):
    np.savez(outfile,w=w,b=b,label_count=label_count,feature_count=feature_count,uniquetags=uniquetags)
class trainer:
    input_file = 0
    validate_file = 0
    bias = 0
    feature_data = 0
    label_data = 0
    unique_labels = 0
    label_count = 0
    feature_count = 0
    example_count = 0
    target_values = 0
    targets = 0
    initial_weight_bound = 0
    batch_size = 0
    max_epoch = 0
    seed = 4788
    weights = 0
    learning_rate = 0
    sample_indices = 0
    batch_features = 0
    batch_targets = 0
    batch_prediction = 0
    batch_bias = 0
    validation_features = 0
    validation_targets = 0
    epoch = 0
    tlu_collection_errors = 0
    tlu_individual_errors = 0
    tlu_collection_errors_train = 0
    tlu_individual_train_errors = 0
    tlu_error_plot = 0
    last_five_errors = 0
    validation_bias = 0
    min_epoch = 0
    initial_learning_rate = 0
    def __init__(self, train_file, validate_file):
        self.input_file = np.loadtxt(train_file).astype(np.float32)
        self.feature_data = self.input_file[:,1:]
        self.label_data = self.input_file[:,0]
        self.unique_labels = np.unique(self.label_data)
        self.label_count = len(self.unique_labels)
        self.feature_count = self.feature_data.shape[1]
        self.example_count = self.feature_data.shape[0]
        self.target_values = [0.1,0.9]
        self.targets = np.zeros((self.example_count, self.label_count)).astype(np.float32)
        self.targets[:,:] = self.target_values[0]
        self.batch_size = 32
        self.max_epoch = 1000
        self.min_epoch = 5
        self.epoch = 0
        if len(sys.argv) == 6:
            self.initial_learning_rate = float(sys.argv[5])
        else:
            self.initial_learning_rate = 2.0
        self.seed = 47834
        #self.seed = 342
        np.random.seed(self.seed)
        self.sample_indices = np.arange(self.example_count).astype(int)
        
        for i in range(self.label_count):
            self.targets[self.label_data==self.unique_labels[i],i] = self.target_values[1]
        self.init_weights_and_bias()
        self.validate_file = np.loadtxt(validate_file)
        self.validation_features = self.validate_file[:,1:]
        self.validation_targets = self.validate_file[:,0].astype(int)
        self.validation_bias = np.zeros(self.validation_features.shape[0])[:,np.newaxis]
        self.tlu_collection_errors = np.zeros(0).astype(np.float32)
        self.tlu_collection_errors_train = np.zeros(0).astype(np.float32)
        self.tlu_individual_errors = np.zeros((1,self.label_count)).astype(np.float32)
        self.tlu_individual_train_errors = np.zeros((1,self.label_count)).astype(np.float32)
        self.last_five_errors = np.zeros(5).astype(np.float32)
        self.error_plot = np.zeros((1,3+self.label_count*2))
    def init_weights_and_bias(self):
        self.initial_weight_bound = 1/(1+math.sqrt(self.feature_count))
        self.weights = np.random.uniform(-self.initial_weight_bound,self.initial_weight_bound,self.feature_count*self.label_count).reshape((self.feature_count,self.label_count)).astype(np.float32)
        self.bias = np.random.uniform(-self.initial_weight_bound, self.initial_weight_bound,self.example_count).reshape((self.example_count,1)).astype(np.float32)
    def increment_lr(self):
        self.learning_rate = (self.initial_learning_rate/(1+math.pow(self.epoch,2)))
        
    def batch_step(self):
        step_count = 0
        np.random.shuffle(self.sample_indices)
        while step_count < self.example_count:
            self.batch_features = self.feature_data[self.sample_indices[step_count:step_count+self.batch_size]]
            self.batch_targets = self.targets[self.sample_indices[step_count:step_count+self.batch_size]]
            self.batch_bias = self.bias[self.sample_indices[step_count:step_count+self.batch_size]]
            self.batch_prediction = predict(self.weights,self.batch_features,self.batch_bias)
            self.weights = self.weights + update(self.batch_targets, self.batch_prediction, self.batch_features, self.learning_rate)
            step_count = step_count + self.batch_size
    
    def tlu_validate(self, tlu_num):
        pred = predict(self.weights[:,tlu_num][:,np.newaxis],self.validation_features,self.validation_bias)
        pred_train = predict(self.weights[:,tlu_num][:,np.newaxis],self.feature_data,self.bias)
        check = (pred >= 0.5)
        check_train = (pred_train >= 0.5)
        val = (self.validation_targets == self.unique_labels[tlu_num])[:,np.newaxis]
        val_train = (self.label_data == self.unique_labels[tlu_num])[:,np.newaxis]
        err = (check==val).astype(int)
        err_train = (check_train == val_train).astype(int)
        return 1-sum(err)/len(err), 1-sum(err_train)/len(err_train)

    def validate(self):
        batch_error = np.zeros((1,self.label_count)).astype(np.float32)
        batch_error_train = np.zeros((1,self.label_count)).astype(np.float32)
        tlu_valid_err = 0
        tlu_train_err = 0
        valerrtrain = 0
        for i in range(self.label_count):
            tlu_valid_err, tlu_train_err = self.tlu_validate(i)
            batch_error[:,i] = tlu_valid_err
            batch_error_train[:,i] = tlu_train_err
        if self.epoch == 1:
            self.tlu_individual_errors = batch_error.copy()
            self.tlu_individual_train_errors = batch_error_train.copy()
        else:
            self.tlu_individual_errors = np.append(self.tlu_individual_errors,batch_error,axis=0)
            self.tlu_individual_train_errors = np.append(self.tlu_individual_train_errors,batch_error_train,axis=0)
        v = predict(self.weights,self.validation_features,self.validation_bias)
        v_train = predict(self.weights,self.feature_data,self.bias)
        p = np.argmax(v,axis=1).astype(int)
        p_train = np.argmax(v_train,axis=1).astype(int)
        valerr = (p == self.validation_targets)
        valerrtrain = (p_train == self.label_data)
        return sum(valerr)/len(valerr),sum(valerrtrain)/len(valerrtrain)
    def train(self):
        self.epoch = 0
        acc = 0
        acc_train = 0
        max_acc = 0
        for i in range(self.max_epoch):
            self.epoch = self.epoch + 1
            self.increment_lr()
            self.batch_step()
            acc,acc_train = self.validate()
            max_acc = self.last_five_errors[np.argmax(self.last_five_errors)]
            self.tlu_collection_errors = np.append(self.tlu_collection_errors,1-acc)
            self.tlu_collection_errors_train = np.append(self.tlu_collection_errors_train,1-acc_train)
            self.last_five_errors = np.roll(self.last_five_errors,-1)
            self.last_five_errors[len(self.last_five_errors)-1] = acc
            if self.epoch == 1:
                self.error_plot[0,:] = np.append(np.append(np.append(np.append([self.epoch],self.tlu_individual_train_errors[0,:]),self.tlu_collection_errors_train[0]),self.tlu_individual_errors[0,:]),self.tlu_collection_errors[0])
            else:
                self.error_plot = np.insert(self.error_plot,self.epoch-1,np.append(np.append(np.append(np.append([self.epoch],self.tlu_individual_train_errors[self.epoch-1,:]),self.tlu_collection_errors_train[self.epoch-1]),self.tlu_individual_errors[self.epoch-1,:]),self.tlu_collection_errors[self.epoch-1]),axis=0)
            if acc < max_acc and self.epoch > self.min_epoch:
                break


    
if __name__ == '__main__':
    t = trainer(sys.argv[1],sys.argv[2])
    t.train()
    save_model(sys.argv[3],t.weights,t.bias,t.label_count, t.feature_count, t.unique_labels)
    if len(sys.argv) >= 5:
        np.savetxt(sys.argv[4],t.error_plot)
    
    




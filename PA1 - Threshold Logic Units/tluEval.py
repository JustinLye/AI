import sys
import numpy as np
import math

def load_model(infile):
    model = np.load(infile)
    w = model['w']
    b = model['b']
    label_count = model['label_count']
    unique_tags = model['uniquetags']
    feature_count = model['feature_count']
    return w,b,int(label_count),feature_count,unique_tags

def predict(w,x,b):
    return (1/(1+np.exp(-np.dot(x,w)+b)))


class tlu:
    weights = 0
    bias = 0
    classification = 0
    def activation(self,x):
        p = predict(self.weights,x,self.bias[0:x.shape[0]])
        a = p>=0.5
        return a
    def __init__(self,w,b,c):
        self.weights = w
        self.bias = b
        self.classification = c
if __name__ == '__main__':
    w,b,label_count,feature_count,u = load_model(sys.argv[2])
    test_file = np.loadtxt(sys.argv[1])
    test_features = test_file[:,1:]
    test_labels = test_file[:,0][:,np.newaxis]
    individual_error = np.zeros((int(label_count))).astype(np.float32)
    
    tlu_b = b[0:test_file.shape[0]].copy()
    for i in range(int(label_count)):
        t = tlu(w[:,i][:,np.newaxis],tlu_b,i)
        pred = t.activation(test_features)
        tar = test_labels == u[i]
        acc = (pred == tar).astype(int)
        individual_error[i] = (1-(sum(acc)/len(acc)))
    
    p = predict(w,test_features,tlu_b)
    p=np.argmax(p,axis=1).astype(int)[:,np.newaxis]
    print("collective error:")
    g = (p==test_labels).astype(int)
    print(1-(sum(g)/len(g)))
    print("individual error:")
    print(individual_error[:,np.newaxis])

    




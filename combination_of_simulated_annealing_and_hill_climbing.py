import random
import numpy as np
import pandas as pd
import copy
import time
from sklearn.preprocessing import OneHotEncoder
import numpy.random as rn
import matplotlib.pyplot as plt  # to plot
import matplotlib as mpl
import numpy as np
from scipy import optimize       # to compare

from sklearn.model_selection import train_test_split

l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l

eyes = generateColumns(1, 12)

def random_start():
    a, b = interval
    return a + (b - a) * rn.random_sample()
def random_neighbour(nn, fraction=1):
    temp=copy.deepcopy(nn)
    for i in range(4):  
        temp.biases[0][i] +=random.uniform(-0.5, 0.5)*6
        for j in range(24):
            temp.weights[0][i][j] += random.uniform(-0.5, 0.5)*5
    for i in range(2):  
        temp.biases[0][i] += random.uniform(-0.5, 0.5)*2
        for j in range(4):
            temp.weights[0][i][j] += random.uniform(-0.5, 0.5)*4
    return temp
def acceptance_probability(cost, new_cost, temperature):
    if new_cost > cost:
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        return p
def temperature(fraction):
    return max(0.01, min(1, 1 - fraction))

class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
        # helper variables
        self.bias_nitem = sum(sizes[1:])
        self.weight_nitem = sum([self.weights[i].size for i in range(self.num_layers-2)])
    def relu(self,z):
        return np.maximum(z, 0)
    def feedforward(self, a):
        #print(a)
        cat=0
        '''Return the output of the network if ``a`` is input.'''
        for b, w in zip(self.biases, self.weights):
            #print(cat)
            
            if(cat==0):
                a=self.relu(np.dot(w,a)+b)
            else:
                a = self.sigmoid(np.dot(w,a)+b)
            cat+=1
        return a
   
    def sigmoid(self, z):
        '''The sigmoid function.'''
        return 1.0/(1.0+np.exp(-z))
    

    def score(self, X, y):

        total_score=0
        for i in range(X.shape[0]):
            predicted = self.feedforward(X[i].reshape(-1,1))
            actual = y[i].reshape(-1,1)
            #print((np.power(predicted-actual,2)/2))
            total_score += np.sum(np.power(predicted-actual,2)/2)  # mean-squared error
        return total_score/len(X)

    def accuracy(self, X, y):
        accuracy = 0
        for i in range(X.shape[0]):
            output = self.feedforward(X[i].reshape(-1,1))
            accuracy += int(np.argmax(output) == np.argmax(y[i]))
        return accuracy / X.shape[0] * 100


def simulated_anneling(nn,X,y):
    temp=copy.deepcopy(nn)
   
    cost=nn.accuracy(X,y)
    maxsteps=1000
    best_cost=cost
    for step in range(1000):
        fraction = step / float(maxsteps)
        T = temperature(fraction)

        new_state = random_neighbour(temp, fraction)
       
        new_cost =new_state.accuracy(X,y)
        
        if acceptance_probability(cost, new_cost, T) > rn.random():
            temp= copy.deepcopy(new_state)
            cost=new_cost
        if(new_cost>best_cost):
            best_cost=new_cost
            nn=copy.deepcopy(new_state)
    
    return nn
def hill(nn,X,y):
    temp=copy.deepcopy(nn)

    final=nn.accuracy(X,y)
    
    count=0
    while(count!=1000 and final <95):
        count+=1
       
        for i in range(4):  
            temp.biases[0][i] += random.uniform(-0.5, 0.5)*8
            for j in range(24):
           
                temp.weights[0][i][j] += random.uniform(-0.5, 0.5)*6
        for i in range(2):  
            temp.biases[0][i] += random.uniform(-0.5, 0.5)*2
            for j in range(4):
             
                temp.weights[1][i][j] += random.uniform(-0.5, 0.5)*4
         
        tm=temp.accuracy(X,y)
        if(tm>final):
            #print(temp.accuracy(X,y),nn.accuracy(X,y),"hhh")
            nn=copy.deepcopy(temp)
            #print(final,tm)
            final=tm
           
        temp=copy.deepcopy(nn)
   
    return nn
            
def main():

    df = pd.read_csv("Eyes.csv")
    X = df[eyes]
    y = df['truth_value']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

    
    X=X_train.values
    y=y_train.values
    
    y = y.reshape(-1, 1)
    
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()

    
    N_POPS = 15
    NET_SIZE = [24,4,2]
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.4
    RETAIN_RATE = 0.4

   
    nn=Network(NET_SIZE)
    start_time = time.time()
   
    nn=simulated_anneling(nn,X,y)
    #print(nn.accuracy(X,y),"simulated_anneling accuracy")
    tr=hill(nn,X,y)
    
    print("traning accuracy",tr.accuracy(X,y))#,high_score.accuracy(X,y))
    X=X_test.values[:]
    y_test=y_test.values
    y_test = y_test.reshape(-1, 1)
    
    enc.fit(y_test)
    y_test = enc.transform(y_test).toarray()

    y=y_test[:]
    print("test acuuracy",tr.accuracy(X,y))
    print("Execution time in seconds = ",(time.time() - start_time))
if __name__ == "__main__":
    main()
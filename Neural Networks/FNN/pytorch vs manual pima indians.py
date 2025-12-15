"""
Created by Taylor Richards
taylordrichards@gmail.com
April 23, 2024

https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

I need to use hooks to inspect the inside of a pytorch model, basically, while it is executing. 
https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from math import log, exp
from copy import deepcopy
import logging
#-------------------------------------

#global var
check_layer = 0 #debugging step

def ReLU(x):
    #I have to put 0.0 as the first argument so that the numpy vectorizer defaults to float as its output type
    return(max(0.0, x))

def d_ReLU(x):
    if x > 0: #I had the break point at x >= 0 before, but changing it to > 0 made everything work properly
        return(1)
    else:
        return(0) #oh my god I took this derivative wrong. good god almighty.
    
def d_ReLU_branchless(x):
    a = max(0,x)
    b = x
    r = a and a/b or 0
    return(r)

def sigmoid(x):
  return 1 / (1 + exp(-x))

def J(y, y_hat):
    """BCE"""
    return(-1*(y*log(y_hat)+(1-y)*log(1-y_hat)))

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def main():

    #region pytorch model
    start_time = time.perf_counter()

    dataset = np.loadtxt('pima-indians-diabetes one row.csv', delimiter=',')
    Xnp = dataset[:,0:8]
    ynp = dataset[:,8]

    X = torch.tensor(Xnp, dtype=torch.float32)
    y = torch.tensor(ynp, dtype=torch.float32).reshape(-1, 1)
    
    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    
    
    def hook_log(self, input, output):
        logging.info(f'input: {input}')

        

    model[0].register_forward_hook(hook_log)
    model[1].register_forward_hook(hook_log)
    model[2].register_forward_hook(hook_log)
    model[3].register_forward_hook(hook_log)
    model[4].register_forward_hook(hook_log)
    model[5].register_forward_hook(hook_log)

    # model[0].register_backward_hook(hook_log)
    # model[1].register_backward_hook(hook_log)
    # model[2].register_backward_hook(hook_log)
    # model[3].register_backward_hook(hook_log)
    # model[4].register_backward_hook(hook_log)
    # model[5].register_backward_hook(hook_log) 


    #print(model)

    # for name, param in g:
    #     print(name, param)
    #     arr = param.detach().cpu().numpy()
    #     print(arr)
    # d = dict(g)
    # print(d.items())

    #need to make sure that the param.detach() isn't merely a reference, but rather is a copy.
    param_names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
    arr_params = [deepcopy(param.detach().cpu().numpy()) for _, param in model.named_parameters()]
    named_params = dict(zip(param_names, arr_params))
    
    #print(next(model.named_parameters()))
    #https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for j in range(1):
        for i in range(0, len(X)):
            optimizer.zero_grad() #This zeros out the gradients, as their default behaviour is to accumulate.
            Xbatch = X[i:i+1]
            y_pred = model(Xbatch)
            ybatch = y[i:i+1]
            loss = loss_fn(y_pred, ybatch)
            if i == check_layer:
                print(f'pytorch y_pred: {y_pred}, ybatch: {ybatch}')
                print(f'pytorch loss: {loss}')
            loss.backward()
            if i == check_layer:
                for name, param in model.named_parameters():
                    print(name, param.grad)
            optimizer.step()

    #print(f'latest loss {loss}')

    with torch.no_grad():
        y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"pytorch accuracy {accuracy}")
    
    #endregion

    vReLU = np.vectorize(ReLU)
    #vsig = np.vectorize(sigmoid)
    vd_ReLU = np.vectorize(d_ReLU)
    
    #Get the column vectors to actually be columns.
    named_params['b1'] = np.reshape(named_params['b1'], (12, 1))
    named_params['b2'] = np.reshape(named_params['b2'], (8, 1))
    named_params['b3'] = np.reshape(named_params['b3'], (1, 1))

    # xtest = [6,148,72,35,0,33.6,0.627,50]
    # ytest = 1
    # _,y_hat,_ = single_pass(xtest, ybatch, **named_params)
    # print(y_hat)

    class FNN():

        def __init__(self, starting_weights, lr=0.001):
            self.lr = lr
            self.W1 = starting_weights.get('W1')
            #print(self.W1)
            self.W2 = starting_weights.get('W2')
            self.W3 = starting_weights.get('W3')
            self.b1 = starting_weights.get('b1')
            self.b2 = starting_weights.get('b2')
            self.b3 = starting_weights.get('b3')
            self.d_J_d_W3 = None
            self.d_J_d_b3 = None
            self.d_J_d_W2 = None
            self.d_J_d_b2 = None
            self.d_J_d_W1 = None
            self.d_J_d_b1 = None
            self.h = None
            self.a = None
            self.y_hat = None
            #The following are just for inspection purposes:
            self.z = None
            self.θ = None
            self.t = None

        def forward_prop(self, x):

            self.z = self.W1@x + self.b1
            self.h = vReLU(self.z)
            #print(f'h shape: {h.shape}, should be (12,1) | W1 shape: {W1.shape}, should be (12,8)')

            self.θ = self.W2@self.h + self.b2
            self.a = vReLU(self.θ)
            #print(f'a shape: {a.shape}, should be (8,1) | W2 shape: {W2.shape}, should be (8,12)')

            self.t = self.W3@self.a + self.b3
            self.y_hat = sigmoid(self.t[0][0]) #doesn't need to be vectorised; one-dimensional.
            #print(f'W3 shape: {W3.shape}, should be (1,8)')
            return(self.y_hat)

        def backward_prop(self, x, y):
            d1 = self.y_hat - y
            d1 = np.reshape(d1,(1,1))
            #print(f'd1 shape: {d1.shape}, should be (1,1)')
            #print(f' d1@W3: {np.shape(d1@W3)}, should be (1,8)')
            d2 = np.multiply(d1@self.W3, np.transpose(vd_ReLU(self.a)))
            #print(f'd2 shape: {d2.shape}, should be (1,8)')
            d3 = np.multiply(d2@self.W2, np.transpose(vd_ReLU(self.h)))
            #print(f'd3 shape: {d3.shape}, should be (1,12)')

            self.d_J_d_W3 = np.outer(d1, self.a)
            #print(f'd_J_d_W3 shape: {d_J_d_W3.shape}, should be (1,8)')
            self.d_J_d_b3 = d1
            #print(f'd_J_d_b3 shape: {d_J_d_b3.shape}, should be (1,1)')
            self.d_J_d_W2 = np.outer(d2, self.h)
            #print(f'd_J_d_W2 shape: {d_J_d_W2.shape}, should be (8,12)')
            self.d_J_d_b2 = np.transpose(d2)
            #print(f'd_J_d_b2 shape: {d_J_d_b2.shape}, should be (8,1)')
            self.d_J_d_W1 = np.outer(d3, x)
            #print(f'd_J_d_W1 shape: {d_J_d_W1.shape}, should be (12,8)')
            self.d_J_d_b1 = np.transpose(d3)
            #print(f'd_J_d_b1 shape: {d_J_d_b1.shape}, should be (12,1)')
        
        def update(self):
            self.W3 -= self.lr*self.d_J_d_W3
            self.b3 -= self.lr*self.d_J_d_b3
            self.W2 -= self.lr*self.d_J_d_W2 
            self.b2 -= self.lr*self.d_J_d_b2
            self.W1 -= self.lr*self.d_J_d_W1
            self.b1 -= self.lr*self.d_J_d_b1

    #train
    M = FNN(starting_weights = named_params)
    for j in range(1):
        for i in range(0, len(Xnp)):
            Xbatch = np.transpose(Xnp[i:i+1])
            ybatch = ynp[i:i+1]
            y_hat = M.forward_prop(Xbatch)
            loss = J(ybatch, y_hat)
            if i == check_layer + 1:
                print(f'taylor y_hat: {y_hat}, ybatch: {ybatch}')
                print(f'taylor loss: {loss}')
                print(f'0. input: {Xbatch}')
                print(f'1. z: {M.z}')
                print(f'2. h: {M.h}')
                print(f'3. θ: {M.θ}')
                print(f'4. a: {M.a}')
                print(f'5. t: {M.t}')
                
            M.backward_prop(Xbatch, ybatch)
            if i == check_layer:
                print(f'6. d_J_d_W3: {M.d_J_d_W3}')
                print(f'7. d_J_d_b3: {M.d_J_d_b3}')
                print(f'8. d_J_d_W2: {M.d_J_d_W2}')
                print(f'9. d_J_d_b2: {M.d_J_d_b2}')
                print(f'10. d_J_d_W1: {M.d_J_d_W1}')
                print(f'11. d_J_d_b1: {M.d_J_d_b1}')
            M.update
    
    #check accuracy
    correct = 0
    for i in range(0, len(Xnp)):
        Xbatch = np.transpose(Xnp[i:i+1])
        ybatch = ynp[i:i+1]
        y_hat = M.forward_prop(Xbatch)
        if round(y_hat) == ybatch:
            correct +=1
        accuracy = correct/len(Xnp)

    print(f'taylor accuracy: {accuracy}')

    #print(f'latest loss {loss}')
    end_time = time.perf_counter()
    print("--- %s seconds ---" % (end_time - start_time))

if __name__ == "__main__":
    main()
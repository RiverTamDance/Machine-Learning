"""
Created by Taylor Richards
taylordrichards@gmail.com
April 22, 2024

https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
"""
import time
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
   return(sigmoid(x)*(1-sigmoid(x)))

def J(actual, observed):
   stream = zip(actual, observed)
   squared = [(p[0]-p[1])**2 for p in stream]
   return((1/2)*sum(squared))

def d_J(actual, observed):
   return(np.subtract(observed, actual))

def main():
    start_time = time.perf_counter()

    vsig = np.vectorize(sigmoid)
    vd_sig = np.vectorize(d_sigmoid)
    learn_rate = 0.5

    #initialize our weights, inputs, and outputs
    x = np.array([0.05, 0.1])
    y = np.array([0.01, .99])

    W = np.array([[0.15, 0.2],[0.25, 0.3]])
    b1 = np.array([0.35, 0.35])

    U = np.array([[.4,.45],[.5,.55]])
    b2 = np.array([.6,.6])
    #-----------------------------------------

    #forward propagation--------------
    z = W@x + b1
    h = vsig(z)

    th = U@h + b2
    y_hat = vsig(th)

    loss = J(y, y_hat)
    #------------------------------------

    #back propagation---------------------
    d1 = np.multiply(d_J(y, y_hat),vd_sig(th))
    d2 = np.multiply(d1@U,vd_sig(z))

    #according to numpy, I don't need to transpose these below
    #because these are 1D vectors which basically can't be tranposed
    #I dont like this :)
    d_J_d_U = np.outer(d1, h) 
    d_J_d_b2 = d1
    d_J_d_W = np.outer(d2,x)
    d_J_d_b1 = d2
    #------------------------------------

    #gradient descent--------------------- 
    U_new = U - learn_rate*d_J_d_U
    b2_new = b2 - learn_rate*d_J_d_b2
    W_new = W - learn_rate*d_J_d_W
    b1_new = b1 - learn_rate*d_J_d_b1
    #------------------------------------

    end_time = time.perf_counter()
    print("--- %s seconds ---" % (end_time - start_time))

if __name__ == "__main__":
    main()
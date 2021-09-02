from operator import mul
from functools import partial
from itertools import accumulate, cycle, islice

data = [([1,1],0), ([4,4],1), ([5,5],1), ([2,2],0), ([2,6],0), ([3,5],0)]

#prepends 1 to each variable-list, which is a dummy value to allow the threshold calculations to work.
def data_prepend_one(data):
    
    pdata = [([1]+x[0], x[1]) for x in data]

    return(pdata)

def data_pipeline(data):

    data = data_prepend_one(data)

    return(data)
#this calculates our predicted value for y, while the true value is d=x[1]
def f(w,x):
    
    if sum(map(mul, w, x)) <= 0:
        y = 0
    else:
        y = 1
    
    return(y)

#this gives data the form (x,d,y)
def yvals(w, data):

    ydata = [(x[0], x[1], f(w,x[0])) for x in data]

    return(ydata)

def step2a(w, data):

    data = yvals(w, data)

    return(data)

def weight_math(wi_1,xi,d,y,r):

    wi_2 = wi_1 + r*(d-y)*xi

    return(wi_2)

def step2b(w,a,r):

    wm_const = partial(weight_math,d=a[1], y=a[2], r=r)
    w = map(wm_const, w, a[0])

    return(list(w))

###################3
""" fundamentally, I need to deal with 2 different timings: the ys get updated after n data points have been consumed, whereas
    the ws get updated after every datapoint. hmm i wonder if I should try a recursive approach. 
    
"""

def perceptron(r, data, w=None):

    if w is None:
         w = [0] * (len(data[0][0]))

    data_y = step2a(w, data)
    #This transposes my data with the predicted ys, so that we have a row of y values and a row of d values, which are then
    # very simple to check for equality. really it is zip(*lst) that does all the work, the rest is just converting back into
    # list form. 
    t_data_y = list(map(list, zip(*data_y)))
    
    #if we find a w i.e. a hyperplane that fits our data we stop.
    if t_data_y[-1] == t_data_y[-2]:
        return([])
    else:
        w = [round(i,1) for i in list(accumulate(data_y, partial(step2b, r=r), initial = w))[-1]]
        return([w] + perceptron(r, data, w))

data = data_pipeline(data)
print(perceptron(0.6, data))



""" mismatch = True

while mismatch == True:

    w = list(accumulate(data_y, partial(step2b, r=r), initial = w))[-1]

    data_y = data_prep(w, data)
    mismatch = ismatch(data_y)

print(w) """

#w = accumulate(datan, partial(step2b, r=r), initial = w)
#print(list(w))

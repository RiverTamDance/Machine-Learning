from operator import mul
from functools import partial


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

def perceptron(r, data, w_init = None):

    bool_stop = False
    if w_init is None:
            w_init = [0] * (len(data[0][0]))
    wList = [w_init]

    while bool_stop == False:
        #stopping condition.
        data_y = step2a(wList[-1], data)
        t_data_y = list(map(list, zip(*data_y)))
        #if we find a w i.e. a hyperplane that fits our data we stop.
        if t_data_y[-1] == t_data_y[-2]:
            bool_stop = True
        else:
            for a in data_y:

                w = [round(i, 1) for i in step2b(wList[-1],a,r)]
                wList = wList + [w]

    return(wList)

data = data_pipeline(data)
print(perceptron(0.6, data))
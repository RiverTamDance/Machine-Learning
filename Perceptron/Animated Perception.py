import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import cycle
import time
import sys
sys.path.insert(1, 'C:\\Users\\Taylo\\OneDrive\\Documents\\Python work\\Machine Learning\\Perceptron')

from Perceptron_Imperative import perceptron, data_pipeline

data = [([1,1],0), ([4,4],1), ([5,5],1), ([2,2],0), ([2,6],0), ([3,5],0)]

start_time = time.time()

#https://matplotlib.org/stable/tutorials/introductory/usage.html

#subplots is kinda a normal starting point for any MPL work, as figures and axes are the main things you need to create visuals
#figsize is just saying that the plot should be 5 inches by 3 inches. Width, height in inches.adjusts the fig.
fig, ax = plt.subplots(figsize=(5, 5))
#Set the x-axis view limits.
ax.set(xlim=(0, 7), ylim=(0, 7))
#This creates an nd-array which contains 91 equally spaced interval numbers from -3 to 3, inclusive.
x = np.linspace(0, 8, 100)
#t = np.linspace(1, 25, 30)
#https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy
#X2, T2 = np.meshgrid(x, t)
 
# sinT2 = np.sin(2*np.pi*T2/T2.max())
# F = 0.9*sinT2*np.sinc(X2*(1 + sinT2))
data0 = [d[0] for d in data]
h, v = list(map(list, zip(*data0)))
scat = ax.scatter(h,v)

c = np.array([1, 2, 3, 4,5,6])
scat.set_array(c)

data = data_pipeline(data)
wList = perceptron(0.6, data)[1:]

#here we compute the first y values for our line. Just calling them F because i'm lazy.
F = x*0

line = ax.plot(x, F, color='k', lw=2)[0]

# def init():
#     line.set_data([], [])
#     return line,

def animate(i):

    w = wList[i]

    if w[2] == 0:
        F = np.linspace(.5,.5,100)
    else:
        F = (-1*x*w[1]/w[2])-(w[0]/w[2])

    line.set_ydata(F)

    l = [1]*6
    l[i % 6] = 2
    c = np.array(l)
    scat.set_array(c)

anim = FuncAnimation(
    fig, animate, interval=500, frames=len(wList), repeat = True, repeat_delay = 5000)
 
# plt.draw()
# plt.show()

anim.save("test1.mp4")

print("--- %s seconds ---" % (time.time() - start_time))


# I think the next way to go with this is to only change the "active datapoint" if it changed w
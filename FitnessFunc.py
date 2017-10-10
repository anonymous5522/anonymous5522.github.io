from numpy import *
from PlotFunction import plotFun3d

def RosenBrock(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    return 100*(x2-x1**2)**2 + (1-x1)**2

def RosenBrockGrad(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    return mat([[400*x1*(x1**2-x2)+2*(x1-1)], 
                [200*(x2-x1**2)]])  

def peaks(x):
    y = float(x[1, 0])
    x = float(x[0, 0])
    return 3*(1-x)**2*exp(-x**2-(y+1)**2) \
            - 10*(x/5-x**3-y**5)*exp(-x**2-y**2) \
            - exp(-(x+1)**2-y**2)/3.0

def peaks_(x):
    y = float(x[1, 0])
    x = float(x[0, 0])
    res = 3*(1-x)**2*exp(-x**2-(y+1)**2) \
            - 10*(x/5-x**3-y**5)*exp(-x**2-y**2) \
            - exp(-(x+1)**2-y**2)/3.0
    return -res

def Ackley(x):
    n = shape(x)[0]
    return -20*exp(-0.2*sqrt(float(x.T * x)/n)) - exp(sum(cos(2*pi*x))/n) + 20


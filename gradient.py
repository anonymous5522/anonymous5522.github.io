from numpy import *
from linearSearch import linearSearch_Secant, plotFun2d, plotFun3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def GradientDescend(x0, fun, grad, eps = 1e-6, maxIter = 50):
    '''
    auto select step size
    '''
    iter = 0
    x = mat(x0)
    n, m = shape(x)
    if n < m:
        x = x.T
    gradVal = grad(x)
    def f(alpha):
        return fun(x - alpha * gradVal)
    def g(alpha):
        return gradVal.T * grad(x - alpha * gradVal)
    print('iter =', iter)
    print('x =', x)
    print('gx =', gradVal)
    print('fx =', fun(x), '\n')
    while gradVal.T * gradVal > eps and iter < 50:
        iter += 1
        alpha = linearSearch_Secant(f, g)
        x = x - alpha * gradVal
        gradVal = grad(x)
        print('iter =', iter)
        print('alpha =', alpha)
        print('x =', x)
        print('gx =', gradVal)
        print('fx =', fun(x), '\n')
    return x

def GradientDescendQuaForm(x0, Q, b, c = 0, eps = 1e-6, maxIter = 50):
    '''
    f(x) = (1/2)*x.T*Q*x - b.T*x
    '''
    def g(x):
        return Q*x-b
    def f(x):
        return 0.5*x.T*Q*x-b.T*x
    iter = 0
    x = mat(x0)
    gTg = float(g(x).T * g(x))
    gTQg = float(g(x).T * Q * g(x))
    while gTg > eps and iter < maxIter:
        iter += 1
        x = x - gTg * g(x) / gTQg
        gTg = float(g(x).T * g(x))
        gTQg = float(g(x).T * Q * g(x))
        print('iter =', iter)
        print('x =', x)
        print('gx =', g(x))
        print('fx =', f(x), '\n')
    return x

def example8_1():
    def fun(x):
        x = mat(x)
        return power(x[0, 0]-4, 4) + power(x[1, 0]-3, 2) + 4*power(x[2, 0]+5, 4)
    def grad(x):    
        x = mat(x)
        return mat([4*power(x[0, 0]-4, 3), 2*(x[1, 0]-3), 16*power(x[2, 0]+5, 3)]).T
    x0 = mat([[4], [2], [-1]])
    x = GradientDescend(x0, fun, grad)
    print(x)
    
def ex8_26():
    '''
    Rosenbrock function
    '''
    def fun(x):
        return 100*(x[1, 0]**2-x[0, 0]**2)**2 + (1-x[0, 0])**2
    def grad(x):
        return mat([[400*(x[0, 0]**2-x[1, 0]**2)*x[0, 0]+2*(x[0, 0]-1)], [400*x[1, 0]*(x[1, 0]**2-x[0, 0]**2)]])  
    x0 = mat([-2, 2]).T
    x = GradientDescend(x0, fun, grad, eps = 1e-10)
    print(x)
    '''
    ax = plotFun3d(fun, -5, 5, -5, 5)
    ax.scatter(x[0, 0], x[1, 0], fun(x), c = 'r', s = 100)
    plt.show()
    '''

def example8_2():
    Q1 = mat(eye(2))
    Q2 = mat([[0.2, 0], [0, 1]])
    x0 = mat([[10], [10.5]])
    b = zeros((2, 1))
    x = GradientDescendQuaForm(x0, Q2, b)    
    print(x)






    
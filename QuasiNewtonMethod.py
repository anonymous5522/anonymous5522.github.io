from numpy import *
from linearSearch import linearSearch_Secant, linearSearch_Golden,\
    linearSearch_secant
from PlotFunction import plotContour, plotFun2d


def QuasiNewton(fun, grad, x0, method = 'rank1', eps = 1e-6, maxIter = 50, printInfo = False, plotInfo = False):
    x = x0
    gradVal = grad(x)
    d = -1 * gradVal
    n = shape(x)[0]
    H = eye(n)
    if printInfo==True:
        print('iter =', 0)
        print('x =', x)
        print('grad(x) =', gradVal)
        print('f(x) =', fun(x), '\n')
    for iter in range(maxIter):
        if gradVal.T * gradVal < eps:
            break
        d = -1 * H * gradVal
        alpha = linearSearch_secant(fun, grad, x, d)
        deltaX = alpha * d
        x = x + deltaX
        deltaG = grad(x) - gradVal
        gradVal = gradVal + deltaG
        z = deltaX - H * deltaG
        if method=='rank1':
            H = H + z * z.T / float(deltaG.T * z)
        elif method=='dfp':
            H = H + (deltaX * deltaX.T / (deltaX.T * deltaG)) \
                 - ((H * deltaG) * (H * deltaG).T / (deltaG.T * H * deltaG))
        elif method=='bfgs':
            H = H + float(1 + deltaG.T * H * deltaG / (deltaG.T * deltaX))   \
                  * (deltaX * deltaX.T) / (deltaX.T * deltaG)   \
                  - (H * deltaG * deltaX.T + mat(H * deltaG * deltaX.T).T) / (deltaG.T * deltaX)
        else:
            print('param method error!')
            break
        if printInfo==True:
            print('iter =', iter+1)
            print('x =', x)
            print('alpha =', alpha)
            print('d =', d)
            print('grad(x) =', gradVal)
            print('f(x) =', fun(x), '\n')
        if plotInfo==True:
            pass
    return x


def example11_1():
    fun = lambda x: 0.5*x.T*mat([[2, 0], [0, 1]])*x + 3
    grad = lambda x: mat([[2, 0], [0, 1]]) * x
    x0 = mat([[1], [2]])
    x = QuasiNewton(fun, grad, x0, method= 'bfgs', printInfo = True)
    print(x)
    
example11_1()
    
    
    
    
    
    
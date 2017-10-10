from numpy import *
from linearSearch import linearSearch_Secant, linearSearch_Golden
from PlotFunction import plotContour, plotFun2d
import matplotlib.pyplot as plt
from FitnessFunc import RosenBrock, RosenBrockGrad

def ConjGradientQuaForm(x0, Q, b, eps = 1e-6, printInfo = False):
    '''
    f(x) = 1/2 * x.T * Q * x - x.T * b
    grad(x) = Q * x - b
    '''
    f = lambda x: 0.5*x.T*Q*x - x.T*b
    g = lambda x: Q*x - b
    gradVal = g(x0)
    if abs(gradVal.T * gradVal) < eps:
        return x0
    direct = -1 * gradVal
    x = mat(x0)
    n = shape(Q)[0]
    if printInfo==True:
        print('Initial State:')
        print('Q =', Q)
        print('b =', b)
        print('x0 =', x0, '\n')
    for iter in range(n):
        gTd = float(gradVal.T * direct)
        dTQd = float(direct.T * Q * direct)
        alpha = -1 * gTd / dTQd                 # update step size alpha
        x = x + alpha * direct                  # update x
        if abs(gradVal.T * gradVal) < eps:
            break
        gradVal = g(x)
        gTQd = float(gradVal.T * Q * direct)
        beta = gTQd / dTQd
        direct = -1 * gradVal + beta * direct   # update direct
        if printInfo==True:
            print('iter =', iter+1)
            print('x =', x)
            print('grad =', g(x))
            print('f(x) =', f(x))
            print('alpha =', alpha)
            print('direct =', direct, '\n')
    return x

def ConjGradient(x0, fun, grad, method = 'hs', dReset = 5, eps = 1e-6, maxIter = 50, printInfo = False):
    '''
    method = 
    'hs' for Hestense-Stiefel
    'pr' for Polak-Ribiere
    'fr' for Flecher-Reeves
    '''
    x = mat(x0)
    xSeq = [[float(x[0, 0])], [float(x[1, 0])]]
    fxSeq = [float(fun(x))]
    n = shape(x)[0]
    gradVal = grad(x)
    if abs(gradVal.T * gradVal) < eps:
        return x, xSeq, fxSeq
    direct = -1 * gradVal
    if printInfo==True:
        print('Initial State:')
        print('x0 =', x0)
        print('direct =', direct, '\n')
    for iter in range(maxIter):
        f = lambda alpha: fun(x + alpha * direct)
        g = lambda alpha: float(direct.T * grad(x + alpha * direct))
        alpha = linearSearch_Golden(f, g, eps)
        x = x + alpha * direct
        xSeq[0].append(float(x[0, 0]))
        xSeq[1].append(float(x[1, 0]))
        fxSeq.append(float(fun(x)))
        prevGradVal = gradVal
        gradVal = grad(x)
        if abs(gradVal.T * gradVal) < eps:
            break
        if (iter+1)%dReset==0:
            direct = -1 * grad(x)
        else:
            diff = gradVal - prevGradVal
            if method=='hs':
                beta = float(gradVal.T * diff) / float(direct.T * diff)
            elif method=='pr':
                beta = float(gradVal.T * diff) / float(prevGradVal.T * prevGradVal)
            elif method=='fr':
                beta = float(gradVal.T * gradVal) / float(prevGradVal.T * prevGradVal)
            else:
                print('parameter method is wrong!')
                return x, xSeq, fxSeq
            direct = -1 * gradVal + beta * direct
        if printInfo==True:
            print('iter =', iter+1)
            print('x =', x)
            print('grad =', grad(x))
            print('f(x) =', fun(x))
            print('alpha =', alpha)
            print('direct =', direct, '\n')
    return x, xSeq, fxSeq
      
def ex10_11():
    import matplotlib.pyplot as plt
    x0 = mat([-2, 2]).T
    x, xSeq, fxSeq = ConjGradient(x0, RosenBrock, RosenBrockGrad, method = 'hs', dReset = 6, printInfo = True)
    print(x)
    
    ax = plotContour(RosenBrock, -3, 3, -3, 3, delta = 0.1)
    n = 15
    ax.plot(xSeq[0][-n:-1], xSeq[1][-n:-1])
    ax.scatter(xSeq[0][-n:-1], xSeq[1][-n:-1], c = 'r')
    plt.show()
    
    '''
    fig = plt.figure()
    plt.plot(fxSeq)
    plt.show()
    '''
    
def example10_3():
    Q = mat([[3, 0, 1], [0, 4, 2], [1, 2, 3]])
    b = mat([3, 0, 1]).T
    x0 = mat([0, 0, 0]).T
    x = ConjGradientQuaForm(x0, Q, b, printInfo=True)
    print(x)
    
ex10_11()







        
        
        
        
        
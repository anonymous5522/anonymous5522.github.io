from numpy import *
import matplotlib.pyplot as plt

def GoldenSeg(initL, initR, fun, eps = 1e-6, rou = 1-0.61803, maxIter = 50):
    l0 = initL
    r0 = initR
    l1 = l0 + rou * (r0 - l0)
    r1 = r0 - rou * (r0 - l0)
    funValL0 = fun(l0)
    funValR0 = fun(r0)
    funValL1 = fun(l1)
    funValR1 = fun(r1)
    for iter in range(1, maxIter):
        '''
        print('iter =', iter)
        print('a0 =', l0, 'fa0 =', funValL0) 
        print('a1 =', l1, 'fa1 =', funValL1) 
        print('b0 =', r0, 'fb0 =', funValR0) 
        print('b1 =', r1, 'fb1 =', funValR1)
        print() 
        '''
        if r0-l0 < eps:
            break
        if funValL1 < funValR1:
            r0 = r1
            funValR0 = funValR1
            r1 = l1
            funValR1 = funValL1
            l1 = l0 + rou * (r0 - l0)
            funValL1 = fun(l1)
        else:
            l0 = l1
            funValL0 = funValL1
            l1 = r1
            funValL1 = funValR1
            r1 = r0 - rou * (r0 - l0)
            funValR1 = fun(r1)    
    '''    
    print('total iter =', iter)
    print('total compress ratio =', rou**iter)
    print('answer =', l0, r0, '\n')
    '''
    return l0, r0      
            
            
def FibSeg(initL, initR, fun, eps = 1e-6):
    '''
    def fib(n):
        sqrt5 = sqrt(5.0)
        return (((1+sqrt5)/2)**n - ((1-sqrt5)/2)**n) / sqrt5
    '''
    l0 = initL
    r0 = initR
    funValL0 = fun(l0)
    funValR0 = fun(r0)
    fib = [0, 1, 1]
    for i in range(3, 51):
        fib.append(fib[-1] + fib[-2])
        if fib[i] >= (r0-l0)/eps:
            n = i-1
            break
    rou = 1 - float(fib[n]) / fib[n+1]
    l1 = l0 + rou * (r0 - l0)
    r1 = r0 - rou * (r0 - l0)
    funValL1 = fun(l1)
    funValR1 = fun(r1)
    for i in range(1, n):
        '''
        print('iter =', i)
        print('rou =', rou)
        print('a0 =', l0, 'fa0 =', funValL0) 
        print('a1 =', l1, 'fa1 =', funValL1) 
        print('b0 =', r0, 'fb0 =', funValR0) 
        print('b1 =', r1, 'fb1 =', funValR1)
        print() 
        '''
        rou = 1 - float(fib[n-i]) / fib[n+1-i]
        if funValL1 < funValR1:
            r0 = r1
            funValR0 = funValR1
            r1 = l1
            funValR1 = funValL1
            l1 = l0 + rou * (r0 - l0)
            funValL1 = fun(l1)
        else:
            l0 = l1
            funValL0 = funValL1
            l1 = r1
            funValL1 = funValR1
            r1 = r0 - rou * (r0 - l0)
            funValR1 = fun(r1)
    print('total iter =', i)
    print('total compress ratio =', 1.0/fib[n+1])
    print('answer =', l0, r0, '\n')
    return l0, r0 
    
def BinSeg(initL, initR, grad):
    pass

def Secant(initL, initR, grad, eps = 1e-6, maxIter = 500):
    x0 = initL
    x1 = initR
    for i in range(maxIter):
        if not abs(x0-x1) > abs(x0)*eps:
            break
        tmp = x1
        x1 = (grad(x1)*x0 - grad(x0)*x1) / (grad(x1) - grad(x0))
        x0 = tmp
        print('x1 =', x1)
        print('gradX1 =', grad(x1))
        print()
    return x1

def linearSearch_secant(fun, grad, x, d, eps = 1e-6, maxIter = 50):
    f = lambda alpha: fun(x + alpha * d)
    g = lambda alpha: d.T * grad(x + alpha * d)
    a0 = 0
    a1 = 0 + 10 * eps
    #print('range =', a0, '~', a1)
    for iter in range(maxIter):
        if abs(g((a1 + a0)/2)) <= (eps * abs(g(0))):
            break
        tmp = a1
        a1 = float(a1 - ((a1-a0) / (g(a1)-g(a0)) ) * g(a1))
        a0 = tmp
    return (a1+a0)/2

def linearSearch_Secant(fun, grad, eps = 1e-6, maxIter = 50):
    a0, a1 = findRange(fun)
    #print('range =', a0, '~', a1)
    for iter in range(maxIter):
        if abs(grad(a1)) <= eps * abs(grad(0)):
            break
        tmp = a1
        a1 = float(a1 - ((a1-a0) / (grad(a1)-grad(a0))) * grad(a1))
        a0 = tmp
    return (a1+a0)/2
    
def linearSearch_Golden(fun, grad, eps = 1e-6, maxIter = 50):  
    a0, a1 = findRange(fun)
    a0, a1 = GoldenSeg(a0, a1, fun, eps)
    return (a0+a1)/2

def Newton(initL, initR, grad1, grad2):
    pass

def ArmijoFindRange():
    pass

def findRange(fun, initL = 0, initR = 1, maxIter = 500):
    x0 = initL
    x1 = (initL + initR) / 2.0
    x2 = initR
    f0 = fun(x0)
    f1 = fun(x1)
    f2 = fun(x2)
    for iter in range(maxIter):
        if f2>f1 and f0>f1:
            break;
        if f2 < f1 and f1 < f0:
            dist = x2 - x1
            x0 = x1; f0 = f1
            x1 = x2; f1 = f2
            x2 = x2 + dist * 2; f2 = fun(x2)
        else:
            dist = x1 - x0
            x2 = x1; f2 = f1
            x1 = x0; f1 = f0
            x0 = x0 - dist * 2; f0 = fun(x0)
    return x0, x2
    
    

def plotFun2d(fun, l = -10, r = 10):
    x = linspace(l, r, 100)
    y = []
    for i in x:
        y.append(float(fun(i)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    return ax

def plotFun3d(fun, x0, x1, y0, y1):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D 
    x = linspace(x0, x1, int(abs(x0-x1)*20))
    y = linspace(y0, y1, int(abs(y0-y1)*20))
    x, y = meshgrid(x, y)
    n, m = shape(x)
    z = zeros((n, m))
    for i in range(n):
        for j in range(m):
            z[i, j] = fun(mat([x[i, j], y[i, j]]).T)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    return ax
    
def normalize(x):
    normx = x.T * x
    x = x / normx

def testfun(x):
    return power(x, 4) - 14 * power(x, 3) + 60 * power(x, 2) - 70 * x

def testfun1(x):
    return 1*power(x, 3)-12.2*power(x, 2)+7.45*x+42
    

def ex7_12():
    def fun1(x):
        return 0.5 * x.T * mat([[2, 1], [1, 2]]) * x
    def grad1(x):
        return mat([[2, 1], [1, 2]]).T * x
    p0 = mat([[0.8], [-0.25]])
    d = -1 * grad1(p0)
    def f(alpha):
        return fun1(p0 + alpha * d)
    def g(alpha):
        return d.T * grad1(p0 + alpha * d)
    
    #a0, a2 = findRange(f, 0, 0.075)!!!!!!!??????wtf???
    a0, a2 = findRange(f, 10, 11)
    print(a0, a2)
    print('Golden Segmentation:')
    alpha0, alpha1 = GoldenSeg(a0, a2, f, eps = 0.01)
    print('alpha =', alpha0/2 + alpha1/2)
    print('min =', f(alpha0/2 + alpha1/2), '\n')
    print('Fibonacci Segmentation:')
    alpha0, alpha1 = FibSeg(a0, a2, f, eps = 0.01)
    print('alpha =', alpha0/2 + alpha1/2)
    print('min =', f(alpha0/2 + alpha1/2), '\n')
    
    print('Secant:')
    alpha = linearSearch_Secant(f, g, eps = 0.01)
    print('alpha =', alpha)
    print('min =', f(alpha))
    
    plotFun2d(f, -2, 2)
    plt.show()

def ex14_1():
    fun = lambda x: (x[1, 0]-x[0, 0])**4 + 12*x[1, 0]*x[0, 0] - x[0, 0] + x[1, 0] - 3











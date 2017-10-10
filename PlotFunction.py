from numpy import *

def plotFun2d(fun, l = -10, r = 10, show = True):
    import matplotlib.pyplot as plt
    x = linspace(l, r, 100)
    y = []
    for i in x:
        y.append(float(fun(i)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    if show==True:
        plt.show()
    else:
        return ax

def plotFun3d(fun, x0 = -10, x1 = 10, y0 = -10, y1 = 10, show = True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D 
    x = linspace(x0, x1, int(abs(x0-x1)*10))
    y = linspace(y0, y1, int(abs(y0-y1)*10))
    x, y = meshgrid(x, y)
    n, m = shape(x)
    z = zeros((n, m))
    for i in range(n):
        for j in range(m):
            z[i, j] = fun(mat([x[i, j], y[i, j]]).T)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    if show==True:
        plt.show()
    else:
        return ax

def plotContour(fun, x0 = -10, x1 = 10, y0 = -10, y1 = 10, delta = 0.1, show = True):
    import matplotlib.pyplot as plt
    x = arange(x0, x1, delta)
    y = arange(y0, y1, delta)
    X, Y = meshgrid(x, y)
    n, m = shape(X)
    Z = zeros((n, m))
    for i in range(n):
        for j in range(m):
            Z[i, j] = fun(mat([X[i, j], Y[i, j]]).T)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(X, Y, Z)
    if show==True:
        plt.show()
    else:
        return ax
    
    
    
    

    
    
    
    
    
    
    
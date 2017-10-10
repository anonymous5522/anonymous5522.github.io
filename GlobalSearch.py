from numpy import *
from PlotFunction import plotFun3d, plotContour, plotFun2d
import matplotlib.pyplot as plt
from FitnessFunc import peaks_

def createSimplex(x0, lamb, constraint):
    pass
def NelderMeadSimplex(fun, xmat, 
                      rou = 1, chi = 2, gamma = 0.5, sigma = 0.5, 
                      eps = 1e-6, maxIter = 50, printInfo = False):
    '''
    rou: reflect
    chi: extend
    gamma: shrink
    sigma: compress
    
    '''
    n, m = shape(xmat)
    psSqx = []
    psSqy = []
    if m!=n+1:
        print('need n+1 initial points')
        return 0
    ps = pl = pnl = 0
    changedIdx = m
    diff = inf
    pG = mat(zeros((n, 1)))
    pR = mat(zeros((n, 1)))
    funValList = zeros(m)
    for iter in range(maxIter):
        for j in range(m):
            funValList[j] = float(fun(xmat[:, j]))
        srtIdx = argsort(funValList) # ascend
        for j in range(m-1):
            if srtIdx[j]==changedIdx and funValList[srtIdx[j]]==funValList[srtIdx[j+1]]:
                tmp = srtIdx[j]; srtIdx[j-1] = srtIdx[j]; srtIdx[j] = tmp 
        ps = srtIdx[0]; pl = srtIdx[-1]; pnl = srtIdx[-2]
        psSqx.append(xmat[0, ps])
        psSqy.append(xmat[1, ps])
        oldpG = pG; pG = (sum(xmat, 1)-xmat[:, pl]) / float(m-1)
        diff = pG - oldpG; diff = float(diff.T*diff)
        print('diff =', diff)
        if diff < eps and iter > 0:
            break
        # reflect
        pR = pG + rou * (pG - xmat[:, pl])
        fr = fun(pR)
        fs = funValList[ps]
        fl = funValList[pl]
        fnl = funValList[pnl]
        if printInfo==True:
            print('iter =', iter+1)
            print('ps =', xmat[:, ps])
            print('fs =', fs, '\n')
        if fs <= fr and fr < fnl: # replace pl with pR
            xmat[:, pl] = pR
        elif fr < fs: # extend
            pE = pG + chi * (pR - pG)
            fe = fun(pE)
            if fe < fr:
                xmat[:, pl] = pE
            else: # fe >= fr
                xmat[:, pl] = pR
        else: # fr >= fnl
            if fr >= fnl and fr < fl: # out shrink
                pC = pG + gamma * (pR - pG)
            
            else: # inner shrink
                pC = pG + gamma * (xmat[:, pl] - pG)
            fc = fun(pC)
            if fc <= fl: # success, replace pl with pc
                xmat[:, pl] = pC
            else: # failed, create a new simplex
                for j in range(m):
                    if j!=ps:
                        xmat[:, j] = xmat[:, ps] + sigma * (xmat[:, j] - xmat[:, ps])
        changedIdx = pl
    return xmat[:, ps], psSqx, psSqy

def randomSearch(fun, x, alpha, maxIter = 100, printInfo = False):
    n = shape(x)[0]
    z = mat(zeros((n, 1)))
    for iter in range(maxIter):
        z = x + random.uniform(-alpha, alpha, (n, 1))
        fz = fun(z); fx = fun(x)
        if fz < fx:
            x = z
        if printInfo:
            print('iter =', iter+1)
            print('x =', x.T)
            print('fx =', fx, '\n')
    return x

def HajekClSch(k, gamma = 1):
    return gamma / log(k + 2)
    
def simulatedAnnealing(fun, x, alpha, T = HajekClSch, maxIter = 100, printInfo = False):
    n = shape(x)[0]
    z = mat(zeros((n, 1)))
    pbest = mat(zeros((n, 1)))
    fbest = inf
    fx = fun(x)
    for iter in range(1, maxIter):
        z = x + random.uniform(-alpha, alpha, (n, 1))
        fz = fun(z)
        if fz < fx:
            x = z; fx = fz
            if fz < fbest:
               pbest = z; fbest = fz
        else:
            p = min(1, exp((fx-fz)/T(iter)))
            if random.rand() < p:
                x = z
                fx = fz                
        if printInfo:
            print('iter =', iter)
            print('x =', x.T)
            print('fx =', fx, '\n')
    return pbest

def clip(x, minVal, maxVal):
    n, m = shape(x)
    for i in range(n):
        for j in range(m):
            x[i, j] = min(maxVal, max(minVal, x[i, j]))
    return x

def particleSwarm(fun, n, m, xrg, vrg = 1, 
                  omega = 0.8, c1 = 2, c2 = 2, 
                  maxIter = 100, printInfo = False, plotInfo = False):
    '''
    n: dimension of input
    m: number of particles
    rg: feasible region
    omega: inertia
    c1: self-cognition
    c2: society
    '''
    xmat = mat(random.uniform(-xrg, xrg, (n, m)))
    pmat = xmat
    fpmat = zeros(m)
    for j in range(m):
        fpmat[j] = fun(xmat[:, j])
    vmat = mat(random.uniform(-vrg, vrg, (n, m)))
    rmat = zeros(n)
    smat = zeros(n)
    minidx = argmin(fpmat)
    gbest = xmat[:, minidx]
    fgbest = fpmat[minidx]
    for iter in range(maxIter):
        rmat = random.rand(n, m)
        smat = random.rand(n, m)
        vmat = omega*vmat \
             + c1*multiply(rmat, pmat-xmat) \
             + c2*multiply(smat, tile(gbest, (1, m))-xmat)
        clip(vmat, -vrg, vrg)
        xmat = xmat + vmat
        for j in range(m):
            fx = fun(xmat[:, j])
            if fx < fpmat[j]:
                pmat[:, j] = xmat[:, j]
                fpmat[j] = fx
                if fx < fgbest:
                    gbest = xmat[:, j]
                    fgbest = fx
        if printInfo:
            print('iter =', iter)
            print('gbest =', gbest)
            print('fgbest =', fgbest, '\n')
        if plotInfo and iter%10==0:
            ax = plotContour(peaks_, -xrg, xrg, -xrg, xrg, show = False)
            ax.scatter(xmat[0, :], xmat[1, :], c = 'r')
            plt.show()
    return gbest

def cvt(n, b = 2):
    i,f = divmod(n,1)
    
    ls=[]
    while i>0:
        i,m = divmod(i,b)
        ls.append(str(int(m)))
    rslt = ''.join(ls[::-1])
    
    ls=[]    
    while f>0:
        i,f = divmod(f*b,1)
        ls.append(str(int(i)))
    rslt += ('.' + ''.join(ls))

    return rslt

def encodeBin(xmat, rg, L = 24):
    n, m = shape(xmat)
    encodedList = [''] * m
    stepSize = 2.0 * rg / 2**L
    for j in range(m):
        for i in range(n):
            b = bin(int((xmat[i, j]+rg)/stepSize-stepSize))[2:]
            encodedList[j] += '0'*(L-len(b)) + b
    return encodedList
    
def decodeBin(bList, rg, L = 24):
    m = len(bList)
    n = int(len(bList[0]) / L)
    bmat = mat(zeros((n, m)), dtype = 'str')
    for i in range(n):
        for j in range(m):
            bmat[i, j] = bList[j][L*i:L*i+L]
    decodedMat = mat(zeros((n, m)))
    stepSize = 2.0 * rg / 2**L
    for i in range(n):
        for j in range(m):
            decodedMat[i, j] = stepSize * int(str(bmat[i, j]), 2) - rg
    return decodedMat

def toBin(xmat, L1 = 10, L2 = 20, d = 5):
    '''
    integer: 3(10) digit
    decimal: 7(10) digit
    d: accuracy in decimal
    d = log10(2**L2)
    '''
    n, m = shape(xmat)
    encodedMat = mat(zeros((n, m)), dtype = 'str')
    for i in range(n):
        for j in range(m):
            x = float(xmat[i, j])
            neg = False
            if x < 0:
                x = -x
                neg = True
            b1 = bin(int(x))[2:]
            b1 = '0'*(L1-len(b1)) + b1
            f = int((x-int(x))*10**d)
            b2 = bin(f)[2:]
            lenb2 = len(b2)
            b2 = '0'*(L2-lenb2) + b2
            if neg==True:
                encodedMat[i, j] =  '-' + b1 + b2
            else:
                encodedMat[i, j] = b1 + b2
    return encodedMat
    
def toDec(bmat, L1 = 10, L2 = 20, d = 5):
    n, m = shape(bmat)
    decodedMat = mat(zeros((n, m)))
    for i in range(n):
        for j in range(m):
            b = str(bmat[i, j])
            sign = 1
            if b[0]=='-':
                sign = -1
                b = b[1:]
            decodedMat[i, j] = sign * (int(b[:L1], 2) + int(b[L1:], 2)/10**d)
    return decodedMat

def createM(P, fP):
    n, m = shape(P)
    M = mat(zeros((n, m)))
    rnd = random.rand(m)
    F = sum(fP)
    cufP = fP/F
    for j in range(1, m):
        cufP[j] += cufP[j-1]
    for j in range(m):
        idx = 0
        while cufP[idx] < rnd[j]:
            idx += 1
        M[:, j] = P[:, j]
    return M

def crossBin(enM, i, j, l, r):
    tmp = enM[i][l:r]
    enM[i][l:r] = enM[j][l:r]
    enM[j][l:r] = tmp

def mutateBin(enMList, pM):
    m = len(enMList)
    L = len(str(enMList[0]))
    rnd = random.rand(m, L)
    returnList = [''] * m
    for j in range(m):
        for k in range(L):
            if rnd[j, k] < pM:
                if enMList[j][k]=='1':
                    returnList[j] += '0'
                else:
                    returnList[j] += '1'
            else:
                returnList[j] += enMList[j][k]
    return returnList 

def crossRealMean(enM, i, j):
    enM[:, j] = (enM[:, i] + enM[:, j]) / 2
    enM[:, i] = enM[:, j]
    
def crossRealMeanRnd(enM, i, j):    
    tmp = (enM[:, i] + enM[:, j]) / 2
    enM[:, j] = tmp + random.normal(0, 1, shape(tmp))
    enM[:, i] = tmp + random.normal(0, 1, shape(tmp))
    
def crossRealConv(enM, i, j):
    enMi = enM[:, i]; enMj = enM[:, j]
    alpha = random.rand()
    enM[:, i] = alpha * enMi + (1 - alpha) * enMj
    enM[:, j] = (1 - alpha) * enMi + alpha * enMj
    
def crossRealConvRnd(enM, i, j):
    enMi = enM[:, i]; enMj = enM[:, j]
    alpha = random.rand()
    enM[:, i] = alpha * enMi + (1 - alpha) * enMj + random.normal(0, 1, shape(enMi))
    enM[:, j] = (1 - alpha) * enMi + alpha * enMj + random.normal(0, 1, shape(enMj))
    
def mutateRealRnd(enM, pM):
    n, m = shape(enM)
    for j in range(m):
        if random.rand() < pM:
            enM[:, j] += random.normal(0, 1, (n, 1))
    return enM

def mutateRealConv(enM, pM, rg):
    n, m = shape(enM)
    for j in range(m):
        if random.rand() < pM:
            alpha = random.rand()
            enM[:, j] = alpha * enM[:, j] + (1 - alpha) * rndPointInRg(rg)
    return enM

def rndPointInRg(rg):
    n = shape(rg)[0]
    p = zeros(n)
    for i in range(n):
        p[i] = random.uniform(rg[i, 0], rg[i, 1])
    return p

def geneticAlgorithm(fun, n, m, rg, 
                     encodeType = 'bin', cross = crossBin, mutate = mutateBin, 
                     L = 24, numPrt = 2, numCrsP = 1,pM = 0.01,  
                     maxIter = 50, printInfo = False, plotInfo = False):
    '''
    fun: target function
    n: input size
    m: population capacity
    
    rg: feasible region
        
        rg = [[dim1 min, dim1 max],    
              [dim2 min, dim2 max], 
               ...
                                  ] 
              
        or rg is a real number, which means -rg < dim n < rg      
        
    encodeType: 'bin' for binary, 'real' for real number
    cross, mutate: function
    L: number of encode digits
    numPrt: number of parent chromosome
    numCrsP: number of cross point
    pM: probability of mutate
    '''
    if shape(rg)==(1, 1):
        P = mat(random.uniform(-rg, rg, (n, m)))
    else:
        n = shape(rg)[0]
        P = mat(zeros((n, m)))
        for i in range(n):
            P[i, :] = mat(random.uniform(rg[i, 0], rg[i, 1], (1, m)))
    fP = zeros(m)
    best = zeros((n, 1))
    fbest = inf
    for j in range(m):
        fP[j] = float(fun(P[:, j]))
    if encodeType == 'bin':
        for iter in range(maxIter):
            # create match pool
            enMList = encodeBin(createM(P, fP), rg, L)
            # choose parent chromosome
            arr = array(range(0, m)); random.shuffle(arr)
            prtIdx = arr[:numPrt]
            # cross
            for i in range(numPrt//2):
                crsP = [0] + list(random.randint(1, n*L-1, numCrsP)) + [n*L-1]
                for j in range(numCrsP//2):
                    cross(enMList, i = prtIdx[2*i], j = prtIdx[2*i+1], l = crsP[2*j], r = crsP[2*j+1])
            # mutate
            enMList = mutate(enMList, pM)
            # new population
            P = decodeBin(enMList, rg, L)
            for j in range(m):
                fP[j] = float(fun(P[:, j]))
                if fP[j] < fbest:
                    fbest = fP[j]
                    best = P[:, j]
            if printInfo:
                print('iter =', iter)
                print('best =', best)
                print('fbest =', fbest, '\n')
    elif encodeType=='real':
        for iter in range(maxIter):
            # create match pool
            enM = createM(P, fP)
            # choose parent chromosome
            arr = array(range(0, m)); random.shuffle(arr)
            prtIdx = arr[:numPrt]
            # cross
            for i in range(numPrt//2):
                cross(enM, prtIdx[i*2], prtIdx[i*2+1])
            # mutate
            if mutate.__name__ == 'mutateRealRnd':
                enM = mutate(enM, pM)
            elif mutate.__name__ == 'mutateRealConvRnd':
                enM = mutate(enM, pM, rg)
            # new population
            P = enM
            for j in range(m):
                fP[j] = float(fun(mat(P[:, j])))
                if fP[j] < fbest:
                    fbest = fP[j]
                    best = P[:, j]
            if printInfo:
                print('iter =', iter)
                print('best =', best)
                print('fbest =', fbest, '\n')
    else:
        print('invalid encode type')
    return best
        
def ex14_1():
    fun = lambda x: (x[1, 0]-x[0, 0])**4 + 12*x[1, 0]*x[0, 0] - x[0, 0] + x[1, 0] - 3
    xmat = mat(zeros((2, 3)))
    x0 = mat([[0.55], [0.7]])
    lamb = 0.1
    xmat[:, 0] = x0
    xmat[:, 1] = x0 + lamb * mat([[0], [1]])
    xmat[:, 2] = x0 + lamb * mat([[1], [0]])
    ps, psx, psy = NelderMeadSimplex(fun, xmat, eps = 1e-32, printInfo = True)
    print(ps)
    ax = plotContour(fun, -1, 1, -1, 1, show = False)
    ax.scatter(psx, psy, c = 'r')
    ax.plot(psx, psy)
    plt.show()
    
def ex14_2():
    x0 = mat([-1, 0]).T
    x = simulatedAnnealing(peaks_, x0, alpha = 2, maxIter = 10000, printInfo = True)
    print(x)

def ex14_3():
    x = particleSwarm(peaks_, 2, 10, 10, omega = 0.5, maxIter = 1000, printInfo = True, plotInfo = True)
    print(x)

def example14_4():
    x = geneticAlgorithm(peaks_, n = 2, m = 20, rg = 3, maxIter = 500, printInfo = True)
    print(x)

def ex14_11():
    fun = lambda x: -1 * (-15*power(sin(2*x), 2) - (x-2)**2 + 160)
    x = geneticAlgorithm(fun, 1, 20, 10, maxIter = 500, printInfo = True)
    plotFun2d(fun)

def ex14_12():
    fun = lambda x: x[0, 0]*sin(x[0, 0]) + x[1, 0]*sin(x[1, 0])
    rg = mat([[0, 10], [4, 6]])
    x = geneticAlgorithm(fun, 2, 20, rg, encodeType = 'real', cross = crossRealConv, mutate = mutateRealConv, maxIter = 500, printInfo = True)
    plotContour(fun, 0, 10, 4, 6)


ex14_12()
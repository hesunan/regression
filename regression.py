from numpy import *

def loadDataSet(fileName):      
    numFeat = len(open(fileName).readline().split('\t')) - 1 #属性特征个数
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():     #遍历每一行
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  #判断行列式是否为0
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)  #w=[(xT*x)^-1]*xT*y
    return ws



#**********绘制原始数据的散点图*********************
def plotData():
    import matplotlib.pyplot as plt
    filename = 'E:\machine learning_programs\ex0.txt'
    xArr, yArr = loadDataSet(filename)
    xMat = mat(xArr);yMat = mat(yArr)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    '''
     取第二个特征绘图
     flatten()将矩阵转化成一维数组
     matrix.A属性返回矩阵变成的数组，和getA()方法一样
     '''
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    #绘制拟合直线
    xCopy = xMat.copy()
    xCopy.sort(0)#从小到大进行排序
    weights = standRegres(xArr, yArr)
    print (weights.shape)
    yHat = xCopy * weights # yHat表示拟合直线的纵坐标，用回归系数求出
    ax.plot(xCopy[:,1], yHat, c = 'green')
    plt.show()


#**********测试单点*******************
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]#矩阵行数（样例个数）
    weights = mat(eye((m)))
    for j in range(m):                      
        diffMat = testPoint - xMat[j,:]     
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    #return testPoint * ws
    print (testPoint*ws)
    
#*************测试多点************************
def lwlrTest(testArr,xArr,yArr,k=1.0):  
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def showlwlr():
    import matplotlib.pyplot as plt
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = mat(xArr)     
    fig = plt.figure() #创建绘图对象
    ax = fig.add_subplot(111)  #111表示将画布划分为1行2列选择使用从上到下第一块        
    #scatter绘制散点图
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T[:,0].flatten().A[0],s=2,c='red')
    
    srtInd = xMat[:,1].argsort(0)#返回升序后下标    
    '''等价于xSort=xMat[srtInd.flatten().A[0]]'''
    xSort = xMat[srtInd][:,0,:] #将xMat升序排列  
    ax.plot(xSort[:,1],yHat[srtInd])
    plt.show()


#****************岭回归***************************
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam    #denom=xT*x+λI(I是单位矩阵)
    if linalg.det(denom) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
    
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     
    #regularize X's
    xMeans = mean(xMat,0)   #获得均值
    xVar = var(xMat,0)      #计算方差
    '''
    def regularize(xMat):  # 特征归一化
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  
    inVar = var(inMat, 0)  
    inMat = (inMat - inMeans) / inVar
    return inMat
    xMat = regularize(xMat) #封装为函数的形式
    '''
    xMat = (xMat - xMeans)/xVar#数据标准化
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))#在30个lamda下调用该函数，计算回归系数
        wMat[i,:]=ws.T
    return wMat

'''
#****************************error***************************
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

abX, abY = loadDataSet("abalone.txt")
yHat01   = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1    = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10   = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

print ("k=0.1,训练集上的误差：",rssError(abY[0:99], yHat01.T))

'''
def regularize(xMat):  # 特征归一化
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  
    inVar = var(inMat, 0)  
    inMat = (inMat - inMeans) / inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     
    xMat = regularize(xMat)#归一化x
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) 
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        #print ws.T
        lowestError = inf; 
        for j in range(n):#对每个特征进行遍历
            '''
            对每个特征进行增大或减小
            改变一个系数得到一个新的w
            计算误差，若误差小于当前误差，则对w进行更新
            '''
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat


from time import sleep
import json
#import urllib2
import urllib.request
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print ("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print ('problem with item %d' % i)
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#90%的样本用于训练
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:       #10%用于测试
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #标准化数据后求得w
        for k in range(30):
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #标准化
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#y估计值
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    print ("the best model from Ridge Regression is:\n",unReg)
    print ("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))#对数据进行还原


''' 
if __name__ == "__main__":
    #plotData()
    abx,aby=loadDataSet('abalone.txt')
    ridgeWeights=ridgeTest(abx,aby)
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()  
'''   


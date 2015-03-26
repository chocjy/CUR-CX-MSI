import numpy as np
import time

def cxPre(A, rank=5, objectiveDim=1):

        #Compute the CX decomposition
        levScores, p = compLevExact(A, rank, objectiveDim)
        infIndices = levScores.argsort()[::-1]
        sortedLevScores = levScores[infIndices]

        #Safe the output results
        return infIndices, sortedLevScores


def cx(A, k, c, lev='exact'):

        if lev == 'exact':
            colLev, p = compLevExact(A, k, 1)
            p = colLev/k
        elif lev == 'apprSpe':
            colLev, p = compLevApprSpe(A.T, k, 5, 8000)
        elif lev == 'apprFro':
            colLev, p = compLevApprFro(A.T, k, 500) 
        else:
            print 'Please enter a valid method!'

        bins = np.add.accumulate(p)
        colInd = np.digitize(np.random.random_sample(c), bins)

        C = np.dot(A[:,colInd], np.diag(1/p[colInd]))
        X = np.dot(np.linalg.pinv(C), A)

        return C, X, colLev


def cx_unif(A, c):

        n, d = A.shape

        p = np.ones(d)/float(d)

        bins = np.add.accumulate(p)
        colInd = np.digitize(np.random.random_sample(c), bins)

        #C = np.dot(A[:,colInd], np.diag(1/p[colInd]))
        C = A[:,colInd]

        X = np.dot(np.linalg.pinv(C), A)

        return C, X


def cx_deter(A, k, c):

        ind, colLev = cxPre(A, k, 1)
        C = np.dot(A[:,ind[:c]], np.diag(1/colLev[:c]))

        X = np.dot(np.linalg.pinv(C), A)

        return C, X, colLev


def cur_cx(A, k, c, r):

        colLev = compLevExact(A, k, 1)
        rowLev = compLevExact(A, k, 0)

        p = colLev / k
        bins = np.add.accumulate(p)
        colInd = np.digitize(np.random.random_sample(c), bins)

        C = np.dot(A[:,colInd], np.diag(1/p[colInd]))

        p = rowLev / k
        bins = np.add.accumulate(p)
        rowInd = np.digitize(np.random.random_sample(r), bins)

        R = np.dot(np.diag(1/p[rowInd]), A[rowInd,:])
       
        U = np.dot( np.dot(np.linalg.pinv(C), A), np.linalg.pinv(R) )

        return C, U, R, rowLev, colLev 


def cur_cx_deter(A, k, c, r):

        ind, colLev = cxPre(A, k, 1)
        C = np.dot(A[:,ind[:c]], np.diag(1/colLev[:c]))

        ind, rowLev = cxPre(A, k, 0)
        R = np.dot(np.diag(1/rowLev[:r]), A[ind[:r],:])

        U = np.dot( np.dot(np.linalg.pinv(C), A), np.linalg.pinv(R) )

        return C, U, R, rowLev, colLev


def compLevExact(A, k, axis):
        """ This function computes the column or row leverage scores of the input matrix.
          
            :param A: n-by-d matrix
            :param k: rank parameter, k <= min(n,d)
            :param axis: 0: compute row leverage scores; 1: compute column leverage scores.
        
            :returns: 1D array of leverage scores. If axis = 0, the length of lev is n.  otherwise, the length of lev is d.
        """

        U, D, V = np.linalg.svd(A, full_matrices=False)

        if axis == 0:
            lev = np.sum(U[:,:k]**2,axis=1)
        else:
            lev = np.sum(V[:k,:]**2,axis=0)

        p = lev/k

        return lev, p


def SHRT(A, r):

        from fht import fht
    
        n, d = A.shape
        n1 = 2**(np.ceil(np.log2(n)))

        A = np.concatenate((A, np.zeros((n1-n,d))),0)

        #bins = np.add.accumulate(np.ones(n1)/n1)
        #S = np.digitize(np.random.random_sample(m), bins)

        #S = np.random.rand(n1) < float(r)/n1
        S = np.random.permutation(int(n1))[:r]

        #m = len(np.where(S == True)[0])
        result = np.zeros((r,d))

        D = np.array(np.random.rand(n1) < 0.5, dtype = 'int')*2 - 1  
        for j in range(d):
            result[:,j] = fht(D*A[:,j])[S]

        return result
    

def compLevAppr(A, r1, secondProj='', r2=1):
 
        n, d = A.shape
        PA = SHRT(A, r1) 

        R = np.linalg.qr(PA, 'r')
        R = np.linalg.inv(R)

        if secondProj == 'Gaussian':
            G = np.random.normal(0,1,r2)
            R = np.dot(R, G)

        lev = np.sum(np.dot(A,R)**2, axis = 1)
        lev = d*lev/np.sum(lev)
        p = lev/sum(lev)

        return lev, p
        

def compLevApprFro(A, k, r):
     
        n, d  = A.shape
        Pi = np.random.randn(d, r)
        print "in Fro, forming B!"
        B = np.dot(A, Pi)

        if r > n or r > d:
            print 'r should not be greater than min(n,d)'
            return None

        print "in Fro, computing QR of B!"
        Q, R  = np.linalg.qr(B)
        print "in Fro, computing SVD of QTA!"
        U, D, V = np.linalg.svd(np.dot(Q.T, A), full_matrices = 0)

        print "in Fro, computing C!"
        C = np.dot(Q, U[:,:k])
        lev = np.sum(C**2, axis = 1)
        p = lev/k

        return lev, p


def compLevApprSpe(A, k, q, r, reo=4):

        n, d = A.shape
        Pi = np.random.randn(d, 2*k);
        B = np.dot(A, Pi);
        print "in Spe, forming B!"
       
        for i in range(q):
            if i % reo == reo-1:
                Q, R = np.linalg.qr(B)
                print "reorthogonalzing!"
                B = Q 
            print "in Spe, q=%d, computing product!",i
            B = np.dot(A.T, B)
            print "in Spe, computing product again!"
            B = np.dot(A, B)

        #lev, p = compLevAppr(B, r)
        print "in Spe, computing SVD!"
        lev, p = compLevExact(B, k, 0)
        #lev, p = compLevApprFro(B, k, r)
        lev = p*k

        return lev, p 
        
    
def cur(A, k, c, r):
 
# This function computes the CUR decomposition of given matrix.
# Input:
#     A: n-by-d matrix.
#     k: a rank parameter.
#     c: number of columns to sample.
#     r: number of rows to sample.
# Output:
#     C,U,R: n-by-c, c-by-r, r-by-d matrices, respectively, such that ||A - CUR||_F is small.
#     lev_V, lev_U: leverage scores computed.

        # Computing leverage scores from A
        colLev = self.compLevExact(A, k, 1)
        p = colLev / k
        bins = np.add.accumulate(p)
        colInd = np.digitize(np.random.random_sample(c), bins)

        # Sampling columns from A
        C = np.dot(A[:,colInd], np.diag(1/p[colInd]))

        # Computing leverage scores from C
        rowLev = self.compLevExact(C, c, 0)
        p = rowLev / c
        bins = np.add.accumulate(p)
        rowInd = np.digitize(np.random.random_sample(r), bins)

        # Sampling rows from A
        R = np.dot(np.diag(1/p[rowInd]),A[rowInd,:])
        W = np.dot(np.diag(1/p[rowInd]),C[rowInd,:])
        U = np.linalg.pinv(W)

        return C, U, R, rowLev, colLev


if __name__ == "__main__":

    k = 10
    c = 150
    r = 500

    filename = 'Brain_Oct9'
    obj = omsiIO()
    obj.read2dData(filename)

    methods = curDecomp()
    A = obj.msiData2d
    #A = A.T

    print A.shape
 
    lev_exact, p_exact = methods.compLevExact(A, k, 0)
    #U, D, V = np.linalg.svd(A, full_matrices=False)
    #lev_appr, p_appr = methods.compLevAppr(A, 5000)
    #print lev_appr.shape
    #print p_exact[:50]
    #print p_appr[:50]
    #print np.mean(abs(p_exact - p_appr)/p_exact)
    #print np.linalg.norm( lev_exact - lev_appr) / np.linalg.norm(lev_exact)

    lev_appr_spe, p_appr_spe = methods.compLevApprSpe(A, k, 3, 1000)
    lev_appr_fro, p_appr_fro = methods.compLevApprFro(A, k, 400) 

    print np.linalg.norm( p_exact - p_appr_spe) / np.linalg.norm(p_exact)
    print p_exact.shape, p_appr_fro.shape
    print np.linalg.norm( p_exact - p_appr_fro) / np.linalg.norm(p_exact) 
    print np.sum(p_appr_spe), np.sum(p_appr_fro)

    #C, X, lev = methods.cx(A, k, c, 'exact') 
    #print np.linalg.norm(A - np.dot(C,X), 'fro') / np.linalg.norm(A, 'fro')

    #C, X, lev = methods.cx(A, k, c, 'apprSpe')
    #print np.linalg.norm(A - np.dot(C,X), 'fro') / np.linalg.norm(A, 'fro')

    #C, X, lev = methods.cx(A, k, c, 'apprFro')
    #print np.linalg.norm(A - np.dot(C,X), 'fro') / np.linalg.norm(A, 'fro')

    #baseline, errors, errors_full = methods.gridSearching(A, [5,10,15,20,25,30,40,50,60,70], [1,1.5,2,3,4,5],1)

    #print baseline, errors, errors_full

'''
    U, D, V = np.linalg.svd(A, full_matrices=False)
    Ak = np.dot( np.dot(U[:,:k], np.diag(D[:k])), V[:k,:])

    baseline = np.linalg.norm(A - Ak, 'fro')
    print baseline

    C, U, R, rowLev, colLev = methods.cur(A, k, c, r)
    tmp = np.linalg.norm(A - np.dot(np.dot(C,U),R), 'fro')
    print C.shape, U.shape, R.shape, tmp 
                                      
    C, U, R, rowLev, colLev = methods.cur_cx(A, k, c, r)
    tmp = np.linalg.norm(A - np.dot(np.dot(C,U),R), 'fro')
    print C.shape, U.shape, R.shape, tmp

    C, U, R, rowLev, colLev = methods.cur_cx_deter(A, k, c, r)
    tmp = np.linalg.norm(A - np.dot(np.dot(C,U),R), 'fro')
    print C.shape, U.shape, R.shape, tmp
'''


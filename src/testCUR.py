import numpy as np
import time
from curDecomp import *

def curcxGridSearching(A, round, kvec, oversampling_vec, deter):
    '''
    A: input matrix
    round: set it to 1 for CX, 2 for cur
    kvec: vector of values for k
    over_sampling_vec: vector storing oversampling factor; for each value v, the number of columns selected will be k*v
    deter: using deterministic CX or not
    '''

    numTrials = 20

    baseline = []
    if deter == 1:
        errors_full1 = np.zeros((len(kvec), len(oversampling_vec),1))
        errors_full2 = np.zeros((len(kvec), len(oversampling_vec),1))
    else:
        errors_full1 = np.zeros((len(kvec), len(oversampling_vec), numTrials))
        errors_full2 = np.zeros((len(kvec), len(oversampling_vec), numTrials))

    U, D, V = np.linalg.svd(A, full_matrices=False)

    i = 0
    for k in kvec:
        P = np.dot(U[:,:k],U[:,:k].T)
        Ak = np.dot(P,A)
        baseline.append(np.linalg.norm(A - Ak, 'fro')) # best rank-k approximation

        j = 0
        for v in oversampling_vec:
            c = k*v
            #print i, j
            for l in range(numTrials):
                if deter:
                    if round == 1:
                        C, X, colLev = cx_deter(A, k, c)
                        CX = np.dot(C,X)
                        errors_full1[i,j,l] = np.linalg.norm(A - CX, 'fro')
                        errors_full2[i,j,l] = np.linalg.norm(A - np.dot(C,np.dot(np.linalg.pinv(np.dot(P,C)),Ak)), 'fro') # this will give you a rank-k approximation
                    else:
                        C, U, R, rowLev, colLev = cur_cx_deter(A, k, c, c)
                        errors_full[i,j,l] = np.linalg.norm(A - np.dot(np.dot(C,U),R), 'fro')
                    break
                else:
                    if round == 1:
                        C, X, colLev = cx(A, k, c)
                        CX = np.dot(C,X)
                        errors_full1[i,j,l] = np.linalg.norm(A - CX, 'fro')
                        errors_full2[i,j,l] = np.linalg.norm(A - np.dot(C,np.dot(np.linalg.pinv(np.dot(P,C)),Ak)), 'fro')
                    else:
                        C, U, R, rowLev, colLev = cur_cx(A, k, c, c)
                        errors_full[i,j,l] = np.linalg.norm(A - np.dot(np.dot(C,U),R), 'fro')
            j += 1
        i += 1

    baseline = np.array(baseline).reshape(len(kvec))

    normA = np.linalg.norm(A, 'fro')
    recons_errors1 = np.mean(errors_full1,axis = 2) / normA
    recons_errors2 = np.mean(errors_full2,axis = 2) / normA
    recons_errors_Ak = baseline / normA

    recons_errors_std1 = np.std(errors_full1, axis = 2)/normA
    recons_errors_std2 = np.std(errors_full2, axis = 2)/normA

    return {'normA': normA, 'recons_errors1': recons_errors1, 'recons_errors2': recons_errors2, 'recons_errors_Ak': recons_errors_Ak, 'recons_errors_std1': recons_errors_std1, 'recons_errors_std2': recons_errors_std2}


def cxUnifSearching(A, kvec, cvec):

    numTrials = 20

    errors_unif_full1 = np.zeros((len(cvec),numTrials))
    errors_unif_full2 = np.zeros((len(kvec),len(cvec),numTrials))

    U, D, V = np.linalg.svd(A, full_matrices=False)

    for c,i in zip(cvec,range(len(cvec))):
        print i
        for l in range(numTrials):
          C, X = cx_unif(A, c)
          CX = np.dot(C,X)
          errors_unif_full1[i,l] = np.linalg.norm(A - CX, 'fro')
          for k,j in zip(kvec,range(len(kvec))):
            P = np.dot( U[:,:k], U[:,:k].T )
            errors_unif_full2[j,i,l] = np.linalg.norm(A - np.dot(C,np.dot(np.linalg.pinv(np.dot(P,C)),np.dot(P,A))), 'fro')
        
    normA = np.linalg.norm(A, 'fro')
    recons_errors_unif1 = np.mean(errors_unif_full1,axis=1) / normA
    recons_errors_unif2 = np.mean(errors_unif_full2,axis=2) / normA

    recons_errors_unif_std1 = np.std(errors_unif_full1,axis=1) / normA
    recons_errors_unif_std2 = np.std(errors_unif_full2,axis=2) / normA

    return {'recons_errors_unif1': recons_errors_unif1, 'recons_errors_unif2': recons_errors_unif2, 'recons_errors_unif_std1': recons_errors_unif_std1, 'recons_errors_unif_std2': recons_errors_unif_std2}


def apprLev(A, k, q_vec, r, c, large=0):

    numTrials = 20

    if large:
        print "Large file!"

        t_exact = 0
        cx_error_exact = 0

    else:
        normA = np.linalg.norm(A, 'fro')
        U, D, V = np.linalg.svd(A.T, full_matrices=False)

        P = np.dot(U[:,:k],U[:,:k].T)
        Ak = np.dot(P,A.T)


        t = time.time()
        lev_exact, p_exact = compLevExact(A, k, 0)
        t_exact = time.time() - t
        print "number of leverage scores: ", len(lev_exact)

        bins = np.add.accumulate(p_exact)
        cx_error_exact = np.zeros(numTrials)
        cx_error_exact2 = np.zeros(numTrials)
        for j in range(numTrials):
            colInd = np.digitize(np.random.random_sample(c), bins)
            C = np.dot(A.T[:,colInd], np.diag(1/p_exact[colInd]))
            X = np.dot(np.linalg.pinv(C), A.T)
            cx_error_exact[j] = np.linalg.norm(A.T - np.dot(C,X), 'fro')/normA
            print cx_error_exact[j]
            cx_error_exact2[j] = np.linalg.norm(A.T - np.dot(C,np.dot(np.linalg.pinv(np.dot(P,C)),Ak)), 'fro')
            

        #ind = np.where(p_exact>float(5)/len(p_exact))[0]
        #print "number of large leverage scores: ", len(ind)
        ind2 = np.where(p_exact>1e-5)[0]

    error_fro = np.zeros(numTrials)
    beta_fro = np.zeros(numTrials)

    error_spe = np.zeros((numTrials,len(q_vec)))
    beta_spe = np.zeros((numTrials,len(q_vec)))

    cx_error_fro = np.zeros(numTrials)
    cx_error_fro2 = np.zeros(numTrials)
    t_fro = np.zeros(numTrials)

    cx_error_spe = np.zeros((numTrials,len(q_vec)))
    cx_error_spe2 = np.zeros((numTrials,len(q_vec)))
    t_spe = np.zeros((numTrials,len(q_vec)))

    lev_fro = np.zeros((A.shape[0],numTrials))
    p_fro = np.zeros((A.shape[0],numTrials))

    lev_spe = np.zeros((A.shape[0],numTrials,len(q_vec)))
    p_spe = np.zeros((A.shape[0],numTrials,len(q_vec)))

    for j in range(numTrials):
            print j, 'fro'
            t = time.time()
            lev, p = compLevApprFro(A, k, r)
            lev_fro[:,j] = lev
            p_fro[:,j] = p
            t_fro[j] = time.time() - t
            if large == 0:
                beta_fro[j] = np.min(p/p_exact)
                error_fro[j] = np.mean(np.abs(p[ind2] - p_exact[ind2])/p_exact[ind2])

            if c > 0:
                bins = np.add.accumulate(p)
                colInd = np.digitize(np.random.random_sample(c), bins)
                C = np.dot(A.T[:,colInd], np.diag(1/p[colInd]))
                X = np.dot(np.linalg.pinv(C), A.T)
                cx_error_fro[j] = np.linalg.norm(A.T - np.dot(C,X), 'fro')/normA
                cx_error_fro2[j] = np.linalg.norm(A.T - np.dot(C,np.dot(np.linalg.pinv(np.dot(P,C)),Ak)), 'fro')

            for i, q in zip(range(len(q_vec)), q_vec):
                print j, i, 'spe'
                t = time.time()
                lev, p = compLevApprSpe(A, k, q, r)
                lev_spe[:,j,i] = lev
                p_spe[:,j,i] = p
                t_spe[j,i] = time.time() - t

                if large == 0:
                    beta_spe[j,i] = np.min(p/p_exact) 
                    error_spe[j,i] = np.mean(np.abs(p[ind2] - p_exact[ind2])/p_exact[ind2])

                if c > 0:
                    bins = np.add.accumulate(p)
                    colInd = np.digitize(np.random.random_sample(c), bins)
                    C = np.dot(A.T[:,colInd], np.diag(1/p[colInd]))
                    X = np.dot(np.linalg.pinv(C), A.T)
                    cx_error_spe[j,i] = np.linalg.norm(A.T - np.dot(C,X), 'fro')/normA
                    cx_error_spe2[j,i] = np.linalg.norm(A.T - np.dot(C,np.dot(np.linalg.pinv(np.dot(P,C)),Ak)), 'fro')
                    print cx_error_spe[j,i]

    return {'cx_error_exact': np.mean(cx_error_exact), 'cx_error_exact_std': np.std(cx_error_exact), 'cx_error_exact2': np.mean(cx_error_exact2), 'cx_error_exact_std2': np.std(cx_error_exact2), 't_exact': t_exact,\
                'beta_fro': np.mean(beta_fro), 'error_fro': np.mean(error_fro), 'error_fro_std': np.std(error_fro), 't_fro': np.mean(t_fro),\
                'cx_error_fro': np.mean(cx_error_fro), 'cx_error_fro_std': np.std(cx_error_fro), 'cx_error_fro2': np.mean(cx_error_fro2), 'cx_error_fro_std2': np.std(cx_error_fro2),\
                'beta_spe': np.mean(beta_spe, axis=0), 'error_spe': np.mean(error_spe, axis=0), 'error_spe_std': np.std(error_spe, axis=0), 't_spe': np.mean(t_spe, axis=0),\
                'cx_error_spe': np.mean(cx_error_spe, axis=0), 'cx_error_spe_std': np.std(cx_error_spe, axis=0), 'cx_error_spe2': np.mean(cx_error_spe2, axis=0), 'cx_error_spe_std2': np.std(cx_error_spe2, axis=0),\
                'lev_fro': lev_fro, 'p_fro': p_fro, 'lev_spe': lev_spe, 'p_spe': p_spe}


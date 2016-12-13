import numpy as np
import scipy.linalg
from scipy.linalg import cholesky, cho_solve, svd, LinAlgError
from sklearn.covariance import MinCovDet
import sys
sys.path.insert(0, 'pyrcca')
import rcca
from sklearn.decomposition import PCA

# Numerical constants / thresholds
CHOLTHRESH = 1e-5
def solve_posdef(A, b):
    """
    Solve the system :math:`A X = b` for :math:`X` where :math:`A` is a
    positive semi-definite matrix.
    This first tries cholesky, and if numerically unstable with solve using a
    truncated SVD (see solve_posdef_svd).
    The log-determinant of :math:`A` is also returned since it requires minimal
    overhead.
    Parameters
    ----------
    A: ndarray
        A positive semi-definite matrix.
    b: ndarray
        An array or matrix
    Returns
    -------
    X: ndarray
        The result of :math:`X = A^-1 b`
    logdet: float
        The log-determinant of :math:`A`
    """
    # Try cholesky for speed
    try:
        lower = False
        L = cholesky(A, lower=lower)
        if any(L.diagonal() < CHOLTHRESH):
            raise LinAlgError("Unstable cholesky factor detected")
        X = cho_solve((L, lower), b)
        logdet = cho_log_det(L)

    # Failed cholesky, use svd to do the inverse
    except LinAlgError:

        U, s, V = svd(A)
        X = svd_solve(U, s, V, b)
        logdet = svd_log_det(s)

    return X, logdet

def cho_log_det(L):
    """
    Compute the log of the determinant of :math:`A`, given its (upper or lower)
    Cholesky factorization :math:`LL^T`.
    Parameters
    ----------
    L: ndarray
        an upper or lower Cholesky factor
    Examples
    --------
    >>> A = np.array([[ 2, -1,  0],
    ...               [-1,  2, -1],
    ...               [ 0, -1,  2]])
    >>> Lt = cholesky(A)
    >>> np.isclose(cho_log_det(Lt), np.log(np.linalg.det(A)))
    True
    >>> L = cholesky(A, lower=True)
    >>> np.isclose(cho_log_det(L), np.log(np.linalg.det(A)))
    True
    """
    return 2 * np.sum(np.log(L.diagonal()))

class MaximumLikelihoodCCA(object):
    def __init__(self,n_components = 2):
        self.n_components = n_components

    def _qr_CCA(self):
        """ Performs CCA through QR decomposition"""
        q1,r1 = np.linalg.qr(self.X)
        q2,r2 = np.linalg.qr(self.Y)

        U,s,Vh = np.linalg.svd(np.dot(q1.T,q2))
        a = np.linalg.solve(r1,U[:,:self.n_components])
        b = np.linalg.solve(r2,Vh.T[:,:self.n_components])
        return a,s[:self.n_components],b
    
    def _CCA(self):
        cca = rcca.CCA(kernelcca = False, reg=0,numCC = self.n_components)
        cca.train([self.X,self.Y])
        return cca.ws[0],cca.cancorrs,cca.ws[1]

    def fit(self,X,Y):
        """ Fit the model to the data"""
        self.X = X
        self.Y = Y
        # we will create the sample covariance matrix
        z = np.concatenate((self.X.T,self.Y.T)) #combine the tranposed data.
        #z = np.concatenate((self.X,self.Y)) 
        #Each row represents a variable and column represents sample
        # X & Y must have the same number of samples
        C = np.cov(z)
            
        sx = self.X.shape[1] #find the dimensions of X and Y
        sy = self.Y.shape[1]
        self.n_samples = X.shape[0]
        #we partition the covariance matrix into the respective elements
        self.Cxx = C[0:sx,0:sx]
        self.Cxy = C[0:sx,sx:sx+sy]
        self.Cyx = self.Cxy.T
        self.Cyy = C[sx:,sx:]

        self.a,s,self.b = self._qr_CCA()
        #self.a,s,self.b = self._CCA()
        self.M1 = np.diag(np.sqrt(s))
        self.M2 = self.M1
        self.Pd = s

        #equations in theorem 2
        self.W1 = np.dot(np.dot(self.Cxx,self.a),self.M1)
        self.W2 = np.dot(np.dot(self.Cyy,self.b),self.M2)
        
        self.Phi1 = self.Cxx - np.dot(self.W1,self.W1.T)
        self.Phi2 = self.Cyy - np.dot(self.W2,self.W2.T)
        self.mu1 = np.mean(self.X,axis = 0)
        self.mu2 = np.mean(self.Y,axis = 0)

    def transform(self):
        self.E_z_x = np.dot(np.dot(self.X - self.mu1,self.a),self.M1)
        self.E_z_y = np.dot(np.dot(self.Y-self.mu2,self.b),self.M2)
        self.var_z_x = np.eye(self.n_components) - np.dot(self.M1,self.M1.T)
        self.var_z_y = np.eye(self.n_components) - np.dot(self.M2,self.M2.T)
        
        #equations for E[p(z|x1,x2)]
        M = np.concatenate((self.M1,self.M2),axis = 0)
        diff_Pd_inv = np.reciprocal(np.ones((self.n_components)) - np.square(self.Pd))
        diff_Pd_mul = diff_Pd_inv * self.Pd
        diff_Pd_inv = np.diag(diff_Pd_inv)
        diff_Pd_mul = np.diag(diff_Pd_mul)
        top_row = np.concatenate((diff_Pd_inv, diff_Pd_mul),axis = 1)
        bottom_row = np.concatenate((diff_Pd_mul, diff_Pd_inv),axis = 1)
        mid_matrix = np.concatenate((top_row,bottom_row),axis = 0) 

        self.E_z_xy = np.zeros((self.n_samples,self.n_components))
        for i in range(self.n_samples):
            U1d = np.dot(self.a.T, self.X[i,:]-self.mu1)
            U2d = np.dot(self.b.T, self.Y[i,:] - self.mu2)
            U_12d = np.reshape(np.concatenate((U1d,U2d),axis = 0),(2*self.n_components,1))
            self.E_z_xy[i,:] = np.dot(np.dot(M.T,mid_matrix),U_12d).T

        self.var_z_xy = np.eye(self.n_components) - np.dot(np.dot(M.T,mid_matrix),M).T

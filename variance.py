# -*- coding: utf-8 -*-
"""

@file      variance.py

@author    Walter Dehnen

@copyright Walter Dehnen (2022,23)

@license   GNU GENERAL PUBLIC LICENSE version 3.0
@version   0.1   Sep-2022 WD  created
@version   0.2   Oct-2022 WD  improved initialisation
@version   0.3   Feb-2023 WD  support for kappa-sigma clipping

"""
version = '0.3'

import numpy as np

class variance:
    """
    mean and co-variance for a sample of size data in ndim dimensions
    
    public data members
    -------------------
    size: int
        size of sample
    ndim: int
        number of dimensions of data points

    public modifying methods
    ------------------------
    scale()           effectuate a scaling of the data by factor(s)
    shift()           effectuate a shift of the data by offsets
    append()          add another sample's mean and variance

    public non-modifying methods
    ----------------------------
    mean()            sample mean
    var()             sample variance
    std()             sample stdandard deviation
    covar()           sample co-variance
    corr()            correlation coefficient
    var_of_mean()     unbiased variance of the mean
    std_of_mean()     unbiased standard deviation of the mean
    covar_of_mean()   unbiased co-variance of the mean
    propagate(f)      mean and variance for f(x)
    """

    def set_from_temp(self, tmp):
        """
        set mean and co-variance from data provided, scrambling the data
        """
        if tmp.ndim != 2:
            raise Exception("tmp not 2D iterable")
        ndim = tmp.shape[0]
        size = tmp.shape[1]
        mean = np.mean(tmp,axis=1)
        cov  = np.empty((ndim,ndim))
        for i in range(ndim):
            tmp[i,:]-= mean[i]
            cov[i,i] = np.mean(tmp[i,:]*tmp[i,:])
            for j in range(i):
                cov[i,j] = np.mean(tmp[i,:]*tmp[j,:])
                cov[j,i] = cov[i,j]
        self.size = size
        self.ndim = mean.shape[0]
        self.__Mean = mean
        self.__Cov  = cov

    @staticmethod
    def make_temp(data):
        tmp = np.array(data,dtype=np.float64,order='C')
        if tmp.ndim == 1:
            tmp = np.expand_dims(tmp,axis=0)
        if tmp.ndim != 2:
            raise Exception("data not 1D or 2D iterable")
        return tmp

    def set_from_data(self, data):
        """
        set mean and co-variance from data provided, which must be 2D
        """
        self.set_from_temp(self.make_temp(data))

    def __init__(self, data, numb_zero=0, size=None, mean=None, cov=None):
        """
        set mean and co-variance from data provided

        Parameters
        ----------
        data: 1D or 2D iterable over floats
            data vector(s)
        numb_zero: int
            complement the data with this many zero entries
        """
        if not data is None:
            self.set_from_data(data)
        else:
            self.size = size
            self.ndim = mean.shape[0]
            self.__Mean = mean
            self.__Cov  = cov
        self.append_zero(numb_zero)

    def __outer(self,vec):
        return np.outer(vec,vec)

    def __stdOstd(self):
        return self.__outer(np.sqrt(np.diagonal(self.__Cov)))

    def __debias(self, var, bias):
        if not type(bias) is bool:
            raise TypeError("bias must be bool")
        return var   if bias or self.size < 2 else \
               var * (self.size / (self.size - 1))

    def append(self, other):
        """append another set of data"""
        if not type(other) is variance:
            raise Exception("other not a variance")
        if other.size == 0:
            return
        if self.size == 0:
            self.ndim = other.ndim
            self.size = other.size
            self.__Mean = np.copy(other.__Mean)
            self.__Cov = np.copy(other.__Cov)
            return
        if other.ndim != self.ndim:
            raise Exception("ndim mismatch")
        size = self.size + other.size
        facS = self.size / size
        facO = 1 - facS
        Mean = facS * self.__Mean + facO * other.__Mean
        Cov  = facS * (self.__Cov  + self.__outer(Mean-self.__Mean)) \
             + facO * (other.__Cov + self.__outer(Mean-other.__Mean))
        self.size += size
        self.__Mean = Mean
        self.__Cov = Cov

    def append_zero(self, numb):
        """append a data set of numb zero values"""
        if not type(numb) is int:
            raise Exception("numb not integer")
        if numb < 0:
            raise Exception("numb="+str(numb)+"<0")
        if numb == 0:
            return
        size = self.size + numb
        fact = self.size / size
        Mean = fact * self.__Mean
        Cov  = fact * self.__Cov + fact*(1-fact)*self.__outer(self.__Mean)
        self.size = size
        self.__Mean = Mean
        self.__Cov = Cov

    def scale(self, fac):
        """effectuate scaling of data by factor(s)"""
        self.__Mean *= fac
        self.__Cov  *= fac*fac if type(fac) is float else np.outer(fac,fac)

    def shift(self, offset):
        """effectuate shifting of data by offsets"""
        self.__Mean += offset

    def clone(self):
        mc = variance(data=None, ndim=self.ndim)
        mc.size = self.size
        mc.__Mean = np.copy(self.__Mean)
        mc.__Cov = np.copy(self.__Cov)
        return mc        

    def mean(self, i=None):
        """sample mean (of component i or all)"""
        return self.__Mean[0]  if self.ndim==1 else \
               self.__Mean     if i is None    else \
               self.__Mean[i]

    def sum(self, i=None):
        """sum (of component i or all)"""
        return self.size * self.mean(i)

    def var(self, i=None, bias=True):
        """sample variance (of component i or all, see also covar)
        
        parameter bias indicates whether the variance is computed as 1/size
        (biased) or 1/(size-1) (unbiased) times the sum over squared deviations
        from mean
        """
        return self.__debias(self.__Cov[0,0]         if self.ndim==1 else \
                             np.diagonal(self.__Cov) if i is None    else \
                             self.__Cov[i,i], bias)

    def std(self, i=None, bias=True):
        """sample stdandard deviation (of component i or all)
        
        parameter bias indicates whether the variance is computed as 1/size
        (biased) or 1/(size-1) (unbiased) times the sum over squared deviations
        from mean
        """
        return np.sqrt(self.var(i,bias))

    def covar(self, i=None, j=None, bias=True):
        """sample co-variance (between components i and j or full matrix)
        
        parameter bias indicates whether the variance is computed as 1/size
        (biased) or 1/(size-1) (unbiased) times the sum over squared deviations
        from mean
        """
        return self.__debias(self.__Cov[0,0]  if self.ndim==1 else \
                             self.__Cov       if i is None    else \
                             self.__Cov[i,i]  if j is None    else \
                             self.__Cov[i,j], bias)

    def corr(self, i=None, j=None):
        """correlation (between components i and j or all)"""
        return 1.0                            if self.ndim==1 else \
               self.__Cov / self.__stdOstd()  if i is None else \
               1.0                            if j is None or i==j else \
               self.covar(i,j) / np.sqrt(self.__Cov[i,i]*self.__Cov[j,j])

    def var_of_mean(self, i=None):
        """variance of the mean (in component i or in all)"""
        return self.var(i) / (self.size-1)

    def std_of_mean(self, i=None):
        """stdandard deviation of the mean (in component i or all)
           unbiased measure for the statistical uncertainty of the mean
        """
        return np.sqrt(self.var_of_mean(i))

    def covar_of_mean(self, i=None,j=None):
        """co-variance of mean (between components i and j or all)"""
        return self.covar(i,j) / (self.size-1)

    def propagate(self, func, args=()):
        """
        linear error propagation of the mean

        Parameters:
        -----------
        func: function f(x) taking array of ndim arguments and returns an
            iterable of kdim values and their derivatives (the Jacobian, a kdim
            x ndim matrix). For kdim=1, the Jacobian can be a 1D numpy.ndarray.
        args: list
            further arguments passed to func(x,args)

        Returns: f(mean) with var computed via linear error propagation 

        for linear functions, this is equivalent to variance(f(data)).
        """
        f, J = func(self.mean(),*args)
        kdim = len(f)
        if 2 != J.ndim:
            raise RuntimeError("Jacobian must be a 2D array (matrix)")
        if J.shape != (kdim,self.ndim):
            raise RuntimeError("Jacobian must be a "+str(kdim)+"x"+ \
                               str(self.ndim)+" matrix, but got a "+ \
                               str(J.shape[0])+"x"+str(J.shape[1])+" matrix")
        return variance(data=None, size=self.size, mean=f, \
                        cov=np.matmul(np.matmul(J,self.__Cov),J.transpose()))

    def mean_and_std(self, func, args=()):
        """
        given a variance object, obtain y=f(<x>) and its standard deviation
        
        Parameters:
        -----------
        var: variance object
             representing sample mean and co-variance
        func: function f(x) taking ndim arguments and returns an iterable of
             kdim values and their derivatives (the Jacobian, a kdim x ndim
             matrix). For kdim=1, the Jacobian can be a 1D numpy.ndarray.
        args: list
             further arguments passed to func(x,args)

        Returns: f(mean) and its standard deviation computed via linear
             error propagation 
        """
        mc = self.propagate(func,*args)
        return mc.mean(), mc.std_of_mean()

    def kappa_square(self, data, debug=0):
        """
        compute (x-μ).invC.(x-μ) where invC is the inverse of the co-variance

        Parameters:
        -----------
        data: numpy array with shape=(ndim, len) or (len) if ndim=1
            data vector(s)

        Returns: array
            (x-μ).invC.(x-μ)
        """
        dat = np.array(data,dtype=np.float64,order='C')
        if dat.ndim == 1:
            dat = np.expand_dims(dat,axis=0)
        if dat.ndim != 2:
            raise Exception("data not 1D or 2D iterable")
        for i in range(self.ndim):
            dat[i,:]-= self.__Mean[i]
        iC = np.linalg.inv(self.__Cov)
        kp = np.einsum('il,li->i',np.einsum('ki,kl->il',dat,iC),dat)
        if debug > 0:
            x = np.array(data,dtype=np.float64,order='C')
            if x.ndim == 1:
                x = np.expand_dims(x,axis=0)
            print('μ =',self.__Mean)
            print('C =',self.__Cov,' invC =',iC)
            for i in range(dat.shape[1]):
                print('x =',x[:,i],'  x-μ =',dat[:,i],\
                      '  (x-μ).invC.(x-μ)=',kp[i])
        return kp

def mean_and_variance(data, kappa=None, maxiter=20):
    """
    obtain variance object, possibly applying κ-σ clipping

    Parameters:
    -----------
    data: 1D or 2D iterable over floats
        data vector(s)
    kappa: float or None
        if not None, apply κ-σ clipping
    maxiter: int
        maximum # iterations for κ-σ clipping

    throws if kappa ≤ 2, maxiter ≤ 0, iterations exceed maxiter, or
           κ-σ clipping removes all data
    """
    var = variance(data)
    if not kappa is None:
        if kappa <= 2:
            raise ValueError('κ =',kappa,'seems too small')
        if maxiter <= 0:
            raise ValueError('maxiter =',maxiter)
        itr = maxiter
        dat = variance.make_temp(data)
        num = len(dat[0])
        while itr > 0 and num > 0:
            itr -= 1
            kapq = var.kappa_square(dat)
            keep = kapq <= kappa*kappa
            dat  = dat[:,keep]
            var.set_from_data(dat)
            if num == len(dat[0]):
                break
            num  = len(dat[0])
        if itr == 0:
            raise RuntimeError('exceeding',maxiter,'iterations in κ-σ clipping')
        if num == 0:
            raise RuntimeError('removing all data in κ-σ clipping')
    return var

def propagate(val, cov, func, args=()):
    """
    linear error propagation of value with covariance matrix

    Parameters:
    -----------
    val  1D array
        n-dimensional value
    cov:  2D array
        n x n co-variance matrix for val
    func: function f(x) taking array of ndim arguments and returns an
        iterable of k values and their derivatives (the Jacobian, a kxn
        matrix). For k=1, the Jacobian can be a 1D numpy.ndarray.
    args: list
        further arguments passed to func(x,args)

    Returns: y = f(val) and co-variance matrix of y computed via linear
        error propagation
    """
    if 1 != val.ndim:
        raise TypeError("val must be a 1D array")
    n = val.shape[0]
    if 2 != cov.ndim:
        raise TypeError("cov must be a 2D array (matrix)")
    if cov.shape != (n,n):
        raise TypeError("cov must be a "+str(n)+'x'+str(n)+" matrix)")
    y, J = func(val,*args)
    if 1 != y.ndim:
        raise RuntimeError("func must return a 1D array")
    k = y.shape[0]
    if k > 1:
        if 2 != J.ndim:
            raise RuntimeError("Jacobian must be a 2D array (matrix)")
        if J.shape != (k,n):
            raise RuntimeError("Jacobian must be a "+str(k)+"x"+ \
                               str(n)+" matrix, but got a "+ \
                               str(J.shape[0])+"x"+str(J.shape[1])+" matrix")
    return y, np.matmul(np.matmul(J,cov),J.transpose())

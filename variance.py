# -*- coding: utf-8 -*-
"""

@file      variance.py

@author    Walter Dehnen

@copyright Walter Dehnen (2022)

@license   GNU GENERAL PUBLIC LICENSE version 3.0
@version   0.1   Sep-2022 WD  created

"""
version = '0.1'

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

    def __init__(self, data, ndim=1):
        """
        set mean and co-variance from data provided

        Parameters
        ----------
        data: iterable over floats or over 1D numpy arrays (of same length)
            data vector(s)
        """
        if data is None:
            if ndim < 1:
                raise Exception("ndim="+str(ndim)+" < 1")
            self.ndim = ndim
            self.size = 0
            self.__Mean = np.zeros(self.ndim)
            self.__Cov = np.zeros((self.ndim,self.ndim))
        else:
            self.ndim = len(data)
            if type(data[0]) is float:
                self.size = 1
                self.__Mean[i] = np.array(data)
                self.__Cov = np.zeros((self.ndim,self.ndim))
            else:
                self.size = len(data[0])
                self.__Mean = np.empty(self.ndim)
                self.__Cov = np.empty((self.ndim,self.ndim))
                deltaY = np.empty((self.ndim,self.size))
                for i in range(self.ndim):
                    if len(data[i]) != self.size:
                        raise RuntimeError("found "+str(self.size)+\
                                           " data in component 0 but "+\
                                           str(len(data[i]))+\
                                           " in component "+str(i))
                    self.__Mean[i] = np.mean(data[i])
                    deltaY[i,:] = data[i] - self.__Mean[i]
                    self.__Cov[i,i] = np.mean(deltaY[i,:]*deltaY[i,:])
                    for j in range(i):
                        self.__Cov[i,j] = np.mean(deltaY[i,:]*deltaY[j,:])
                        self.__Cov[j,i] = self.__Cov[i,j]

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
        if self.size == 0:
            self.ndim = other.ndim
            self.size = other.size
            self.__Mean = np.copy(other.__Mean)
            self.__Cov = np.copy(other.__Cov)
            return
        if other.size == 0:
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

    def scale(self, fac):
        """effectuate scaling of data by factor(s)"""
        self.__Mean *= fac
        self.__Cov  *= fac if type(fac) is float else np.outer(fac,fac)

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
        func: function f(x) taking ndim arguments and returns an iterable of
            kdim values and their derivatives (the Jacobian, a kdim x ndim
            matrix). For kdim=1, the Jacobian can be a 1D numpy.ndarray.
        args: list
            further arguments passed to func(x,args)

        Returns: f(mean) with var computed via linear error propagation 

        for linear functions, this is equivalent to variance(f(data)).
        """
        f, J = func(self.mean(),*args)
        kdim = len(f)
        if kdim != len(J):
            raise RuntimeError("func returns "+str(kdim)+" values, but "+\
                               len(J)+" derivatives")
        mc = variance(data=None, ndim=kdim)
        mc.size = self.size
        mc.__Mean = f
        mc.__Cov = np.matmul(np.matmul(J,self.__Cov),J.transpose())
        return mc

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

def mean_and_variance(data):
    return variance(data)

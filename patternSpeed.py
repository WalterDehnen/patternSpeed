# -*- coding: utf-8 -*-
"""

@file      patternSpeed.py

@author    Walter Dehnen, Marcin Semczuk

@copyright Walter Dehnen, Marcin Semczuk (2022,23)

@license   GNU GENERAL PUBLIC LICENSE version 3.0
           see file LICENSE for details

@version   0.1   jun-2022 MS  initial code
@version   0.2   sep-2022 WD  patternSpeed.py
@version   0.3   feb-2023 WD  general m; avoid empty last bin in createBins()
@version   0.4   may-2023 WD  analyse m=1...maxm, using pandas, better bar find

"""
version = '0.4'

import numpy as np
import pandas as pd
import warnings
from variance import variance

debug = 0

def window(Q):
    """compute W(Q) = 2(1-Q)²(1+2Q)"""
    Q1 = 1-Q
    return 2*Q1*Q1*(1+Q+Q)

def windowDeriv(Q):
    """compute W(Q) = 2(1-Q)²(1+2Q) and dW/dQ = -12 Q(1-Q)"""
    Q1 = 1-Q
    return 2*Q1*Q1*(1+Q+Q), -12*Q*Q1

def atan(sin,cos):
    """arctan(sin/cos) in the range [0,2π]"""
    psi = np.arctan2(sin,cos)
    return np.where(psi<0, psi+2*np.pi, psi)

def asfarray(array, copy=False, checkFinite=False):
    """obtain a new float array from input array"""
    arr = np.asfarray(array)             # ensure float data type
    if checkFinite:
        arr = np.asarray_chkfinite(arr)  # check for NaN or inf
    if copy and arr is array:
        arr = np.array(arr,copy=True)    # ensure we make a copy
    return arr

class harmonic:
    """ provide cosmφ, sinmφ at incrementable m
        attributes:
            m       harmonic order
            Cm,Sm   cosmφ, sinmφ    
    """
    
    def increment(self, dm=1):
        """ increment m to m+dm """
        for k in range(dm):
            self.Cm,self.Sm = self.Cm*self.C1-self.Sm*self.S1,\
                              self.Sm*self.C1+self.Cm*self.S1
            self.m += 1        

    def set_order(self, m):
        """ set m to the value given (must be ≥ m)"""
        if m < self.m:
            raise ValueError("m="+str(m)+" < current m="+str(self.m))
        else:
            self.increment(m-self.m)
        
    def __init__(self, C, S, m=0):
        """ construct from cosφ, sinφ """
        self.C1 = C
        self.S1 = S
        if m == 0:
            self.Cm = np.ones (C.shape)
            self.Sm = np.zeros(S.shape)
            self.m = 0
        else:
            self.m  = 1
            self.Cm = C
            self.Sm = S
            self.set_order(m)
            
def convertFourier(x, m):
    """ given   x={<1>, <cosmφ>, <sinmφ>,
                    [<∂cosmφ/∂t>, <∂sinmφ/∂t>],
                    [<∂cosmφ/∂X>, <∂sinmφ/∂X>] } and m
        compute y={Am=Σm/Σ0, ψm, [∂ψm/∂t], [∂ψm/∂X] } 
    """
    K = (len(x)+1)//2
    y = np.zeros((K))
    J = np.zeros((K,len(x)))
    Sigm    = np.hypot(x[1],x[2])           # Σm  = √(Cm²+Sm²)
    iSigm   = np.reciprocal(Sigm)           # 1/Σm
    iSig0   = np.reciprocal(x[0])           # 1/Σ0
    y[0]    = Sigm * iSig0                  # Am = Σm/Σ0
    J[0,0]  =-y[0] * iSig0                  # ∂Am/∂C0 = -Σm/Σ0² = -Am/Σ0
    J[0,1]  = x[1] * iSigm * iSig0          # ∂Am/∂Cm = Cm/Σm/Σ0
    J[0,2]  = x[2] * iSigm * iSig0          # ∂Am/∂Sm = Sm/Σm/Σ0
    im      = 1/m
    iQ      = iSigm * iSigm
    y[1]    = im*atan(x[2],x[1])            # ψ     = 1/m atan(S/C)
    J[1,1]  =-im*x[2]*iQ                    # ∂ψ/∂C =-S/Σm²/m
    J[1,2]  = im*x[1]*iQ                    # ∂ψ/∂S = C/Σm²/m
    for k in range(2,K):
        k2 = k+k
        k1 = k2-1
        y[k]    = J[1,1]*x[k1] + J[1,2]*x[k2]    # ψ'      = C' ∂ψ/∂C + S' ∂ψ/∂S
        J[k,k1] = J[1,1]                         # ∂ψ'/∂C' = ∂ψ/∂C
        J[k,k2] = J[1,2]                         # ∂ψ'/∂S' = ∂ψ/∂S
        J[k,1]  =( im*x[k2]-2*x[1]*y[k])*iQ      # ∂ψ'/∂C  = ( S'/m - 2Cψ')/Σm²
        J[k,2]  =(-im*x[k1]-2*x[2]*y[k])*iQ      # ∂ψ'/∂S  = (-C'/m - 2Sψ')/Σm²
    return y,J

def convertAlpha(x):
    """ given x={...,q} compute y={...,atan(q)} and Jacobian """
    k = len(x)-1
    y = np.copy(x)
    J = np.zeros((k+1,k+1))
    for i in range(0,k):
        J[i,i] = 1.0
    J[k,k] = 1.0/(1.0+y[k]*y[k])           # ∂α/∂Q = 1/(1+Q²)
    y[k]   = np.arctan(y[k])               # α = atan(Q)
    return y,J

class FourierMethod:
    """
    Fourier method for measuring pattern speed, as described by
    Dehnen et al. (2023, MNRAS, 518, 2712) 

    Methods
    -------
    initialisation (__init__)
        prepares the data by sorting them in radius
    analyseRegion()
        Fourier analysis in a radial range
    createBins()
        create radial bins, auxiliary for analyseDisc()
    analyseDisc()
        Fourier analysis in radial bins
    findBarRegion()
        takes output from analyseDisc() and identifies bar region
    findBar()
        combines analyseDisc() and findBarRegion()
    measureOmega()
        measures phase and pattern speed for a radial range
    patternSpeed()
        static method combining initialisation, findBar(), and measureOmega()

    Measuring bar pattern speed
    ---------------------------
    The measurement of the bar pattern speed takes three steps:
    0  preparing data tables sorted in radius R (at initialisation)
    1  finding the bar region
    2  measuring {ψ,Ω} for the bar region
    Users may skip step 1 if they want to measure {ψ,Ω} for another radial
    range of particles than that detected in step 1. See below for details.

    Step 1
    ------
    Finding the bar region is done in 2 steps:
    1.1  analyseDisc()     Fourier analysis radial bins
    1.2  findBarRegion()   identifies the bar region as range of bins
    This results in a pair of indices (i0,i1) into the sorted arrays.

    Step 2
    ------
    Measuring ψ and Ω for the bar region takes only one call to
    measureOmega()  computes {ψ,Ω} and requires (i0,i1) or (R0,R1) as input

    Skipping step 1
    ---------------
    The input for measureOmega() can be any pair of valid indices, specifying
    a radial range, or a pair of radii (R0,R1). In order to move between index
    and radius, we provide the member methods
        radius(i)   returns radius at given index i into the sorted tables
        indexR(R)   returns i such that radius(i) ≤ R < radius(i+1)
        numPart()   returns the size of the sorted tables

    A note on bias
    --------------
    Our method in step 2 is unbiased and consistent, i.e. satisfies ψ=∫Ω dt,
    as long as
    (1) the radial bar region (R0,R1) does not change; and
    (2) any particle selection inside the bar region is not based on
        evolving particle properties (position,velocity)

    see §2.3 of Dehnen et al. (2023). Strictly, our method in step 1 violates
    condition (1), but we found any bias to be insignificant. Violations of
    condition (2) are avoided when selecting particles on conserved quantities,
    such as stellar birth properties (time, location, metallicity) or a random
    sub-sample (selecting on id).

    """
    
    class Default:
        """ hold default parameters used in methods """
        maxm=6
        minNBin=4000
        maxNBin=32000
        maxDexBin=0.1
        minBarStrength=0.05
        minMaxBarStrength=0.2
        maxDPsi=10.0
        minDexBar=0.2
        minNumBar=100000
        
    def __init__(self, x,y, vx,vy, mu=1.0, checkFinite=False):
        """step 0  create sorted data arrays

        Input Data:
        -----------
        x : array-like
            centred x positions
        y : array-like
            centred y positions
        vx : array-like
            centred x velocities
        vy : array-like
            centred y velocities
        mu : scalar, or array-like
            particle mass(es)

        When providing only a sub-set or cut-out of simulation particles,
        care must be exercised to avoid a selection bias (see note on bias
        above). In particular, any positional cut must not be close to the
        bar.

        Parameters:
        -----------
        checkFinite: bool
            check input data for NaN or Inf

        """
        # 0   get input data to numpy arrays
        X = asfarray(x,checkFinite=checkFinite)
        Y = asfarray(y,checkFinite=checkFinite)
        Vx = asfarray(vx,checkFinite=checkFinite)
        Vy = asfarray(vy,checkFinite=checkFinite)
        # 1   compute derived data and sort them ascendingly in R²
        Rq = X*X + Y*Y
        dRq = (X*Vx+Y*Vy)*2
        dPh = (X*Vy-Y*Vx)/Rq
        i = np.argsort(Rq)
        if type(mu) is float:
            self.M = np.full(X.shape,mu)
        else:
            self.M = asfarray(mu,checkFinite=checkFinite)[i]
        self.Rq = Rq[i]
        self.dRq = dRq[i]
        self.dPh = dPh[i]
        R = np.sqrt(self.Rq)
        self.Cph = X[i]/R
        self.Sph = Y[i]/R
        self.debug = 0
        
    def radius(self,i):
        """radius for given index (or indices) into sorted tables"""
        return np.sqrt(self.Rq[i])
    
    def indexR(self,R):
        """index i such that radius(i) ≤ R < radius(i+1) or 0 or N"""
        return min(len(self.Rq),max(0,np.searchsorted(self.Rq, R*R)))

    def numPart(self):
        """number of particles loaded"""
        return len(self.Rq)

    def unpackRegion(self, region):
        """
        unpack region into two integers i0 < i1 in range [0,nP)

        Parameter:
        region: tuple of int, tuple or floats, or 'all'
            specifying region to analyse: in case of ints: indices into sorted
            arrays, in case of floats: radii, or simply all particles
        """
        nP = len(self.Rq)
        if type(region) is str and region == 'all':
            return 0,nP
        i0 = region[0]
        i1 = region[1]
        if type(i0) is float:
            i0 = self.indexR(i0)
        if type(i1) is float:
            i1 = self.indexR(i1)
        if i1 <= i0:
            raise RuntimeError("i1="+str(i1)+" ≤ i0="+str(i0))
        if i0 < 0:
            warnings.warn("i0="+str(i0)+" < 0: will take i0=0 instead")
            i0 = 0
        if nP < i1:
            warnings.warn("i1="+str(i1)+" > N="+str(nP)+
                          ": will take i1="+str(nP)+" instead")
            i1 = nP
        return i0,i1
    
    def analyseAux(self,H,dPh,iD,mW,mdWt=None,mdWx=None,correlation=False):
        """
        auxiliary for analyseRegion() and measureOmega()
        compute A,ψ[,Ω][,α] and their uncertainties for a set of particles
        """
        lst = [mW]
        lst.append(mW*H.Cm)                          # μ cosmφ
        lst.append(mW*H.Sm)                          # μ sinmφ
        inp = ['c0','cm','sm']
        prp = ['A','ps']
        if not mdWt is None:
            lst.append(mdWt*H.Cm - H.m*mW*H.Sm*dPh)  # μ d(W cosmφ)/dt
            lst.append(mdWt*H.Sm + H.m*mW*H.Cm*dPh)  # μ d(W sinmφ)/dt
            prp.append('Om')
            inp.append('dcm/dt')
            inp.append('dsm/dt')
        if not mdWx is None:
            lst.append(mdWx*H.Cm)                    # μ d(W cosmφ)/dlnR
            lst.append(mdWx*H.Sm)                    # μ d(W sinmφ)/dlnR
            prp.append('al')
            inp.append('dcm/dX')
            inp.append('dsm/dX')
        var = variance(lst)
        var.scale(dPh.size*iD/np.pi)
        var = var.propagate(convertFourier, args=(H.m,))
        if not mdWx is None:
            var = var.propagate(convertAlpha)
        prp = [p + str(H.m) for p in prp]
        val = pd.Series(data=var.mean(),index=prp)
        err = pd.Series(data=var.std_of_mean(),index=[p + '_e' for p in prp])
        ans = pd.concat((val,err))
        if correlation:
            return ans,var.corr()
        return ans
    
    def analyseRegion(self, region, maxm=Default.maxm,
                      computeOmega=False, computeAlpha=False, tophat=False):
        """
        compute A,ψ[,Ω][,α] and their uncertainties for particles in region

        Parameters:
        -----------
        region: tuple of int, tuple of floats, or 'all'
            specifying region to analyse: in case of ints: indices into sorted
            arrays, in case of floats: radii, or simply all particles
        maxm: int
            maximum azimuthal wavenumber m
            Default: Default.maxm
        computeOmega: bool
            compute pattern speeds Ω
            Default: False
        computeAlpha: bool
            compute angle α=atan(∂ψ/∂lnR)
            Default: False
        tophat: bool
            use a tophat window. A tophat window causes biased pattern
            speeds Ωm and cannot be used for computing the angles αm.
            Default: False
        """
        if maxm < 2:
            raise Exception("maxm =",maxm,"< 2")
        if tophat is None:
            tophat = not (computeOmega or computeAlpha)
        if computeAlpha and tophat:
            raise Exception("cannot compute alpha using tophat")
        if computeOmega and tophat:
            text = "measuring a pattern speed with a top-hat "+\
                   "window gives biased results"
            warnings.warn(text)
        i0,i1 = self.unpackRegion(region)
        nP = len(self.Rq)
        nB = i1-i0
        H = harmonic(self.Cph[i0:i1],self.Sph[i0:i1],m=0)
        Rq0 = 0.5*(  self.Rq[i0]   + self.Rq[i0-1])  if i0 > 0  else 0
        R0  = np.sqrt(Rq0)
        Rq1 = 0.5*(  self.Rq[i1-1] + self.Rq[i1]  )  if i1 < nP else \
              0.5*(3*self.Rq[i1-1] + self.Rq[i1-2])
        R1  = np.sqrt(Rq1)
        iD  = 1.0/(Rq1-Rq0)
        im  = (i1+i0)//2
        Rqm = self.Rq[im] if (nB%2)==0 else 0.5 * (self.Rq[im] + self.Rq[im+1])
        Rm  = np.sqrt(Rqm)
        Q   = self.Rq[i0:i1]-Rqm                    # R²-Rm²
        fac = np.where(Q<0,1/(Rq0-Rqm),1/(Rq1-Rqm)) # 1/(Re²-Rm²)
        Q  *= fac                                   # Q=(R²-Rm²)/(Re²-Rm²)
        if tophat:
            W,dW = 1.0,0.0
        else:
            W,dW = windowDeriv(Q)                   # W(Q), dW/dQ
        dW  = dW*fac                                # dW/dR²
        mu  = self.M[i0:i1]
        mW  = mu*W                                  # μ W
        mdWt= mu*dW*self.dRq[i0:i1] if computeOmega else None    # μ dW/dt
        mdWx= mu*dW*(-2*Rm*Rm)      if computeAlpha else None    # μ dW/dlnR
        sR = pd.Series(data=(i0,i1,Rm), index=('i0','i1','R'))
        v  = variance((nB*iD*mW/np.pi))
        s0 = pd.Series(data=(v.mean(0),v.std_of_mean(0)), index=('Sig','Sig_e'))
        ls = [sR,s0]
        for m in range(1,maxm+1):
            H.increment()
            ls.append(self.analyseAux(H,self.dPh[i0:i1],iD,mW,mdWt,mdWx))
        return pd.concat(ls)
    
    def createBins(self,
                   minNBin=Default.minNBin,
                   maxNBin=Default.maxNBin,
                   maxDexBin=Default.maxDexBin):
        """
        create radial bins

        Input data:
        -----------
        Rq: array like
            sorted table of radius-squared in ascending order

        Parameters:
        -----------
        minNbin: int
            minimum number of particles in radial bin
            Default: Default.minNBin
        maxNbin: int
            maximum number of particles in radial bin
            Default: Default.maxNbin
        maxDexBin: float
            maximum size of radial bin in log10(R)
            Default: Default.maxDexBin

        Returns:
        --------
        bins: list of tuples(i0,i1)
            i0,i1 = start and end indices into the sorted data arrays
        """
        # 0 sanity checks on parameters
        nP = len(self.Rq)
        if not type(minNBin) is int:
            raise Exception("minNBin must be int")
        if minNBin <= 100:
            raise Exception("minNBin="+str(minNBin)+" is too small")
        if minNBin > nP:
            raise Exception("minNBin="+str(minNBin)+" > numPart="+str(nP))
        if not type(maxNBin) is int:
            raise Exception("maxNBin must be int")
        if minNBin > maxNBin:
            raise Exception("maxNBin="+str(maxNBin)+" < minNBin="+str(minNBin))
        if not (type(maxDexBin) is float or type(maxDexBin) is int):
            raise Exception("maxDexBin must be scalar")
        if maxDexBin <= 0.0:
            raise Exception("maxDexBin="+str(maxDexBin)+" ≤ 0")
        if maxDexBin > 0.2:
            raise Exception("maxDexBin="+str(maxDexBin)+" is too large")
        maxRqFac = 10**(2*maxDexBin)
        # 1 create primary bins
        wl = nP - minNBin
        b1 = []
        i0 = 0
        i1 = minNBin
        while i1 < wl:
            Rqm = maxRqFac * self.Rq[i0]
            im  = min(nP, i0 + maxNBin)
            i1 += np.searchsorted(self.Rq[i1:im], Rqm)
            b1.append((i0,i1))
            i0 = i1
            i1 = i0 + minNBin
        if i0 < nP:
            b1.append((i0,nP))
        # 2 create intermittent bins
        i0 = (b1[0][0] + b1[0][1])//2
        bins = [b1[0]]
        for b in b1[1:]:
            i1 = (b[0] + b[1])//2
            bins.append((i0,i1))
            bins.append(b)
            i0 = i1
        return bins

    def analyseDisc(self, maxm=Default.maxm,
                    computeOmega=False,computeAlpha=False, tophat=None,
                    minNBin=Default.minNBin,
                    maxNBin=Default.maxNBin,
                    maxDexBin=Default.maxDexBin):
        """
        create radial bins and analyse each to find Σ and Am,ψm[,Ωm][,αm]
        for m=1...maxm, including statistical uncertainties

        Parameters:
        -----------
        maxm: int
            maximum azimuthal wavenumber m
            Default: Default.maxm
        computeOmega: bool
            compute pattern speeds Ωm
            Default: False
        computeAlpha: bool
            compute angles αm=atan(∂ψm/∂lnR)
            Default: False
        tophat: bool
            use a tophat window. A tophat window causes biased pattern
            speeds Ωm and cannot be used for computing the angles αm.
            Default: not (computeOmega or computeAlpha)
        minNbin: int
            minimum number of particles in radial bin
            Default: Default.minNBin
        maxNbin: int
            maximum number of particles in radial bin
            Default: Default.maxNbin
        maxDexBin: float
            maximum size of radial bin in log10(R)
            Default: Default.maxDexBin

        Returns:
        --------
        pandas.DataFrame with each row the result of analyseRegion() for bins
        are returned by createBins()
        """
        if maxm < 2:
            raise Exception("maxm =",maxm,"< 2")
        b = self.createBins(minNBin,maxNBin,maxDexBin)
        a = self.analyseRegion(b[0],maxm=maxm,computeOmega=computeOmega,
                               computeAlpha=computeAlpha,tophat=tophat)
        d = pd.DataFrame(columns=a.index)
        d.loc[0] = a.to_numpy()
        for k in range(1,len(b)):
            a = self.analyseRegion(b[k],maxm=maxm,computeOmega=computeOmega,
                                   computeAlpha=computeAlpha,tophat=tophat)
            d.loc[k] = a.to_numpy()
        return d
    
    def maximumWaveNumber(self,discAnalysis):
        """ auxiliary: find maximum m used in disc analysis """
        names = discAnalysis.columns.to_list()
        m = 0
        while(names.count('A'+str(m+1))):
            m += 1
        return m
 
    def findBarRegion(self, discAnalysis,
                      minBarStrength=Default.minBarStrength,
                      minMaxBarStrength=Default.minMaxBarStrength,
                      maxDPsi=Default.maxDPsi,
                      minDexBar=Default.minDexBar,
                      minNumBar=Default.minNumBar,
                      maxm=None):
        """
        find bar region using a variation of the method described by Dehnen
        et al (2023) in Appendix C. In particular, instead of A2, we use
            S = rms{A[m=even]} - rms{A[m=odd]}
        as measure of bar strength.

        Input data:
        -----------
        discAnalysis: pandas.DataFrame
            output from analyseDisc()

        Parameters:
        -----------
        minMaxBarStrength: float
            require max{S} ≥ minMaxBarStrength for bar
            Default: Default.minMaxBarStrength
        minBarStrength: float
            require S ≥ minBarStrength in bar region
            Default: Default.minBarStrength
        maxDPsi: scalar
            maximum angular width of bar [degrees]
            Default: Default.maxDPsi
        minDexBar: float
            minimum required length of bar in log10(R)
            Default: Default.minDexBar
        minNumBar: int
            minimum required number of particles in bar region
            Default: Default.minNumBar

        Returns: i0,i1,Rm
        -----------------
            i0,i1: indices into sorted data arrays for bar region
            Rm:    radius at which S is maximal
        """
        # 0  sanity checks
        if not type(minMaxBarStrength) is float:
            raise Exception("minMaxBarStrength must be float")
        if minMaxBarStrength <= 0.0:
            raise Exception("minMaxBarStrength = "+str(minMaxBarStrength)
                            +" ≤ 0")
        if minMaxBarStrength > 0.4:
            raise Exception("minMaxBarStrength = "+str(minMaxBarStrength)
                            +" is too large")
        if not (type(maxDPsi) is float or type(maxDPsi) is int):
            raise Exception("maxDPsi must be float")
        if maxDPsi < 2:
            raise Exception("maxDPsi"+str(maxDPsi)+" is too small")
        if maxDPsi > 20:
            raise Exception("maxDPsi"+str(maxDPsi)+" is too large")
        if minNumBar < 1000:
            text = "minNumBar"+str(minNumBar)+\
                " is too small -- will use 1000 instead"
            warning.warn(text)
            minNumBar = 1000
        maxDPsi = maxDPsi * np.pi / 180.0
        if maxm is None:
            maxm = self.maximumWaveNumber(discAnalysis)
        if maxm < 2:
            raise Exception("require maxm ≥ 2 with analyseDisc()")
        # 0  obtain m=2 phase and bar strength
        psi = discAnalysis['ps2']
        Am = discAnalysis[['A'+str(m) for m in range(1,maxm+1)]].to_numpy()
        Ae = Am[:,1::2]
        Ao = Am[:,0::2]
        S  = np.sqrt((Ae*Ae).sum(axis=1)) - np.sqrt((Ao*Ao).sum(axis=1))
        # 1  find maximum bar strength
        b0 = np.argmax(S)
        Rm = discAnalysis['R'][b0]
        if S[b0] < minMaxBarStrength:
            return 0,0,0.0
        # 2  set ψ = ψ2 - ψ2(maximum S) in [-π/2,π/2]
        psi-= psi[b0]
        psi = np.where(psi> 0.5*np.pi, psi-np.pi, \
              np.where(psi<-0.5*np.pi, psi+np.pi, psi))
        # 3  extend bar region of bins [b0,b1]
        nB = len(psi)
        b1 = b0
        psimin = psi[b0]
        psimax = psi[b1]
        width = lambda ps : max(ps,psimax) - min(ps,psimin)
        # w0 and w1 are the widths in psi of the bar region IF it would be
        # extended by bin b0-1 or b1+1, respectively. We set them to 2 if
        # such an extension is not possible.
        w0 = width(psi[b0-1]) if b0  >0  and S[b0-1]>minBarStrength else 2
        w1 = width(psi[b1+1]) if b1+1<nB and S[b1+1]>minBarStrength else 2
        while min(w0,w1) < maxDPsi:
            if w0 < w1:
                b0-= 1
                w0 = width(psi[b0-1]) \
                    if b0  >0  and S[b0-1]>minBarStrength else 2
                psimin = min(psi[b0],psimin)
                psimax = max(psi[b0],psimax)
            else:
                b1+= 1
                w1 = width(psi[b1+1]) \
                    if b1+1<nB and S[b1+1]>minBarStrength else 2
                psimin = min(psi[b1],psimin)
                psimax = max(psi[b1],psimax)
        # 4  obtain bar region of indices [i0,i1] into sorted tables
        i0 = int(discAnalysis['i0'][b0])
        i1 = int(discAnalysis['i1'][b1])
        if i1 < i0 + minNumBar or \
            np.log10(self.Rq[i1]/self.Rq[i0]) < 2*minDexBar:
            return 0,0,0.0
        return i0,i1,Rm

    def findBar(self, maxm=Default.maxm, tophat=True,
                minNBin=Default.minNBin,
                maxNBin=Default.maxNBin,
                maxDexBin=Default.maxDexBin,
                minBarStrength=Default.minBarStrength,
                minMaxBarStrength=Default.minMaxBarStrength,
                maxDPsi=Default.maxDPsi,
                minDexBar=Default.minDexBar,
                minNumBar=Default.minNumBar):
        """
        combines analyseDisc() and findBarRegion()

        Parameters:
        -----------
        maxm: int
            maximum azimuthal wavenumber m
            Default: Default.maxm
        tophat: bool
            use a tophat window. A tophat window causes no bias for
            Fourier amplitude and phase, which are all we need here
            Default: True
        minNbin: int
            minimum number of particles in radial bin
            Default: Default.minNBin
        maxNbin: int
            maximum number of particles in radial bin
            Default: Default.maxNbin
        maxDexBin: float
            maximum size of radial bin in log10(R)
            Default: Default.maxDexBin
        minMaxBarStrength: float
            require max{S} ≥ minMaxBarStrength for bar
            Default: Default.minMaxBarStrength
        minBarStrength: float
            require S ≥ minBarStrength in bar region
            Default: Default.minBarStrength
        maxDPsi: scalar
            maximum angular width of bar [degrees]
            Default: Default.maxDPsi
        minDexBar: float
            minimum required length of bar in log10(R)
            Default: Default.minDexBar
        minNumBar: int
            minimum required number of particles in bar region
            Default: Default.minNumBar

        Returns: i0,i1,Rm
        -----------------
            i0,i1: indices into sorted data arrays for bar region
            Rm:    radius at which S is maximal
        """

        return self.findBarRegion(self.analyseDisc(maxm=maxm, tophat=tophat,
                                                   minNBin=minNBin,
                                                   maxNBin=maxNBin,
                                                   maxDexBin=maxDexBin),
                                  maxm=maxm,
                                  minBarStrength=minBarStrength,
                                  minMaxBarStrength=minMaxBarStrength,
                                  maxDPsi=maxDPsi,
                                  minDexBar=minDexBar,
                                  minNumBar=minNumBar)
    
    def measureOmega(self, region, m=2):
        """ 
        compute ψ,Ω for given radial region
        
        Parameters:
        -----------
        region: iterable with two or three entries: (i0,i1[,Rm]) or (R0,R1[,Rm])
            i0,i1 = first and end indices of bar region
            R0,R1 = inner and outer radius of bar region
            Rm    = central point (where to centre window)
            for example the output from findBarRegion()
        m: int
            azimuthal wave number for which to measure ψ,Ω
            Default: 2
    
        Return:
        pandas.Series holding R0,Rm,,R1,ψ,Ω,ψ_e,Ω_e,corr(ψ,Ω)
        
        """
        if m < 1:
            raise Exception("m =",m,"< 1")
        i0,i1 = self.unpackRegion(region)
        nP  = len(self.Rq)
        nB  = i1-i0
        H   = harmonic(self.Cph[i0:i1],self.Sph[i0:i1],m=m)
        Rq0 = 0.5*(  self.Rq[i0]   + self.Rq[i0-1])  if i0 > 0  else 0
        R0  = np.sqrt(Rq0)
        Rq1 = 0.5*(  self.Rq[i1-1] + self.Rq[i1]  )  if i1 < nP else \
              0.5*(3*self.Rq[i1-1] + self.Rq[i1-2])
        R1  = np.sqrt(Rq1)
        iD  = 1.0/(Rq1-Rq0)
        if len(region)>2:
            Rm  = region[2]
            Rqm = Rm*Rm
        else:
            im  = (i1+i0)//2
            Rqm = self.Rq[im] if (nB%2)==0 else \
                  0.5 * (self.Rq[im] + self.Rq[im+1])
            Rm  = np.sqrt(Rqm)
        Q   = self.Rq[i0:i1]-Rqm                    # R²-Rm²
        fac = np.where(Q<0,1/(Rq0-Rqm),1/(Rq1-Rqm)) # 1/(Re²-Rm²)
        Q  *= fac                                   # Q=(R²-Rm²)/(Re²-Rm²)
        W,dW= windowDeriv(Q)                        # W(Q), dW/dQ
        dW  = dW*fac                                # dW/dR²
        mu  = self.M[i0:i1]
        mW  = mu*W                                  # μ W
        mdWt= mu*dW*self.dRq[i0:i1]                 # μ dW/dt
        sR  = pd.Series(data=(R0,Rm,R1), index=('R0','Rm','R1'))
        v   = variance((nB*iD*mW/np.pi))
        ApO,corr = self.analyseAux(H,self.dPh[i0:i1],iD,mW,mdWt,
                                   correlation=True)
        sM  = pd.Series(data=(ApO['ps'+str(m)],ApO['ps'+str(m)+'_e'],
                              ApO['Om'+str(m)],ApO['Om'+str(m)+'_e'],
                              corr[1,2]),
                        index=('psi','psi_e','Om','Om_e','corr'))
        return pd.concat((sR,sM))

    @staticmethod
    def patternSpeed(x,y, vx, vy, mu=1.0, m=2, checkFiniteInput=False,
                     maxm=Default.maxm, minNbin=Default.minNBin,
                     maxNBin=Default.maxNBin, maxDexBin=Default.maxDexBin,
                     minBarStrength=Default.minBarStrength,
                     minMaxBarStrength=Default.minMaxBarStrength,
                     maxDPsi=Default.maxDPsi, minDexBar=Default.minDexBar,
                     minNumBar=Default.minNumBar, tophatFourier=True):
        """
        improved Dehnen et al. (2023) Fourier method for measurng pattern
        speed in one go (discarding any intermediate results, such as the
        Fourier analysis in radial bins)

        Input Data:
        -----------
        x : numpy 1D array
            centred x positions
        y : numpy 1D array
            centred y positions
        vx : numpy 1D array
            centred x velocities
        vy : numpy 1D array
            centred y velocities
        mu : float or 1D numpy array
            particle mass(es)
    
        Parameters:
        -----------
        m: int
            azimuthal wavenumber of wave to measure pattern speed for
        checkFiniteInput: bool
            check input data for NaN or Inf
            Default: False
        maxm: int
            maximum azimuthal wavenumber used in Fourier analysis of disc
            Detault: Default.maxm
        minNbin: int
            minimum number of particles in radial bin
            Default: Default.minNbin
        maxNbin: int
            maximum number of particles in radial bin
            Default: Default.maxNbin
        maxDexBin: float
            maximum size of radial bin in log10(R)
            Default: Default.maxDexBin
        minMaxBarStrength: float
            require max{S} ≥ minMaxBarStrength for bar
            Default: Default.minMaxBarStrength
        minBarStrength: float
            require S ≥ minBarStrength in bar region
            Default: Default.minBarStrength
        maxDPsi: scalar
            maximum angular width of bar [degrees]
            Default: Default.maxDPsi
        minDexBar: float
            minimum required length of bar in log10(R)
            Default: Default.minDexBar
        minNumBar: int
            minimum required number of particles in bar region
            Default: Default.minNumBar
        tophatFourier: bool
            use a top-hat weighting (instead of a smooth window) for the Fourier
            analysis in radial bins. This is recommended and causes no bias.
            Default: True

        Returns:
            pandas.Series holding R0,R1,Rm,ψ,Ω,ψ_e,Ω_e,corr(ψ,Ω)
        """
        tool = FourierMethod(x,y,vx,vy,mu,checkFinite=checkFiniteInput)
        bar  = tool.findBar(maxm=maxm, tophat=tophatFourier,
                            minNBin=minNBin, maxNBin=maxNBin,
                            maxDexBin=maxDexBin, minBarStrength=minBarStrength,
                            minMaxBarStrength=minMaxBarStrength,
                            maxDPsi=maxDPsi, minDexBar=minDexBar,
                            minNumBar=minNumBar)
        return tool.measureOmega(bar,m=m)


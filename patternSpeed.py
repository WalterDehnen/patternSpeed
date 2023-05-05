# -*- coding: utf-8 -*-
"""

@file      patternSpeed.py

@author    Walter Dehnen, Marcin Semczuk

@copyright Walter Dehnen, Marcin Semczuk (2022,23)

@license   GNU GENERAL PUBLIC LICENSE version 3.0
           see file LICENSE for details

@version   0.1   jun-2022 MS  initial code
@version   0.2   sep-2022 WD  patternSpeed.py
@version   0.3   Feb-2023 WD  general m; avoid empty last bin in createBins()

"""
version = '0.3'

import numpy as np
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

def settrigonometric(x,y,m):
    """set R² and cosmφ, sinmφ"""
    if m <= 0:
        raise Exception("m="+str(m)+" ≤ 0")
    x2 = x*x
    y2 = y*y
    R2 = x2 + y2
    iR2 = np.reciprocal(R2)
    nrm = np.sqrt(iR2) if (m&1) else iR2
    k,Ck,Sk = (1,x*nrm,y*nrm) if (m&1) else (2,(x2-y2)*nrm,2*x*y*nrm)
    Cm,Sm = (Ck,Sk) if (m&k) else (1,0)
    kmax = 1<<(m.bit_length()-1)
    while k+k <= kmax:
        k,Ck,Sk = k+k, Ck*Ck-Sk*Sk, 2*Ck*Sk
        if (m&k):
            Cm,Sm = Cm*Ck-Sm*Sk, Cm*Sk+Sm*Ck
    return R2,iR2,Cm,Sm

def amplPhase3(x, m=2):
    """x=[<1>, <cos(mφ)>, <sin(mφ)>], compute A2 and ψ, and Jacobian"""
    f = np.empty((2))
    J = np.empty((2,3))
    Z = np.hypot(x[1],x[2])
    f[0] = Z/x[0]               # A = √(Cm²+Sm²)/C0
    f[1] = atan(x[2],x[1])/m    # ψ = 1/m atan(Sm/Cm)
    J[0,0] =-f[0]/x[0]          # ∂A/∂C0 = -A/C0
    J[0,1] = x[1]/(Z*x[0])      # ∂A/∂Cm = Cm/√(Cm² + Sm²)/C0
    J[0,2] = x[2]/(Z*x[0])      # ∂A/∂Sm = Sm/√(Cm² + Sm²)/C0
    J[1,0] = 0.0                # ∂ψ/∂C0 = 0
    J[1,1] =-x[2]/(m*Z*Z)       # ∂ψ/∂Cm =-Sm/(Cm² + Sm²)/m
    J[1,2] = x[1]/(m*Z*Z)       # ∂ψ/∂Sm = Cm/(Cm² + Sm²)/m
    return f,J

def amplPhase2(x, C0, m=2):
    """x=[<cos(mφ)>, <sin(mφ)>], C0=<1>: compute A2 and ψ, and Jacobian"""
    f,J = amplPhase3((C0,x[0],x[1]), m=m)
    return f,J[1:,:]

def phaseOmega(x, m=2):
    """compute ψ and Ω=dψ/dt from <cosmφ> <sinmφ> and their time derivatives"""
    f = np.empty((2))
    J = np.empty((2,4))
    im = 1/m
    iQ = 1/(x[0]*x[0]+x[1]*x[1])         # 1/(C²+S²)
    J[0,0] =-im*x[1]*iQ                  # ∂ψ/∂C =-S/(C²+S²)/m
    J[0,1] = im*x[0]*iQ                  # ∂ψ/∂S = C/(C²+S²)/m
    J[0,2] = 0
    J[0,3] = 0
    f[0]   = im*atan(x[1],x[0])          # ψ      = 1/m atan(S/C)
    f[1]   = J[0,0]*x[2] + J[0,1]*x[3]   # Ω = dψ = (C dS - S dC)/(C²+S²)/m
    J[1,0] =( im*x[3] - 2*x[0]*f[1])*iQ  # ∂Ω/∂C  = ( dS/m - 2 C Ω)/(C²+S²)
    J[1,1] =(-im*x[2] - 2*x[1]*f[1])*iQ  # ∂Ω/∂S  = (-dC/m - 2 S Ω)/(C²+S²)
    J[1,2] = J[0,0]                      # ∂Ω/∂dC = ∂ψ/∂C
    J[1,3] = J[0,1]                      # ∂Ω/∂dS = ∂ψ/∂S
    return f,J

def asfarray(array, copy=False, checkFinite=False):
    """obtain a new float array from input array"""
    arr = np.asfarray(array)             # ensure float data type
    if checkFinite:
        arr = np.asarray_chkfinite(arr)  # check for NaN or inf
    if copy and arr is array:
        arr = np.array(arr,copy=True)    # ensure we make a copy
    return arr

def createBins(rq, minNBin=4000, maxNBin=50000, maxDexBin=0.15):
    """
    create radial bins, used in FourierMethod

    Input data:
    -----------
    rq: array like
        sorted table of radius-squared in ascending order

    Parameters:
    -----------
    minNbin: int
        minimum number of particles in radial bin
        Default: 4000
    maxNbin: int
        maximum number of particles in radial bin
        Default: 50000
    maxDexBin: float
        maximum size of radial bin in log10(R)
        Default: 0.15

    Returns:
    --------
    bins: list of tuples(i0,i1)
        i0,i1 = start and end indices into the sorted data arrays
    """
    # 0.  sanity checks on parameters
    nP = len(rq)
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
        rqm = maxRqFac * rq[i0]
        im  = min(nP, i0 + maxNBin)
        i1 += np.searchsorted(rq[i1:im], rqm)
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

class FourierMethod:
    """Dehnen et al. (2022) Fourier method for measuring pattern speed.

    Overview
    --------
    The measurement of the bar pattern speed takes three steps:
    0  preparing data tables sorted in radius R (at initialisation)
    1  finding the bar region
    2  measuring {ψ,Ω} for the bar region
    Users may skip step 1 if they want to measure {ψ,Ω} for another radial
    range of particles than that detected in step 1. See below for details.

    Step 1
    ------
    Finding the bar region is done in 3 steps:
    1.1  createBins()      creates bins in cylindrical radius
    1.2  analyseBins()     finds A2=Σ2/Σ0 and ψ=ψ2 in each bin
    1.3  findBarRegion()   identifies the bar region as range of bins
    This results in a pair of indices (i0,i1) into the sorted arrays.

    In rare situations, step 1.3 fails to find the bar. This occurs if the
    maximum A2 is outside the bar, for example due to an infalling satellite.
    In such cases, a simple fix is to reduce the radial range of the data.

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

    see §2.3 of paper. Strictly, our method in step 1 violates condition (1),
    but we found any bias to be insignificant. Violations of condition (2) are
    avoided when selecting particles on conserved quantities, such as stellar
    birth properties (time, location, metallicity) or a random sub-sample
    (selecting on id).

    """

    def radius(self,i):
        """radius for given index (or indices) into sorted table"""
        return np.sqrt(self.Rq[i])
    
    def indexR(self,R):
        """index i such that radius(i) ≤ R < radius(i+1) or 0 or N"""
        return min(len(self.Rq),max(0,np.searchsorted(self.Rq, R*R)))

    def numPart(self):
        """number of particles loaded"""
        return len(self.Rq)

    def __init__(self, mu, x,y, vx,vy, m=2, checkFinite=False):
        """step 0  create sorted data arrays

        Input Data:
        -----------
        mu : scalar or array-like
            particle mass(es)
        x : array-like
            centred x positions
        y : array-like
            centred y positions
        vx : array-like
            centred x velocities
        vy : array-like
            centred y velocities

        When providing only a sub-set of simulation particles, care must be
        exercised to avoid a selection that leads to a violation of the
        continuity equation, in particular selecting on the particles
        instantaneous position or velocity inside the bar region will
        unavoidably lead to biased results. (See also the explanation
        in the documentation of patternSpeed.FourierMethod)

        Parameters:
        -----------
        m : int
            azimuthal wave number > 0
        checkFinite: bool
            check input data for NaN or Inf

        """
        # 0   get input data to numpy arrays
        asarray = np.asarray_chkfinite if checkFinite else np.asarray
        M = asfarray(mu,checkFinite=checkFinite)
        X = asfarray(x,checkFinite=checkFinite)
        Y = asfarray(y,checkFinite=checkFinite)
        Vx = asfarray(vx,checkFinite=checkFinite)
        Vy = asfarray(vy,checkFinite=checkFinite)
        # 1   compute derived data
        self.Rq,iRq,Cm,Sm = settrigonometric(X,Y,m)
        self.m = m
        self.MCm = M*Cm                 # μ cosmφ
        self.MSm = M*Sm                 # μ sinmφ
        self.dRq =   2 * (X*Vx + Y*Vy)  # dR²/dt
        self.dPh = iRq * (X*Vy - Y*Vx)  # dφ/dt
        # 2   sort data to ascending R²
        i = np.argsort(self.Rq)
        self.Rq = self.Rq[i]
        self.dRq = self.dRq[i]
        self.dPh = self.dPh[i]
        self.MCm = self.MCm[i]
        self.MSm = self.MSm[i]
        self.M = M[i] if len(M)>1 else M

    def createBins(self, minNBin=4000, maxNBin=50000, maxDexBin=0.15):
        """
        step 1.1  create radial bins

        Parameters:
        -----------
        minNbin: int
            minimum number of particles in radial bin
            Default: 4000
        maxNbin: int
            maximum number of particles in radial bin
            Default: 50000
        maxDexBin: float
            maximum size of radial bin in log10(R)
            Default: 0.15

        Returns:
        --------
        bins: list of tuples(i0,i1)
            i0,i1 = start and end indices into the sorted data arrays
        """
        return createBins(self.Rq, minNBin=minNBin, maxNBin=maxNBin, \
                          maxDexBin=maxDexBin)

    def analyseRegion(self, region, tophat=True):
        """
        analyse shape and orientation in given region

        Parameters:
        -----------
        region: tuple of int
            indices into sorted arrays, specifying region to analyse
        tophat: bool
            use a tophat weighting (instead of a smooth window) for estimating 
            A2 and ψ in each bin. This is recommended and causes no bias.
            Default: True

        Result:
        -------
        nB:       number of particles in region
        R0,Rm,R1: inner, middle, and outer radius of region
        Σ0:       mean surface density in region
        Am,Ame:   Am=Σm/Σ0 and its statistical uncertainty
        ψm,ψme    ψm (phase angle) and its statistical uncertainty
        """
        i0 = region[0]
        i1 = region[1]
        if type(i0) is float:
            i0 = indexR(i0)
        if type(i1) is float:
            i1 = indexR(i1)
        # 0.2 sanity checks on parameters
        nP = len(self.Rq)
        if i0 < 0:
            raise runtimeError("i0="+str(i0)+" < 0")
        if i1 <= i0:
            raise runtimeError("i1="+str(i1)+" ≤ i0="+str(i0))
        if nP < i1:
            raise runtimeError("i1="+str(i1)+" > N="+str(nP))
        nB = i1-i0
        Rq0 = 0.5*(  self.Rq[i0]   + self.Rq[i0-1])  if i0 > 0  else 0
        Rq1 = 0.5*(  self.Rq[i1-1] + self.Rq[i1]  )  if i1 < nP else \
              0.5*(3*self.Rq[i1-1] + self.Rq[i1-2])
        Rqm = 0.5*(Rq0+Rq1)
        Rq = self.Rq[i0:i1]
        c0 = self.M[i0:i1] if type(self.M) is np.ndarray else self.M
        cm = self.MCm[i0:i1]
        sm = self.MSm[i0:i1]
        iD = 1.0/(Rq1-Rq0)
        # 1.2  multiply by W(Q) if using smooth window
        if not tophat:
            iD = 2*iD
            q  = np.abs(Rq-Rqm)*iD
            W  = window(q)
            c0 = W*c0
            cm = W*cm
            sm = W*sm
        fac = nP * iD / (2*np.pi)
        c0 *= fac
        cm *= fac
        sm *= fac
        # TEST
        if debug > 1:
            print('c0=',c0)
            print('cm=',cm)
            print('sm=',sm)
        # TSET
        # 1.3  sum terms and compute results
        c0isArray = type(c0) is np.ndarray
        CCS = variance([c0,cm,sm])       if c0isArray else variance([cm,sm])
        # TEST
        if debug > 0:
            print('m=2: μ(c0,cm,sm)=',CCS.mean())
            print('m=2: σ(c0,cm,sm)=',CCS.std_of_mean())
        # TSET
        Sd0 = CCS.mean(0)                if c0isArray else c0 * fac
        APm = CCS.propagate(amplPhase3,args=(self.m,))    if c0isArray else \
              CS2.propagate(amplPhase2,args=(Sd0,self.m,))
        # TEST
        if debug > 0:
            print('m=2: μ(A,ψ)=',APm.mean())
            print('m=2: σ(A,ψ)=',APm.std_of_mean())
        # TSET
        return nB, np.sqrt(Rq0), np.sqrt(Rqm), np.sqrt(Rq1), Sd0, \
            APm.mean(0), APm.std_of_mean(0), APm.mean(1), APm.std_of_mean(1)

    def analyseBins(self, bins, tophat=True):
        """
        step 1.2  analyse bins: compute for each bin A2,ψ2 and uncertainties

        Parameters:
        -----------
        bins: list of tuples(i0,i1)
            as returned by createBins()
        tophat: bool
            use a tophat weighting (instead of a smooth window) for estimating 
            A2 and ψ in each bin. This is recommended and causes no bias.
            Default: True

        Returns:
        --------
        binData : numpy.ndarray with shape = (nbin,9)
            binData[:,0]    number of particles in bin
            binData[:,1:3]  inner, middle, and outer radius of bin
            binData[:,4]    mean surface density Σ0 in bin
            binData[:,5:6]  Am=Σm/Σ0 and its statistical uncertainty
            binData[:,7:8]  ψm and its statistical uncertainty
        """
        binData = np.empty((len(bins),9))
        i = 0
        for b in bins:
            binData[i,:] = self.analyseRegion(b,tophat)
            i += 1
        return binData
            
    def findBarRegion(self, bins, binData, minA2Bar=0.2, maxDPsi=10.0, \
                      minDexBar=0.2, minNumBar=100000):
        """
        step 1.3  find bar region (see Dehnen et al 2022, Appendix C).
        ONLY sensible for self.m == 2

        In rare situations, this fails to find the bar. This occurs if the
        maximum A2 is outside the bar, for example due to a satellite.
        In such cases, a simple fix is to reduce the radial range of the data.

        Input data:
        -----------
        bins: list of tuples (i0,i1), as returned by createBins()
            (i0,i1) = begin and end index of bin into sorted tables
        binData: list of tuples (N,R0,R1,Σ0,A2,A2e,ψ2,ψ2e), as returned
            by analyseBins()

        Parameters:
        -----------
        minA2bar: float
            minimum A2 = Σ2/Σ0 needed for bar
            Default: 0.2
        maxDPsi: float (or int)
            maximum angular width of bar [degrees]
            Default: 10
        minDexBar: float
            minimum required size of bar in log10(R)
            Default: 0.2
        minNumBar: int
            minimum required number of particles in bar region
            Default: 100000

        Returns: i0,i1
        --------------
            i0,i1: indices into sorted data arrays for bar region
        """
        # 0.  sanity checks
        if not self.m == 2:
            raise Exception("the bar region should be defined using "\
                            "the m=2 Fourier analysis")
        if not type(minA2Bar) is float:
            raise Exception("minA2Bar must be float")
        if minA2Bar <= 0.0:
            raise Exception("minA2Bar"+str(minA2Bar)+" ≤ 0")
        if minA2Bar > 0.2:
            raise Exception("minA2Bar"+str(minA2Bar)+" is too large")
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
        # 1.  find bin with maximum A2
        A2 = binData[:,5]
        b0 = np.argmax(A2)
        if A2[b0] < minA2Bar:
            return 0,0
        minA2 = max(minA2Bar, 0.5*A2[b0])
        # 2.  set ψ = ψ2 - ψ2(maximum A2) in [-π/2,π/2]
        psi = binData[:,7] - binData[b0,7]
        psi = np.where(psi> 0.5*np.pi, psi-np.pi, \
              np.where(psi<-0.5*np.pi, psi+np.pi, psi))
        # 3.  extend bar region of bins [b0,b1]
        nB = len(binData)
        b1 = b0
        psimin = psi[b0]
        psimax = psi[b1]
        width = lambda ps : max(ps,psimax) - min(ps,psimin)
        # w0 and w1 are the widths in psi of the bar region IF it would be
        # extended by bin b0-1 or b1+1, respectively. We set them to 2 if
        # such an extension is not possible.
        w0 = width(psi[b0-1]) if b0  >0  and A2[b0-1]>minA2 else 2
        w1 = width(psi[b1+1]) if b1+1<nB and A2[b1+1]>minA2 else 2
        while min(w0,w1) < maxDPsi:
            if w0 < w1:
                b0-= 1
                w0 = width(psi[b0-1]) if b0  >0  and A2[b0-1]>minA2 else 2
                psimin = min(psi[b0],psimin)
                psimax = max(psi[b0],psimax)
            else:
                b1+= 1
                w1 = width(psi[b1+1]) if b1+1<nB and A2[b1+1]>minA2 else 2
                psimin = min(psi[b1],psimin)
                psimax = max(psi[b1],psimax)
        # 4.  obtain bar region of indices [i0,i1] into sorted tables
        i0 = bins[b0][0]
        i1 = bins[b1][1]
        if i1 < i0 + minNumBar or \
            np.log10(self.Rq[i1]/self.Rq[i0]) < 2*minDexBar:
            return 0,0
        return i0,i1

    def findBarRange(self,  minNBin=4000, maxNBin=50000, maxDexBin=0.15,\
                minA2Bar=0.2, maxDPsi=10.0, minDexBar=0.2, minNumBar=100000,
                tophat=True):
        """step 1:  find bar region in one call

        ONLY sensible for self.m == 2

        Parameters:
        -----------
        minNbin: int
            minimum number of particles in radial bin
            Default: 4000
        maxNbin: int
            maximum number of particles in radial bin
            Default: 50000
        maxDexBin: float
            maximum size of radial bin in log10(R)
            Default: 0.15
        minA2bar: float
            minimum A2 = Σ2/Σ0 needed for bar
            Default: 0.2
        maxDPsi: float (or int)
            maximum angular width of bar [degrees]
            Default: 10
        minDexBar: float
            minimum required size of bar in log10(R)
            Default: 0.2
        minNumBar: int
            minimum required number of particles in bar region
            Default: 100000
        tophat: bool
            use a tophat weighting (instead of a smooth window) for estimating 
            A2 and ψ in each bin. This is recommended and causes no bias.
            Default: True

        Returns: i0,i1
        --------------
            i0,i1: indices into sorted data arrays for bar region
        """
        bins = self.createBins(minNBin=minNBin, maxNBin=maxNBin, \
                               maxDexBin=maxDexBin)
        binData = self.analyseBins(bins, tophat=tophat)
        return self.findBarRegion(bins,binData,minA2Bar=minA2Bar, \
                                  maxDPsi=maxDPsi, minDexBar=minDexBar, \
                                  minNumBar=minNumBar)

    def measureOmega(self, barRegion, tophat=False, fullSample=False):
        """step 2:  obtain ψ, Ω and their uncertainties

        Parameters:
        -----------
        barRegion: iterable with two entries: (i0,i1) or (R0,R1)
            first and end indices or radii of bar region, for example the
            output from findBarRegion()
            If there are less than 100 particles in the bar region, no analysis
            is performed and all zero returned.
        fullSample: boolean
            compute uncertainties based on the full sample or only the
            sub-sample in the barRegion? If true, we add for each particle
            outside the barRegion a data entry with zero weight and use for N
            the full sample size.

        Returns: R0, Rm, R1, ψ, ψe, Ω, Ωe, C
        ------------------------------------
            R0,Rm,R1 = inner, median, and outer radius of bar region
            ψ,ψe     = bar phase and its statistical uncertainty
            Ω,Ωe     = bar pattern speed and its statistical uncertainty
            C        = statistical correlation between ψ and Ω

        A note on bias
        --------------
        The measurement of Ω will be unbiased, if either all particles within
        the bar region have been selected (at initialisation) or if any
        selection within that region is based on strictly conserved quantities
        such as stellar birth properties (time, location, and metallicity), or
        particle type (baryonic/non-baryonic)

        """
        # 0.  prepare
        # 0.1 unpack input
        i0 = barRegion[0]
        i1 = barRegion[1]
        if type(i0) is float:
            i0 = indexR(i0)
        if type(i1) is float:
            i1 = indexR(i1)
        # 0.2 sanity checks on parameters
        nP = len(self.Rq)
        if i0 < 0:
            raise runtimeError("i0="+str(i0)+" < 0")
        if i1 <= i0:
            raise runtimeError("i1="+str(i1)+" ≤ i0="+str(i0))
        if nP < i1:
            raise runtimeError("i1="+str(i1)+" > N="+str(nP))
        nB = i1-i0
        if nB < 100:
            return 0.,0.,0.,0.,0.,0.,0.,0.
        if tophat:
            text = "measuring a pattern speed with a top-hat "+\
                   "window gives biased results"
            warnings.warn(text)
        # 1.  find edge and median R²
        Rq0 = 0.5*(  self.Rq[i0]   + self.Rq[i0-1])  if i0 > 0  else 0
        Rq1 = 0.5*(  self.Rq[i1-1] + self.Rq[i1]  )  if i1 < nP else \
              0.5*(3*self.Rq[i1-1] + self.Rq[i1-2])
        im  = (i1+i0)//2
        Rqm = self.Rq[im] if (nB%2)==0 else 0.5 * (self.Rq[im] + self.Rq[im+1])
        # 2.  set  m W{cos(mφ),sin(mφ)} and their time derivatives
        mu  = self.M[i0:i1]
        cm  = self.MCm[i0:i1]                           # μ cosmφ
        sm  = self.MSm[i0:i1]                           # μ sinmφ
        dcm =-self.m*self.dPh[i0:i1]*sm                 # μ d(cosmφ)/dt
        dsm = self.m*self.dPh[i0:i1]*cm                 # μ d(sinmφ)/dt
        if not tophat:
            Q   = self.Rq[i0:i1]-Rqm                    # R²-Rm²
            fac = np.where(Q<0,1/(Rq0-Rqm),1/(Rq1-Rqm)) # 1/(Re²-Rm²)
            Q  *= fac                                   # Q=(R²-Rm²)/(Re²-Rm²)
            W,dW= windowDeriv(Q)                        # W(Q), dW/dQ
            dW *= fac                                   # dW/dR²
            dW *= self.dRq[i0:i1]                       # dW/dt
            mu  = W*mu
            cm  = W*cm                                  # μ W cosmφ
            sm  = W*sm                                  # μ W sinmφ
            dcm = W*dcm + dW*self.MCm[i0:i1]            # μ d(W cosmφ)/dt
            dsm = W*dsm + dW*self.MSm[i0:i1]            # μ d(W sinmφ)/dt
        # 3.  compute sample mean and co-variance
        var = variance([cm,sm,dcm,dsm])
        if debug > 1:
            vmu = variance([mu])
            print("<1,cm,sm>=",vmu.mean(),var.mean(0),var.mean(1))
        if fullSample:
            var.append_zero(len(self.MCm) - len(cm))
        if debug > 1:
            print("μ=",var.mean())
            print("σ=",var.std_of_mean())
        # 4.  compute ψ, Ω, their uncertainties, and correlation
        var = var.propagate(phaseOmega, args=(self.m,))
        return np.sqrt(Rq0),np.sqrt(Rqm),np.sqrt(Rq1), \
               var.mean(0), var.std_of_mean(0), \
               var.mean(1), var.std_of_mean(1), \
               var.corr(0,1)

def patternSpeedFourier(mu, x,y, vx, vy, checkFiniteInput=False,
                        minNBin=4000, maxNBin=50000, maxDexBin=0.15, \
                        minA2Bar=0.2, maxDPsi=10.0, minDexBar=0.2, \
                        minNumBar=100000, tophatWithBins=True, tophat=False):
    """
    Dehnen et al. (2022) Fourier method for measurng pattern speed in one go

    Input Data:
    -----------
    mu : float or 1D numpy array
        particle mass(es)
    x : numpy 1D array
        centred x positions
    y : numpy 1D array
        centred y positions
    vx : numpy 1D array
        centred x velocities
    vy : numpy 1D array
        centred y velocities
    
    Parameters:
    -----------
    checkFiniteInput: bool
        check input data for NaN or Inf
    minNbin: int
        minimum number of particles in radial bin
        Default: 4000
    maxNbin: int
        maximum number of particles in radial bin
        Default: 50000
    maxDexBin: float
        maximum size of radial bin in log10(R)
        Default: 0.15
    minA2bar: float
        minimum A2 = Σ2/Σ0 needed for bar
        Default: 0.2
    maxDPsi: float (or int)
        maximum angular width of bar [degrees]
        Default: 10
    minDexBar: float
        minimum required size of bar region in log10(R)
        Default: 0.2
    minNumBar: int
        minimum required number of particles in bar region
        Default: 100000
    tophatWithBins: bool
        use a top-hat weighting (instead of a smooth window) for estimating 
        A2 and ψ in each bin. This is recommended and causes no bias.
        Default: True
    tophat: bool
        use a top-hat weighting (instead of a smooth window) in the bar region
        for estimating ψ and Ω. Top-hat weighting causes biased estimates of Ω.
        Default: False

    Returns: binData,result
    -----------------------
    binData : numpy.ndarray with shape(nbin,9)
        binData[:,0]    number of particles in bin
        binData[:,1:3]  inner, middle, and outer radius of bin
        binData[:,4]    mean surface density Σ0 in bin
        binData[:,5:6]  A2=Σ2/Σ0 and its statistical uncertainty
        binData[:,7:8]  ψ2 and its statistical uncertainty
    result: R0, Rm, R1, ψ, ψe, Ω, Ωe
        R0,Rm,R1 = inner, median, and outer radius of bar region
        ψ,ψe     = bar phase and its statistical uncertainty
        Ω,Ωe     = bar pattern speed and its statistical uncertainty
        C        = statistical correlation between ψ and Ω
    """
    tool = FourierMethod(mu,x,y,vx,vy,checkFinite=checkFiniteInput)
    bins = tool.createBins(minNBin=minNBin,maxNBin=maxNBin,maxDexBin=maxDexBin)
    binData = tool.analyseBins(bins,tophat=tophatWithBins)
    barRegion = tool.findBarRegion(bins,binData,minA2Bar=minA2Bar,\
                                   maxDPsi=maxDPsi,minDexBar=minDexBar,\
                                   minNumBar=minNumBar)
    result = tool.measureOmega(barRegion,tophat=tophat)
    return binData,result

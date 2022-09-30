# -*- coding: utf-8 -*-
"""

@file      patternSpeed.py

@author    Walter Dehnen, Marcin Semczuk

@copyright Walter Dehnen, Marcin Semczuk (2022)

@license   GNU GENERAL PUBLIC LICENSE version 3.0
           see file LICENSE for details

@version   0.1   jun-2022 MS  initial code
@version   0.2   sep-2022 WD  patternSpeed.py

"""
version = '0.2'

import numpy as np
import warnings
from variance import variance

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

def amplPhase3(x, m=2):
    """compute amplitude and phase as functions of cos and sin, also Jacobian"""
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
        
class FourierMethod:
    """Dehnen et al. (2022) m=2 Fourier method for measuring pattern speed.

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

    Step 2
    ------
    Measuring ψ and Ω for the bar region takes only one call to
    measureOmega()   computes {ψ,Ω} and only requires (i0,i1) as input

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
    Our method in step 2 is unbiased, i.e. satisfies ψ=∫Ω dt, as long as
    (1) the bar region does not change; and
    (2) the continuity equation holds, i.e. mass is conserved.
        Strictly, our method in step 1 violates condition (1), but we found the
    resulting bias to be very minor, much smaller than the statistical
    uncertainty. Such violations are smaller for narrower bins.
        Violations of condition (2) may arise from star formation (and stellar
    mass loss). This problem is avoided by applying the method not only to the
    star particles, but to all baryonic components combined (if star formation
    is modelled as transition from the gas phase).

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

    def __init__(self, m, x,y, vx,vy, checkFinite=False):
        """step 0  create sorted data arrays

        Input Data:
        -----------
        m : scalar or array-like
            particle mass(es)
        x : array-like
            centred x positions
        y : array-like
            centred y positions
        vx : array-like
            centred x velocities
        vy : array-like
            centred y velocities

        Parameters:
        -----------
        checkFinite: bool
            check input data for NaN or Inf

        Beware of the note on biases in the documention of
        patternSpeed.FourierMethod

        """
        # 0   get input data to numpy arrays
        asarray = np.asarray_chkfinite if checkFinite else np.asarray
        m = asarray(m)
        x = asarray(x)
        y = asarray(y)
        vx = asarray(vx)
        vy = asarray(vy)
        # 1   compute derived data
        self.Rq = x*x + y*y                 # R²
        iRq = 1/self.Rq                     # 1/R²
        self.dRq =       2 * (x*vx + y*vy)  # dR²/dt
        self.dPh =     iRq * (x*vy - y*vx)  # dφ/dt
        self.mC2 =   m*iRq * (x*x - y*y)    # μ cos2φ
        self.mS2 = 2*m*iRq *  x*y           # μ sin2φ
        # 2   sort data to ascending R²
        i = np.argsort(self.Rq)
        self.Rq = self.Rq[i]
        self.dRq = self.dRq[i]
        self.dPh = self.dPh[i]
        self.mC2 = self.mC2[i]
        self.mS2 = self.mS2[i]
        if len(m) > 1:
            self.m = m[i]

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
        # 0.  sanity checks on parameters
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
            binData[:,5:6]  A2=Σ2/Σ0 and its statistical uncertainty
            binData[:,7:8]  ψ2 and its statistical uncertainty
        """
        nP = len(self.Rq)
        binData = np.empty((len(bins),9))
        i = 0
        # 1.  loop bins
        for b in bins:
            # 1.1  find edge and rms radii, set reduced arrays
            i0 = b[0]
            i1 = b[1]
            nB = i1-i0
            Rq0 = 0.5*(  self.Rq[i0]   + self.Rq[i0-1])  if i0 > 0  else 0
            Rq1 = 0.5*(  self.Rq[i1-1] + self.Rq[i1]  )  if i1 < nP else \
                  0.5*(3*self.Rq[i1-1] + self.Rq[i1-2])
            Rqm = 0.5*(Rq0+Rq1)
            Rq = self.Rq[i0:i1]
            c0 = self.m[i0:i1] if type(self.m) is np.ndarray else self.m
            c2 = self.mC2[i0:i1]
            s2 = self.mS2[i0:i1]
            iD = 1.0/(Rq1-Rq0)
            # 1.2  multiply by W(Q) if using smooth window
            if not tophat:
                iD = 2*iD
                q  = np.abs(Rq-Rqm)*iD
                W  = window(q)
                c0 = W*c0
                c2 = W*c2
                s2 = W*s2
            fac = nP * iD / (2*np.pi)
            AP2 = None
            Sd0 = None
            # 1.3  sum terms and compute results
            if type(c0) is np.ndarray:
                CCS = variance([c0,c2,s2])
                CCS.scale(fac)
                Sd0 = CCS.mean(0)
                AP2 = CCS.propagate(amplPhase3)
            else:
                CS2 = variance([c2,s2])
                CS2.scale(fac)
                Sd0 = c0 * fac
                AP2 = CS2.propagate(amplPhase2,args=(Sd0))
            binData[i,:] = nB,np.sqrt(Rq0),np.sqrt(Rqm),np.sqrt(Rq1),Sd0, \
                AP2.mean(0), AP2.std_of_mean(0), \
                AP2.mean(1), AP2.std_of_mean(1)
            i += 1
        return binData
            
    def findBarRegion(self, bins, binData, minA2Bar=0.2, maxDPsi=10.0, \
                      minDexBar=0.2, minNumBar=100000):
        """
        step 1.3  find bar region (see Dehnen et al 2022, Appendix C).

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
        # 0.  sanity checks on parameters
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

    def measureOmega(self, barRegion, tophat=False):
        """step 2:  obtain ψ, Ω and their uncertainties

        Parameters:
        -----------
        barRegion: iterable with (at least) two entries: (i0,i1) or (R0,R1)
            first and end indices or radii of bar region, for example the
            output from findBarRegion()
            If there are less than 100 particles in the bar region, no analysis
            is performed and all zero returned.

        Returns: R0, Rm, R1, ψ, ψe, Ω, Ωe, C
        ------------------------------------
            R0,Rm,R1 = inner, median, and outer radius of bar region
            ψ,ψe     = bar phase and its statistical uncertainty
            Ω,Ωe     = bar pattern speed and its statistical uncertainty
            C        = statistical correlation between ψ and Ω

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
        if i1 < i0:
            raise runtimeError("i1="+str(i1)+" ≤ i0="+str(i0))
        if nP <= i1:
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
        # 2.  set  m W{cos2φ,sin2φ} and their time derivatives
        c2  =   self.mC2[i0:i1]                         # μ cos2φ
        s2  =   self.mS2[i0:i1]                         # μ sin2φ
        dc2 =-2*self.dPh[i0:i1]*s2                      # μ d(cos2φ)/dt
        ds2 = 2*self.dPh[i0:i1]*c2                      # μ d(sin2φ)/dt
        if not tophat:
            Q   = self.Rq[i0:i1]-Rqm                    # R²-Rm²
            fac = np.where(Q<0,1/(Rq0-Rqm),1/(Rq1-Rqm)) # 1/(Re²-Rm²)
            Q  *= fac                                   # Q=(R²-Rm²)/(Re²-Rm²)
            W,dW= windowDeriv(Q)                        # W(Q), dW/dQ
            dW *= fac                                   # dW/dR²
            dW *= self.dRq[i0:i1]                       # dW/dt
            c2  = W*c2                                  # μ W cos2φ
            s2  = W*s2                                  # μ W sin2φ
            dc2 = W*dc2 + dW*self.mC2[i0:i1]            # μ d(W cos2φ)/dt
            ds2 = W*ds2 + dW*self.mS2[i0:i1]            # μ d(W sin2φ)/dt
        # 3.  compute sample mean and co-variance
        var = variance([c2,s2,dc2,ds2])
        # 4.  compute ψ, Ω, their uncertainties, and correlation
        var = var.propagate(phaseOmega)
        return np.sqrt(Rq0),np.sqrt(Rqm),np.sqrt(Rq1), \
               var.mean(0), var.std_of_mean(0), \
               var.mean(1), var.std_of_mean(1), \
               var.corr(0,1)
    
def patternSpeedFourier(m, x,y, vx, vy, checkFiniteInput=False,
                        minNBin=4000, maxNBin=50000, maxDexBin=0.15, \
                        minA2Bar=0.2, maxDPsi=10.0, minDexBar=0.2, \
                        minNumBar=100000, tophatWithBins=True, tophat=False):
    """
    Dehnen et al. (2022) Fourier method for measurng pattern speed in one go

    Input Data:
    -----------
    m : float or 1D numpy array
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
    tool = FourierMethod(m,x,y,vx,vy,checkFinite=checkFiniteInput)
    bins = tool.createBins(minNBin=minNBin,maxNBin=maxNBin,maxDexBin=maxDexBin)
    binData = tool.analyseBins(bins,tophat=tophatWithBins)
    barRegion = tool.findBarRegion(bins,binData,minA2Bar=minA2Bar,\
                                   maxDPsi=maxDPsi,minDexBar=minDexBar,\
                                   minNumBar=minNumBar)
    result = tool.measureOmega(barRegion,tophat=tophat)
    return binData,result

import numpy as np
from utils import *
from utils import HANN as hann
from utils import BARK as bark
from math import log10, sqrt, cos, pi
from scipy import fftpack
from scipy.fftpack import fft
import math

def bandwidth(processing, out={"sumBandwidthRefb": 0, "sumBandwidthTestb": 0, "countref": 0, "counttest": 0, "BandwidthRefb": 0, "BandwidthTestb": 0}):
    # Initialize variables
    BwRef = 0
    BwTest = 0
    fftref = processing.fftref
    ffttest = processing.ffttest
    ZEROTHRESHOLD = 921
    BwMAX = 346


    ZeroThreshold = 20.0 * log10(ffttest[ZEROTHRESHOLD])
    # Step 2: Update ZeroThreshold
    for k in range(ZEROTHRESHOLD, int(hann / 2)):
        Flevtest = 20.0 * np.log10(float(ffttest[k]))        
        if Flevtest > ZeroThreshold:
            ZeroThreshold = Flevtest

    for k in range(ZEROTHRESHOLD - 1, -1, -1):
        Flevref = 20.0 * np.log10(float(fftref[k]))
        if Flevref >= 10.0 + ZeroThreshold:
            BwRef = k + 1
            break

    # Step 4: Calculate BwTest
    for k in range(BwRef - 1, -1, -1):
        Flevtest = 20.0 * np.log10(ffttest[k])
        if Flevtest >= 5.0 + ZeroThreshold:
            BwTest = k + 1
            break

    # Step 5: Update cumulative sum and count
    if BwRef > BwMAX:
        out['sumBandwidthRefb'] += BwRef
        out['countref'] += 1

    if BwTest > BwMAX:
        out['sumBandwidthTestb'] += BwTest
        out['counttest'] += 1

    # Step 6: Calculate average bandwidth
    if out['countref'] == 0:        
        out['BandwidthRefb'] = 0
    else:
        out['BandwidthRefb'] = out['sumBandwidthRefb'] / out['countref']

    if out['counttest'] == 0:
        out['BandwidthTestb'] = 0
    else:
        out['BandwidthTestb'] = out['sumBandwidthTestb'] / out['counttest']
    return out


def nmr(processing, state):
    block_idx = state["count"]
    nmrtmp = state["nmrtmp"]
    PNoise = processing.ppnoise
    M = processing.Mref
    sum_val = np.sum(PNoise / M)
    sum_val /= BARK
    nmrtmp += sum_val
    return 10 * np.log10(nmrtmp/block_idx), nmrtmp

def reldistframes(processing, state):
    block_idx = state["count"]
    PNoise = processing.ppnoise
    M = processing.Mref
    reldisttemp = 0
    for k in range(BARK):        
        if 10*np.log10(PNoise[k]/M[k]) >=1.5:
            reldisttemp += 1
    return reldisttemp/block_idx, reldisttemp

def energyth(test, ref, hann=2048, ENERGYLIMIT=8000):
    # Check the energy for the second half of the test signal
    sum_test = sum(test[hann//2:] ** 2)
    if sum_test > ENERGYLIMIT:
        return 1
    
    # Check the energy for the second half of the reference signal
    sum_ref = sum(ref[hann//2:] ** 2)
    if sum_ref > ENERGYLIMIT:
        return 1
    
    return 0

def harmstruct(processing, state, rate=16000, harmsamples=1024):
    fftref = processing.fftref
    ffttest = processing.ffttest
    EHStmp = state["EHStmp"]
    Cffttmp = state["Cffttmp"]
    n = state["count"]
    # Initialize variables
    F0 = np.zeros(harmsamples)
    C = np.zeros(harmsamples)
    hannwin = np.zeros(harmsamples)
    Csum = 0
    max_value = 0
    PATCH = 0  # Assuming PATCH is not defined in the C code
    
    # Calculate F0 values
    for k in range(harmsamples):
        if fftref[k] == 0 or ffttest[k] == 0:
            F0[k] = 0
        else:
            F0[k] = log10(fftref[k] ** 2) - log10(ffttest[k] ** 2)
    
    # Calculate C values
    for i in range(harmsamples):
        num = 0
        denoma = 0
        denomb = 0
        for k in range(harmsamples):
            num += F0[k] * F0[(i+k) % harmsamples]  # Use modulo to stay within bounds
            denoma += F0[k] ** 2
            denomb += F0[(i+k) % harmsamples] ** 2  # Use modulo to stay within bounds

        hannwin[i] = 0.5 * sqrt(8.0 / 3.0) * (1.0 - cos(2.0 * pi * i / (harmsamples - 1.0)))
        C[i] = num / (sqrt(denoma) * sqrt(denomb))
        C[i] *= hannwin[i]
        Csum += C[i]
    
    # Update C values
    for i in range(harmsamples):
        C[i] -= Csum / harmsamples
        C[i] *= hannwin[i]
    
    # Perform FFT on C values
    Cfft = np.abs(fft(C)) ** 2
    
    # Update EHStmp and calculate the return value
    for k in range(1, harmsamples // 2):  # Start from 1 as per the C code
        if Cfft[k] > max_value:
            max_value = Cfft[k]
    
    EHStmp += max_value
    return (EHStmp * 1000.0 / n), EHStmp



def moddiff(Modtest, Modref, Etilderef, fC):
    ModDiff1 = 0
    ModDiff2 = 0
    TempWt = 0
    
    for k in range(bark):
        ModDiff1 += np.abs(Modtest[k] - Modref[k]) / (1.0 + Modref[k])        
        if Modtest[k] > Modref[k]:
            ModDiff2 += np.abs(Modtest[k] - Modref[k]) / (0.01 + Modref[k])
        else:
            ModDiff2 += 0.1 * np.abs(Modtest[k] - Modref[k]) / (0.01 + Modref[k])
            
        Pthres = np.power(10.0, 0.4 * 0.364 * np.power(fC[k] / 1000.0, -0.8))
        TempWt += Etilderef[k] / (Etilderef[k] + np.power(Pthres, 0.3) * 100.0)
    
    ModDiff1 *= 100.0 / bark
    ModDiff2 *= 100.0 / bark

    return ModDiff1, ModDiff2, TempWt

class ModDiffOut:
    def __init__(self, ModDiff1: float, ModDiff2: float, TempWt: float):
        self.ModDiff1 = ModDiff1
        self.ModDiff2 = ModDiff2
        self.TempWt = TempWt

    def __repr__(self):
        return f"ModDiffOut(ModDiff1={self.ModDiff1}, ModDiff2={self.ModDiff2}, TempWt={self.TempWt})"

# Define ModDiffIn class
class ModDiffIn:
    def __init__(self, L: int=4, num2: int=0, denom2: int=0, num3: int=0, denom3: int=0):
        self.L = L
        self.Lcount = 0
        self.mod = [0.0] * L
        self.modtmp = 0.0
        self.win = 0.0
        self.num2 = 0
        self.denom2 = 0
        self.num3 = 0
        self.denom3 = 0

    def __repr__(self):
        return f"ModDiffIn(L={self.L}, Lcount={self.Lcount}, mod={self.mod}, modtmp={self.modtmp}, win={self.win}, num2={self.num2}, denom2={self.denom2}, num3={self.num3}, denom3={self.denom3})"

# Define the function ModDiff1
def ModDiff1(in_vals: ModDiffOut, intmp: ModDiffIn, n: int):
    intmp.mod[intmp.Lcount] = in_vals.ModDiff1
    intmp.Lcount += 1
    if intmp.Lcount == intmp.L:
        intmp.Lcount = 0

    if n < intmp.L:
        return 0.0, intmp

    intmp.modtmp = 0.0
    for i in range(intmp.L):
        intmp.modtmp += sqrt(intmp.mod[i])

    intmp.modtmp /= intmp.L

    intmp.win += intmp.modtmp ** 4.0    
    winmoddiff = sqrt(intmp.win / (n - intmp.L + 1.0))
    return winmoddiff, intmp

def ModDiff2(mod_diff_out: ModDiffOut, mod_diff_in: ModDiffIn) -> (float, ModDiffIn):
    mod_diff_in.num2 += mod_diff_out.ModDiff1 * mod_diff_out.TempWt
    mod_diff_in.denom2 += mod_diff_out.TempWt
    return mod_diff_in.num2 / mod_diff_in.denom2, mod_diff_in

def ModDiff3(mod_diff_out: ModDiffOut, mod_diff_in: ModDiffIn) -> tuple:
    """
    Translated ModDiff3 function from C to Python
    
    Parameters:
    mod_diff_out (ModDiffOut): A namedtuple containing ModDiff1, ModDiff2, and TempWt values.
    mod_diff_in (ModDiffIn): A namedtuple containing num3 and denom3 for internal calculations.
    
    Returns:
    float: The result of ModDiff3 calculation.
    ModDiffIn: Updated namedtuple with new num3 and denom3 values.
    """
    # Extract values from namedtuple for calculations
    mod_diff1 = mod_diff_out.ModDiff1
    temp_wt = mod_diff_out.TempWt
    num3 = mod_diff_in.num3
    denom3 = mod_diff_in.denom3
    
    # Update num3 and denom3
    num3 += mod_diff1 * temp_wt
    denom3 += temp_wt
    
    # Calculate and return ModDiff3
    mod_diff3_result = num3 / denom3
    
    # Update ModDiffIn namedtuple
    mod_diff_in_new = ModDiffIn(num2=mod_diff_in.num2, denom2=mod_diff_in.denom2, num3=num3, denom3=denom3)
    
    return mod_diff3_result, mod_diff_in_new

def loudness(E, fC, bark, CONST=1.0):
    Ntot = 0.0

    for k in range(bark):
        s = 10 ** (-2.0 - 2.05 * np.arctan(fC[k] / 4000.0) - 0.75 * np.arctan((fC[k] / 1600.0) ** 2) / 10.0)
        Ethres = 10 ** (0.364 * (fC[k] / 1000.0) ** -0.8)
        N = CONST * (Ethres / (s * 10000.0)) ** 0.23 * ((1.0 - s + s * E[k] / Ethres) ** 0.23 - 1.0)
        
        if N > 0:
            Ntot += N

    Ntot *= 24.0 / bark
    return Ntot

class LevPatAdaptIn:
    def __init__(self, bark: int):
        self.Ptest = [0.0] * bark
        self.Pref = [0.0] * bark
        self.Rnum = [0.0] * bark
        self.Rdenom = [0.0] * bark
        self.PattCorrTest = [0.0] * bark
        self.PattCorrRef = [0.0] * bark

@dataclass
class LevPatAdaptOut:
    def __init__(self, Epref: np.ndarray, Eptest: np.ndarray):
        self.Epref = Epref
        self.Eptest = Eptest

    def __getitem__(self, key):
        if key == "Epref":
            return self.Epref
        elif key == "Eptest":
            return self.Eptest
        else:
            return KeyError


def levpatadapt(Etest: np.ndarray, Eref: np.ndarray, rate: int, 
                tmp: LevPatAdaptIn, hann: int, fC: float, 
                Tmin: float, T100: float) -> LevPatAdaptOut:
    
    M = 8
    M1 = M/2 - 1
    M2 = M/2
    bark = len(Etest)  # Assuming Etest and Eref have the same length as bark
    Epref = [0.0] * bark
    Eptest = [0.0] * bark
    Rtest = np.ones(bark)
    Rref = np.ones(bark)

    numlevcorr = 0.0
    denomlevcorr = 0.0
    
    for k in range(bark):
        T = Tmin + (100.0 / fC[k]) * (T100 - Tmin)
        a = math.exp(-hann / (2.0 * rate * T))
        
        tmp.Ptest[k] = tmp.Ptest[k] * a + (1 - a) * Etest[k]
        tmp.Pref[k] = tmp.Pref[k] * a + (1 - a) * Eref[k]
        
        numlevcorr += math.sqrt(tmp.Ptest[k] * tmp.Pref[k])
        denomlevcorr += tmp.Ptest[k]
    
    levcorr = (numlevcorr / denomlevcorr) ** 2
    
    for k in range(bark):
        Elref = Eref[k] / levcorr if levcorr > 1 else Eref[k]
        Eltest = Etest[k] * levcorr if levcorr <= 1 else Etest[k]
        
        tmp.Rnum[k] *= a
        tmp.Rdenom[k] *= a
        tmp.Rnum[k] += Elref * Eltest
        tmp.Rdenom[k] += Elref * Elref
        
        R = tmp.Rnum[k] / tmp.Rdenom[k] if tmp.Rdenom[k] != 0 else 1.0
        Rtest[k] = 1.0 / R if R >= 1.0 else 1.0
        Rref[k] = R if R < 1.0 else 1.0
        
        m1, m2 = min(M1, k), min(M2, bark - k - 1)
        m1 = int(m1)
        m2 = int(m2)
        pattcoefftest = sum(Rtest[k + i] for i in range(-m1, m2 + 1))
        pattcoeffref = sum(Rref[k + i] for i in range(-m1, m2 + 1))
        
        tmp.PattCorrTest[k] = a * tmp.PattCorrTest[k] + pattcoefftest * (1 - a) / (m1 + m2 + 1)
        tmp.PattCorrRef[k] = a * tmp.PattCorrRef[k] + pattcoeffref * (1 - a) / (m1 + m2 + 1)
        
        Epref[k] = Elref * tmp.PattCorrRef[k]
        Eptest[k] = Eltest * tmp.PattCorrTest[k]
        
    return LevPatAdaptOut(Epref=Epref, Eptest=Eptest)




def noiseloudness(Modtest, Modref, lev, nltmp, n, fC):
    
    THRESFAC0 = 0.15
    S0 = 0.5
    E0 = 1.0
    ALPHA = 1.5
    

    nl = 0
    for k in range(BARK):
        Pthres = 10**(0.4 * 0.364 * (fC[k] / 1000)**(-0.8))
        stest = THRESFAC0 * Modtest[k] + S0
        sref = THRESFAC0 * Modref[k] + S0

        if lev['Eptest'][k] == 0 and lev['Epref'][k] == 0:
            beta = 1.0
        elif lev['Epref'][k] == 0:
            beta = 0
        else:
            beta = np.exp(-ALPHA * (lev['Eptest'][k] - lev['Epref'][k]) / lev['Epref'][k])

        num = stest * lev['Eptest'][k] - sref * lev['Epref'][k]
        denom = Pthres + sref * lev['Epref'][k] * beta

        if num < 0:
            num = 0

        nl += (Pthres / (E0 * stest))**0.23 * ((1.0 + num / denom)**0.23 - 1.0)

    nl *= 24.0 / bark
    if nl < 0:
        nl = 0

    nltmp += nl ** 2
    return np.sqrt(nltmp / n), nltmp


s_f = lambda L: 5.95072*p(6.39468/L, 1.71332)+9.01033*p(10.0, -11.0)*p(L, 4.0)+5.05622*p(10.0, -6.0)*p(L, 3.0)-0.00102438*p(L, 2.0)+0.0550197*L-0.198719

def detprob(Etestch1, Erefch1, state, hann=HANN, bark=BARK):
    prod = 1.0
    Q = 0.0
    Qsum = state["QSum"]
    PMtmp = state["PMtmp"]
    Ptildetemp = state["Ptildetemp"]
    ndistorcedtmp = state["ndistorcedtmp"]

    for k in range(bark):
        Etildetestch1 = 10.0 * np.log10(Etestch1[k])
        Etilderefch1 = 10.0 * np.log10(Erefch1[k])
        L = 0.3 * max(Etilderefch1, Etildetestch1) + 0.7 * Etildetestch1
        s = s_f(L)
        e = Etilderefch1 - Etildetestch1
        b = 4.0 if Etilderefch1 > Etildetestch1 else 6.0
        a = p(10.0, np.log10(np.log10(2.0)) / b) / s
        pch1 = 1.0 - p(10.0, -p(a * e, b))
        qch1 = abs(e) / s

        pbin = pch1
        qbin = qch1
       

        prod *= (1.0 - pbin)
        Q += qbin

    P = 1.0 - prod
    if P > 0.5:
        Qsum += Q
        ndistorcedtmp += 1

    if ndistorcedtmp == 0:
        ADBb = 0
    elif Qsum > 0:
        ADBb = np.log10(Qsum / ndistorcedtmp)
    else:
        ADBb = -0.5

    c0 = p(0.9, hann / (2.0 * 1024.0))
    c1 = p(0.99, hann / (2.0 * 1024.0))  # Assuming C1 is not defined
    Ptildetemp = (1.0 - c0) * P + Ptildetemp * c0
    PMtmp = max(Ptildetemp, PMtmp * c1)

    return ADBb, PMtmp, Ptildetemp, Qsum, ndistorcedtmp

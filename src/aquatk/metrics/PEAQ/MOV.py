import math
from math import log10, sqrt, cos, pi
from scipy.fftpack import fft
from .utils import *
from . import utils
from .utils import HANN as hann


def bandwidth(processing,
              out=None):
    # Initialize variables
    if out is None:
        out = {"sumBandwidthRefb": 0,
               "sumBandwidthTestb": 0,
               "countref": 0,
               "counttest": 0,
               "BandwidthRefb": 0,
               "BandwidthTestb": 0}
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
    # Use countboundary (frames within data boundary), not overall count
    n = state["countboundary"]
    nmrtmp = state["nmrtmp"]
    PNoise = processing.ppnoise
    M = processing.Mref
    sum_val = np.sum(PNoise / M)
    sum_val /= BARK
    nmrtmp += sum_val
    return 10 * np.log10(nmrtmp / n), nmrtmp


def reldistframes(processing, state):
    # Use countboundary, not overall count
    n = state["countboundary"]
    reldisttmp = state.get("RelDistTmp", 0)
    PNoise = processing.ppnoise
    M = processing.Mref
    # Check if ANY band exceeds threshold (then increment by 1 and break)
    for k in range(BARK):
        if 10 * np.log10(PNoise[k] / M[k]) >= 1.5:
            reldisttmp += 1
            break
    return reldisttmp / n, reldisttmp


def energyth(test, ref, hann=2048, ENERGYLIMIT=8000):
    # Check the energy for the second half of the test signal
    sum_test = sum(test[hann // 2:] ** 2)
    if sum_test > ENERGYLIMIT:
        return 1

    # Check the energy for the second half of the reference signal
    sum_ref = sum(ref[hann // 2:] ** 2)
    if sum_ref > ENERGYLIMIT:
        return 1

    return 0


def harmstruct(processing, state, rate=48000, harmsamples=256):
    """Error Harmonic Structure calculation matching C implementation.

    Args:
        processing: Processing struct containing fftref and ffttest
        state: State dict containing EHStmp and countenergy
        rate: Sample rate in Hz
        harmsamples: Number of harmonic samples (p in C code), calculated as:
                    harmsamples = 1; while(harmsamples < (18000/rate)*(HANN/4)) harmsamples *= 2

    Note: C implementation has AVGHANN defined, which means:
    - First compute correlation C[i] WITHOUT Hann windowing
    - Then subtract mean from C
    - Finally apply Hann window

    Also: C implementation has SKIPFRAME defined, which skips frames
    with log(0) issues entirely (returns 0 and decrements n).

    Also: C implementation has GETMAX defined, which uses simple max
    search starting from index 1 (skipping DC component).
    """
    fftref = processing.fftref
    ffttest = processing.ffttest
    EHStmp = state["EHStmp"]
    n = state["countenergy"]  # Use countenergy, not count (only frames with enough energy)

    p = harmsamples  # Using harmsamples as p to match C code

    # Calculate F0 values (log of squared magnitude ratio)
    # F0[k] = log10(fftref[k]^2) - log10(ffttest[k]^2)
    # C code uses indices 0 to 2*p-2, need to ensure we don't go out of bounds
    max_k = min(2 * p - 1, len(fftref))
    F0 = np.zeros(2 * p - 1)

    # SKIPFRAME behavior: if any value is 0, skip the entire frame
    # C code: (*n)--; return 0;
    # We decrement countenergy (the caller will increment it back, so net effect is 0)
    # and return 0 for this frame's contribution
    for k in range(max_k):
        if fftref[k] == 0 or ffttest[k] == 0:
            state["countenergy"] -= 1  # Will be incremented back by caller
            return 0, EHStmp  # Return 0 for this frame, preserve EHStmp
        else:
            F0[k] = log10(fftref[k] ** 2) - log10(ffttest[k] ** 2)

    # Compute normalized correlation C[i] for i in [0, p)
    # With AVGHANN defined: compute C WITHOUT Hann window first, accumulate Csum
    C = np.zeros(p)
    hannwin = np.zeros(p)
    Csum = 0.0

    for i in range(p):
        num = 0.0
        denoma = 0.0
        denomb = 0.0
        for k in range(p):
            num += F0[k] * F0[i + k]
            denoma += F0[k] ** 2
            denomb += F0[i + k] ** 2

        hannwin[i] = 0.5 * sqrt(8.0 / 3.0) * (1.0 - cos(2.0 * pi * i / (p - 1.0)))

        denom = sqrt(denoma) * sqrt(denomb)
        if denom > 0:
            C[i] = num / denom
        else:
            C[i] = 0.0

        # With AVGHANN: do NOT apply hannwin here, just accumulate C for mean
        # (without AVGHANN it would be: C[i] *= hannwin[i])
        Csum += C[i]

    # With AVGHANN: subtract mean, THEN apply Hann window
    for i in range(p):
        C[i] -= Csum / p
        C[i] *= hannwin[i]  # Apply Hann window after mean subtraction

    # Perform FFT on C values
    Cfft_complex = fft(C)

    # Scale by 1/p and compute power spectrum (matching C code)
    Cfft = np.zeros(p // 2)
    for k in range(p // 2):
        re = Cfft_complex[k].real / p
        im = Cfft_complex[k].imag / p
        Cfft[k] = re ** 2 + im ** 2

    # With GETMAX defined: simple max search starting from index 1
    # (PATCH=1, so i=0+PATCH=1, then for(k=i+1;...) means k starts at 1)
    # Actually with GETMAX: i=0, then for(k=i+1;...) so k starts at 1
    max_value = 0.0
    for k in range(1, p // 2):
        if Cfft[k] > max_value:
            max_value = Cfft[k]

    EHStmp += max_value
    return (EHStmp * 1000.0 / n), EHStmp


def moddiff(Modtest, Modref, Etilderef, fC):
    """Modulation difference calculation matching MATLAB PQmovModDiffB.m

    MATLAB convention (from PQModPatt.m and PQmovModDiffB.m):
    - M(1,:) = reference modulation
    - M(2,:) = test modulation
    - Denominator uses M(1) = reference
    - negWt2B=0.1 applied when ref > test, full weight when test >= ref
    - ERavg = reference average energy (Eavg(1,:))
    """
    ModDiff1 = 0
    ModDiff2 = 0
    TempWt = 0

    # Internal noise threshold for weighting (e=0.3 exponent)
    # This matches MATLAB's Ete calculation in PQmovModDiffB.m
    e = 0.3

    for k in range(utils.BARK):
        # MATLAB: num1B = abs(M(1) - M(2))
        num1B = np.abs(Modref[k] - Modtest[k])

        # MATLAB: MD1B = num1B / (offset1B + M(1)) where offset1B=1.0
        # M(1) = reference modulation
        ModDiff1 += num1B / (1.0 + Modref[k])

        # MATLAB asymmetric weighting (negWt2B = 0.1):
        # if (M(1) > M(2)):  # ref > test
        #     num2B = negWt2B * num1B  # 0.1 multiplier
        # else:  # test >= ref
        #     num2B = num1B  # full weight
        if Modref[k] > Modtest[k]:
            num2B = 0.1 * num1B  # Reduced weight when ref > test
        else:
            num2B = num1B  # Full weight when test >= ref
        ModDiff2 += num2B / (0.01 + Modref[k])

        # Temporal weighting: Wt += ERavg(m) / (ERavg(m) + levWt * Ete(m))
        # ERavg = Fmem.Eavg(1,:) = reference average energy = Etilderef
        # levWt = 100
        # Ete = Et^e where Et = internal noise threshold
        Et = PQIntNoise_single(fC[k])  # Internal noise for this band
        Ete = np.power(Et, e)
        TempWt += Etilderef[k] / (Etilderef[k] + 100.0 * Ete)

    ModDiff1 *= 100.0 / utils.BARK
    ModDiff2 *= 100.0 / utils.BARK

    return ModDiff1, ModDiff2, TempWt


def PQIntNoise_single(fc):
    """Calculate internal noise threshold for a single frequency.

    From PQIntNoise.m (the simple version used in moddiff):
    INdB = 1.456 * (fc/1000)^(-0.8)
    EIN = 10^(INdB/10)
    """
    INdB = 1.456 * np.power(fc / 1000.0, -0.8)
    EIN = np.power(10.0, INdB / 10.0)
    return EIN


class ModDiffOut:
    def __init__(self, ModDiff1: float, ModDiff2: float, TempWt: float):
        self.ModDiff1 = ModDiff1
        self.ModDiff2 = ModDiff2
        self.TempWt = TempWt

    def __repr__(self):
        return f"ModDiffOut(ModDiff1={self.ModDiff1}, ModDiff2={self.ModDiff2}, TempWt={self.TempWt})"


# Define ModDiffIn class
class ModDiffIn:
    def __init__(self, L: int = 4, num2: int = 0, denom2: int = 0, num3: int = 0, denom3: int = 0):
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
    # Use ModDiff2 (asymmetric version) for AvgModDiff2b, not ModDiff1
    # This matches the C implementation: intmp->num3 += in.ModDiff2 * in.TempWt
    mod_diff_in.num3 += mod_diff_out.ModDiff2 * mod_diff_out.TempWt
    mod_diff_in.denom3 += mod_diff_out.TempWt

    # Calculate and return ModDiff3
    mod_diff3_result = mod_diff_in.num3 / mod_diff_in.denom3

    return mod_diff3_result, mod_diff_in


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
    M1 = M / 2 - 1
    M2 = M / 2
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
    """Noise loudness calculation matching MATLAB PQmovNLoudB.m

    Key formula: s = sum((Et/stest)^e * ((1 + a/b)^e - 1))
    where Et = internal noise threshold
    """
    THRESFAC0 = 0.15  # TF0 in MATLAB
    S0 = 0.5
    ALPHA = 1.5
    e = 0.23

    nl = 0
    for k in range(BARK):
        # Internal noise threshold Et (same formula as PQIntNoise.m)
        # INdB = 1.456 * (f/1000)^(-0.8); Et = 10^(INdB/10)
        INdB = 1.456 * (fC[k] / 1000) ** (-0.8)
        Et = 10 ** (INdB / 10)

        # Modulation-weighted scaling factors
        sref = THRESFAC0 * Modref[k] + S0
        stest = THRESFAC0 * Modtest[k] + S0

        # Beta calculation for masking
        if lev['Epref'][k] == 0:
            if lev['Eptest'][k] == 0:
                beta = 1.0
            else:
                beta = 0
        else:
            beta = np.exp(-ALPHA * (lev['Eptest'][k] - lev['Epref'][k]) / lev['Epref'][k])

        # MATLAB: a = max(stest * EP(2) - sref * EP(1), 0)
        # where EP(1)=ref, EP(2)=test
        a = max(stest * lev['Eptest'][k] - sref * lev['Epref'][k], 0)

        # MATLAB: b = Et + sref * EP(1) * beta
        b = Et + sref * lev['Epref'][k] * beta

        # MATLAB: s = s + (Et/stest)^e * ((1 + a/b)^e - 1)
        nl += (Et / stest) ** e * ((1.0 + a / b) ** e - 1.0)

    # Scale by 24/Nc
    nl *= 24.0 / BARK
    if nl < 0:
        nl = 0

    # RMS accumulation
    nltmp += nl ** 2
    return np.sqrt(nltmp / n), nltmp


s_f = lambda L: 5.95072 * p(6.39468 / L, 1.71332) + 9.01033 * p(10.0, -11.0) * p(L, 4.0) + 5.05622 * p(10.0, -6.0) * p(
    L, 3.0) - 0.00102438 * p(L, 2.0) + 0.0550197 * L - 0.198719


def detprob(Etestch1, Erefch1, state, hann=HANN, bark=BARK):
    """Detection probability calculation matching C implementation (detprob.c).

    Key differences from MATLAB version:
    - L calculation: Always L = 0.3*max(ref,test) + 0.7*test (C code logic)
    - q calculation: Uses integer truncation abs((int)e)/s (C code behavior)
    - C1 constant: Uses C1=1.0 when defined (as in peaqb-fast common.h)
    """
    prod = 1.0
    Q = 0.0
    Qsum = state["QSum"]
    PMtmp = state["PMtmp"]
    Ptildetemp = state["Ptildetemp"]
    ndistorcedtmp = state["ndistorcedtmp"]

    for k in range(bark):
        Etildetestch1 = 10.0 * np.log10(Etestch1[k])  # Test in dB
        Etilderefch1 = 10.0 * np.log10(Erefch1[k])    # Ref in dB

        # C code L calculation (detprob.c:43-48):
        # if(Etilderefch1 > Etildetestch1) L = 0.3*Etilderefch1;
        # else L = 0.3*Etildetestch1;
        # L += 0.7*Etildetestch1;
        if Etilderefch1 > Etildetestch1:
            L = 0.3 * Etilderefch1
        else:
            L = 0.3 * Etildetestch1
        L += 0.7 * Etildetestch1

        if L > 0:
            s = s_f(L)
        else:
            s = p(10.0, 30.0)

        # e = ref - test (same as C code)
        e = Etilderefch1 - Etildetestch1

        # b depends on sign of difference (C code detprob.c:59-62)
        if Etilderefch1 > Etildetestch1:
            b = 4.0
        else:
            b = 6.0

        a = p(10.0, np.log10(np.log10(2.0)) / b) / s
        pch1 = 1.0 - p(10.0, -p(a * e, b))

        # C code uses integer truncation: qch1 = abs((int)e)/s
        qch1 = abs(int(e)) / s

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
    # C code: C1 is defined as 1.0 in common.h, so c1 = C1 = 1.0
    c1 = 1.0
    Ptildetemp = (1.0 - c0) * P + Ptildetemp * c0
    # C code: if(*Ptildetmp > (*PMtmp)*c1) *PMtmp = *Ptildetmp; else *PMtmp = (*PMtmp)*c1;
    if Ptildetemp > PMtmp * c1:
        PMtmp = Ptildetemp
    else:
        PMtmp = PMtmp * c1

    return ADBb, PMtmp, Ptildetemp, Qsum, ndistorcedtmp

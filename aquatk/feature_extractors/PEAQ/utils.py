from dataclasses import dataclass
import numpy as np

B = lambda f: 7 * np.arcsinh(f / 650.0)
BI = lambda z: 650 * np.sinh(z / 7.0)


NORM=11361.301063573899
FREQADAP=23.4375

def safe_pow(x, y):
    if x == 0 and y <= 0:
        if y == 0:
            return 1  # 0^0 in C is 1
        else:
            return float('inf')  # 0^-x in C sets errno to EDOM and returns a domain error
    return x ** y

p = lambda x, y: safe_pow(x, y)

HANN = 2048
BARK = 109
THRESHOLDDELAY = 0.050
AVERAGINGDELAY = 0.5

module = lambda x: abs(x)

@dataclass
class Processing:
    fftref: np.ndarray
    ffttest: np.ndarray
    ffteref: np.ndarray
    fftetest: np.ndarray
    fnoise: np.ndarray
    pptest: np.ndarray
    ppref: np.ndarray
    ppnoise: np.ndarray
    E2test: np.ndarray
    E2ref: np.ndarray
    Etest: np.ndarray
    Eref: np.ndarray
    Mref: np.ndarray
    Modtest: np.ndarray
    Modref: np.ndarray

    

@dataclass
class Moddiffin:
    win: float
    Lcount: int
    modtmp: float
    mod: np.ndarray
    num2: float
    denom2: float
    num3: float
    denom3: float

@dataclass
class Moddiffout:
    ModDiff1: float
    ModDiff2: float
    TempWt: float
    
@dataclass
class Levpatadaptout:
    Epref: np.ndarray
    Eptest: np.ndarray

@dataclass
class Levpatadaptin:
    Ptest: np.ndarray
    Pref: np.ndarray
    PattCorrTest: np.ndarray
    PattCorrRef: np.ndarray
    Rnum: np.ndarray
    Rdenom: np.ndarray

  
@dataclass
class MOV:
    WinModDiff1b: float = 0
    AvgModDiff1b: float = 0
    AvgModDiff2b: float = 0
    RmsNoiseLoudb: float = 0
    BandwidthRefb: float = 0 
    BandwidthTestb: float = 0
    TotalNMRb: float = 0 
    RelDistFramesb: float = 0
    ADBb: float = 0 
    MFPDb: float = 0 
    EHSb: float = 0

    def __repr__(self):
        return f'''MOV:
        WinModDiff1b: {self.WinModDiff1b}
        AvgModDiff1b: {self.AvgModDiff1b}
        AvgModDiff2b: {self.AvgModDiff2b}
        RmsNoiseLoudb: {self.RmsNoiseLoudb}
        BandwidthRefb: {self.BandwidthRefb}
        BandwidthTestb: {self.BandwidthTestb}
        TotalNMRb: {self.TotalNMRb}
        RelDistFramesb: {self.RelDistFramesb}
        ADBb: {self.ADBb}
        MFPDb: {self.MFPDb}
        EHSb: {self.EHSb}
        '''

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def to_dict(self):
        return { k: getattr(self, k) for k in self.__dict__}

    

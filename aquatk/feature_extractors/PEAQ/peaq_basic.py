import argparse
from do_spreading import *
from time_spreading import *
from fft_ear_model import *
from utils import *
from group_into_bands import * 
from create_bark import * 
from modulation import *
import soundfile as sf
from soundfile import SoundFile
from threshold import *
import numpy as np
from MOV import *
from scipy.io import wavfile
from wavfile_utils import *

def process_audio_block(ch1ref, ch1test, rate=16000, hann=HANN, lpref=92, lptest=92):
    fC, fL, fR = calculate_bark_bands(80, 18000)
    ffteref, fftref = earmodelfft(ch1ref, 1, lpref, hann)
    fftetest, ffttest = earmodelfft(ch1test, 1, lptest, hann)
    
    ppref = critbandgroup(ffteref, rate, hann, (fC, fL, fR))
    ppref = AddIntNoise(ppref, fC)
    
    pptest = critbandgroup(fftetest, rate, hann, (fC, fL, fR))
    pptest = AddIntNoise(pptest, fC)
    
    fnoise = np.abs(ffteref) - np.abs(fftetest)
    ppnoise = critbandgroup(fnoise, rate, hann, (fC, fL, fR))
    
    E2test = spreading(pptest, fC)
    E2ref = spreading(ppref, fC)
    
    Etest, Etmptest = time_spreading(E2test, rate, fC)
    Eref, Etmpref = time_spreading(E2ref, rate, fC)
    
    Mref = threshold(Eref)
    test_modulationIn = ModulationIn(e2_tmp=Etmptest, etilde_tmp=Etmpref, eder_tmp=np.zeros_like(Etmptest))
    ref_modulationIn = ModulationIn(e2_tmp=Etmpref, etilde_tmp=Etmpref, eder_tmp=np.zeros_like(Etmpref))
    Modtest = modulation(E2test, rate, in_struct=test_modulationIn, fC=fC)
    Modref = modulation(E2ref, rate, in_struct=ref_modulationIn, fC=fC)
    
    return Processing(fftref, ffttest, ffteref, fftetest, fnoise, 
                      pptest, ppref, ppnoise, E2test, E2ref, 
                      Etest, Eref, Mref, Modtest, Modref)

def extract_movs_from_block(processed: Processing, blocks:np.ndarray, accumulation_values:dict, **kwargs):
    # compute bandwidth
    pass
    


if __name__ == "__main__":
    ref_blocks = []
    test_blocks = []
    ref_file = SoundFile("ref.wav")
    test_file = SoundFile("test.wav")
    ref_subtype = ref_file.subtype
    test_subtype = test_file.subtype
    ref_rate = ref_file.samplerate
    test_rate = test_file.samplerate
    
    ref_blocks = read_wav_blocks("ref.wav")
    test_blocks = read_wav_blocks("test.wav")

    ref_blocks = np.array(ref_blocks)
    test_blocks = np.array(test_blocks)

    blocked_processed_outs = []
    
    # pass into process_audio_block
    for i in range(len(ref_blocks)):
        blocked_processed_outs.append(process_audio_block(ref_blocks[i], test_blocks[i]))
    
    harm_samples = 1
    while harm_samples < (18000/test_rate) * (HANN/4):
        harm_samples *= 2
    
    print("harm_samples: ", harm_samples)
    delaytime1 = np.ceil(THRESHOLDDELAY*test_rate*2/HANN)
    delaytime2 = np.ceil(AVERAGINGDELAY*test_rate*2/HANN)

    print("delaytime1: ", delaytime1)
    print("delaytime2: ", delaytime2)
    
    print(energyth(ref_blocks[1], test_blocks[1]))
    # what is the size of fftref[0]
    print(f"{blocked_processed_outs[0].ffteref.shape}")

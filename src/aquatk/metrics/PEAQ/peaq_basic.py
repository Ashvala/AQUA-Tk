import argparse
from .do_spreading import *
from .time_spreading import *
from .fft_ear_model import *
from .utils import *
from .group_into_bands import *
from .create_bark import *
from .modulation import *
import soundfile as sf
from soundfile import SoundFile
from .threshold import *
import numpy as np
from .MOV import *
from scipy.io import wavfile
from .wavfile_utils import *
from .neural import *
from tqdm import tqdm


def boundary(ch1ref, ch1test, rate, hann=HANN, BOUNDLIMIT=200, BOUNDWIN=5):
    for k in range(0, hann - BOUNDWIN + 1):
        ch1t = sum(abs(ch1test[k : k + BOUNDWIN])) > BOUNDLIMIT
        ch1r = sum(abs(ch1ref[k : k + BOUNDWIN])) > BOUNDLIMIT

        if ch1t or ch1r:
            return True

    return False


def process_audio_block(
    ch1ref,
    ch1test,
    rate=16000,
    hann=HANN,
    lpref=92,
    lptest=92,
    state={},
    boundflag=0,
    test_rate=16000,
):
    fC, fL, fR = calculate_bark_bands(80, 18000)
    harm_samples = 1
    while harm_samples < (18000 / test_rate) * (HANN / 4):
        harm_samples *= 2

    delaytime1 = np.ceil(THRESHOLDDELAY * test_rate * 2 / HANN)
    delaytime2 = np.ceil(AVERAGINGDELAY * test_rate * 2 / HANN)

    ffteref, fftref = earmodelfft(ch1ref, 1, lpref, hann)
    fftetest, ffttest = earmodelfft(ch1test, 1, lptest, hann)
    ppref = critbandgroup(ffteref, rate, hann=HANN, bark_table=(fC, fL, fR))
    ppref = AddIntNoise(ppref, fC)

    pptest = critbandgroup(fftetest, rate, hann=HANN, bark_table=(fC, fL, fR))
    pptest = AddIntNoise(pptest, fC)

    fnoise = np.abs(ffteref) - np.abs(fftetest)
    ppnoise = critbandgroup(fnoise, rate, hann, bark_table=(fC, fL, fR))

    E2test = spreading(pptest, fC)
    E2ref = spreading(ppref, fC)

    # Use persistent Etmp state for time spreading (critical for temporal filtering)
    Etest, Etmptest = time_spreading_with_state(E2test, rate, fC, state.get("Etmptest"))
    Eref, Etmpref = time_spreading_with_state(E2ref, rate, fC, state.get("Etmpref"))
    state["Etmptest"] = Etmptest
    state["Etmpref"] = Etmpref

    Mref = threshold(Eref)

    # Use persistent modulation state (critical for temporal filtering)
    if state.get("test_modulationIn") is None:
        state["test_modulationIn"] = ModulationIn(
            e2_tmp=np.zeros(BARK), etilde_tmp=np.zeros(BARK), eder_tmp=np.zeros(BARK)
        )
    if state.get("ref_modulationIn") is None:
        state["ref_modulationIn"] = ModulationIn(
            e2_tmp=np.zeros(BARK), etilde_tmp=np.zeros(BARK), eder_tmp=np.zeros(BARK)
        )

    Modtest, state["test_modulationIn"] = modulation(
        E2test, rate, in_struct=state["test_modulationIn"], fC=fC
    )
    Modref, state["ref_modulationIn"] = modulation(
        E2ref, rate, in_struct=state["ref_modulationIn"], fC=fC
    )

    proc = Processing(
        fftref,
        ffttest,
        ffteref,
        fftetest,
        fnoise,
        pptest,
        ppref,
        ppnoise,
        E2test,
        E2ref,
        Etest,
        Eref,
        Mref,
        Modtest,
        Modref,
    )

    # compute MOVs for this block
    movs = MOV()

    if boundflag:
        bandwidth_out = bandwidth(proc, out=state)
        movs.update(
            BandwidthRefb=bandwidth_out["BandwidthRefb"],
            BandwidthTestb=bandwidth_out["BandwidthTestb"],
        )
        # FIX: Actually update state with bandwidth values (map() was never consumed)
        for key in bandwidth_out.keys():
            state[key] = bandwidth_out[key]
        NMR_out, nmrtmp = nmr(proc, state)
        state["nmrtmp"] = nmrtmp
        movs.update(TotalNMRb=NMR_out)
        reldist, reldisttmp = reldistframes(proc, state)
        state["RelDistTmp"] = reldisttmp
        state["RelDistFramesb"] = reldist
        movs.update(RelDistFramesb=reldist)
        state["countboundary"] += 1
        if energyth(test=ch1test, ref=ch1ref):
            hs, ehstmp = harmstruct(
                proc,
                state,
                rate=rate,
                harmsamples=harm_samples,
            )
            movs.update(EHSb=hs)
            state["EHStmp"] = ehstmp  # Update state with accumulated EHStmp
            state["countenergy"] += 1

    if state["count"] > delaytime2:
        # Use persistent ref_modulationIn from state
        mouts = moddiff(Modtest, Modref, state["ref_modulationIn"].Etildetmp, fC)
        ModDiff = ModDiffOut(mouts[0], mouts[1], mouts[2])

        # FIX: Use persistent ModDiffInVars from state instead of recreating
        if state.get("ModDiffInVars") is None:
            state["ModDiffInVars"] = ModDiffIn()

        o = ModDiff1(ModDiff, state["ModDiffInVars"], state["count"] - delaytime2)
        movs.update(WinModDiff1b=o[0])
        state["ModDiffInVars"] = o[1]
        md2, state["ModDiffInVars"] = ModDiff2(ModDiff, state["ModDiffInVars"])
        movs.update(AvgModDiff1b=md2)
        md3, state["ModDiffInVars"] = ModDiff3(ModDiff, state["ModDiffInVars"])
        movs.update(AvgModDiff2b=md3)

        Ntotaltest = loudness(Etest, fC=fC, bark=BARK)
        Ntotalref = loudness(Eref, fC=fC, bark=BARK)
        noise = 0
        if Ntotaltest > 0.1 or Ntotalref > 0.1:
            noise = 1
        if noise and state["internal_count"] <= delaytime1:
            state["internal_count"] += 1
            state["loudcounter"] += 1
        else:
            # FIX: Use persistent LevPatAdaptIn from state
            if state.get("LevPatAdaptIn") is None:
                state["LevPatAdaptIn"] = LevPatAdaptIn(bark=BARK)
            state["LevPatAdaptIn"].Ptest = pptest
            state["LevPatAdaptIn"].Pref = ppref
            # C code: levpatadapt.h defines T100=0.05
            lev = levpatadapt(
                Etest,
                Eref,
                rate,
                hann=HANN,
                fC=fC,
                Tmin=0.008,
                T100=0.05,
                tmp=state["LevPatAdaptIn"],
            )
            # FIX: Use persistent nltmp from state instead of resetting to 0
            n_l, nltmp = noiseloudness(
                Modtest,
                Modref,
                lev,
                state.get("nltmp", 0),
                state["count"] - delaytime2 - state["loudcounter"],
                fC=fC,
            )
            state["nltmp"] = nltmp
            movs.update(RmsNoiseLoudb=n_l)

    ADBb, PMtemp, Ptildetemp, Qsum, ndistorcedtmp = detprob(Etest, Eref, state)
    state.update(
        {"PMtemp": PMtemp, "Ptildetemp": Ptildetemp, "ndistorcedtmp": ndistorcedtmp, "QSum": Qsum}
    )
    movs.update(ADBb=ADBb)
    movs.update(MFPDb=PMtemp)

    # Update MOVs with accumulated state values (like C implementation)
    # These values persist across frames in the C code's global 'processed' struct
    # Calculate TotalNMRb using the proper formula: 10*log10(nmrtmp / countboundary)
    nmrtmp = state.get("nmrtmp", 0)
    countboundary = max(state.get("countboundary", 1), 1)
    if nmrtmp > 0:
        total_nmr = 10 * np.log10(nmrtmp / countboundary)
    else:
        total_nmr = 0

    # Calculate EHSb from accumulated state values
    # EHSb = EHStmp * 1000.0 / countenergy (like C implementation)
    ehstmp = state.get("EHStmp", 0)
    countenergy = max(state.get("countenergy", 1), 1)
    ehsb = (ehstmp * 1000.0 / countenergy) if ehstmp > 0 else 0

    movs.update(
        BandwidthRefb=state.get("BandwidthRefb", 0),
        BandwidthTestb=state.get("BandwidthTestb", 0),
        TotalNMRb=total_nmr,
        RelDistFramesb=state.get("RelDistFramesb", 0),
        EHSb=ehsb,
    )

    # convert MOVs to a dict
    mov_dict = movs.to_dict()
    neural_out = neural(mov_dict)
    DI = neural_out["DI"]
    ODG = neural_out["ODG"]
    return proc, state, movs, DI, ODG


def init_state():
    return {
        "countboundary": 1,
        "RelDistFramesb": 0,
        "nmrtmp": 0,
        "countenergy": 1,
        "EHStmp": 0,
        "nltmp": 0,
        "noise": 0,
        "internal_count": 0,
        "loudcounter": 0,
        "sumBandwidthRefb": 0,
        "sumBandwidthTestb": 0,
        "countref": 0,
        "counttest": 0,
        "BandwidthRefb": 0,
        "BandwidthTestb": 0,
        "count": 1,
        "RelDistTmp": 0,
        "CFFTtemp": 0,
        "Ptildetemp": 0,
        "PMtmp": 0,
        "QSum": 0,
        "ndistorcedtmp": 0,
        "Cffttmp": np.zeros(1024),
        # Persistent state for ModDiff calculations
        "ModDiffInVars": None,
        # Persistent state for level-pattern adaptation
        "LevPatAdaptIn": None,
        # Persistent state for modulation calculations
        "test_modulationIn": None,
        "ref_modulationIn": None,
        # Persistent state for time spreading
        "Etmptest": None,
        "Etmpref": None,
    }


def process_audio_files(ref_filename: str, test_filename: str):
    def read_and_process_soundfile(filename: str) -> np.ndarray:
        sound_file = SoundFile(filename)
        sound_blocks = np.array(read_wav_blocks(filename))
        # mono
        if sound_blocks.shape[-1] == 1:
            sound_blocks = np.squeeze(sound_blocks, axis=-1)
        # stereo
        elif sound_blocks.shape[-1] == 2:
            sound_blocks = sound_blocks.mean(axis=-1)
        return sound_blocks, sound_file.samplerate

    ref_blocks, ref_rate = read_and_process_soundfile(ref_filename)
    test_blocks, test_rate = read_and_process_soundfile(test_filename)

    processed_blocks_list = []
    state = init_state()
    num_blocks = len(ref_blocks)
    result = {"MOV_list": [], "DI_list": [], "ODG_list": []}

    for i in tqdm(range(num_blocks)):
        boundaryflag = boundary(ref_blocks[i], test_blocks[i], ref_rate)
        proc, state, movs, di, odg = process_audio_block(
            ref_blocks[i],
            test_blocks[i],
            rate=ref_rate,
            state=state,
            boundflag=boundaryflag,
            test_rate=test_rate,
        )
        result["MOV_list"].append(movs)
        processed_blocks_list.append(proc)
        result["DI_list"].append(di)
        result["ODG_list"].append(odg)
        state["count"] += 1

    avg_DI = np.mean(result["DI_list"])
    avg_ODG = np.mean(result["ODG_list"])
    print(f"Distortion Index: {avg_DI}, Objective Difference Grade: {avg_ODG}")
    return result


if __name__ == "__main__":
    process_audio_files("ref.wav", "test.wav")

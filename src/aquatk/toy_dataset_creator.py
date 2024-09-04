import numpy as np
import os 
import soundfile as sf


sine_signal = lambda fr, d, fs: np.sin(2*np.pi*np.arange(fs*d)*fr/fs).astype(np.float32)
noise = lambda d, fs: np.random.normal(0, 1, int(d*fs)).astype(np.float32)

def create_paths(): 
    TOY_PATH = "./toy_dataset"

    if not os.path.exists(TOY_PATH):
        os.mkdir(TOY_PATH)

    # create a reference subdir and a noise subdir
    ref_path = os.path.join(TOY_PATH, "ref")
    noise_path = os.path.join(TOY_PATH, "noise")

    if not os.path.exists(ref_path):
        os.mkdir(ref_path)

    if not os.path.exists(noise_path):
        os.mkdir(noise_path)
        
def generate_sine_waves(): 
    n_files = 10 
    fs = 16000
    d = 4
    frs = np.random.randint(100, 1000, n_files)
    waves = np.zeros((n_files, fs*d))
    for i, fr in enumerate(frs): 
        waves[i] = sine_signal(fr, d, fs)
    
    return waves



if __name__ == "__main__": 
    print("[TOY] Creating toy dataset paths")
    create_paths()
    print("[TOY] Generating sine waves")
    waves = generate_sine_waves()
    print("[TOY] Writing sine waves to disk")
    for i, wave in enumerate(waves): 
        path = os.path.join("./toy_dataset/ref", f"ref_{i}.wav")
        sf.write(path, wave, 16000)
    
    print("[TOY] Creating noise")
    noise = noise(4, 16000)
    print("[TOY] Writing noise to disk")
    for i in range(10): 
        path = os.path.join("./toy_dataset/noise", f"noise_{i}.wav")
        sf.write(path, waves[i] + noise, 16000)

    
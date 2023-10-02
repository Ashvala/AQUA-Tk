from .extractor import Extractor
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os
from joblib import Parallel, delayed
from panns_inference import AudioTagging

class PANNs(Extractor):
    def __init__(self, verbose=True, batch_size=32, n_jobs=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.at = AudioTagging(checkpoint_path=None, device="cuda")
        
    def get_embeddings(self, x, sr=16000):                
        if isinstance(x, list):
            for result in Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_batch)(x[i:i+self.batch_size]) 
                for i in tqdm(range(0, len(x), self.batch_size), disable=(not self.verbose))
            ):
                yield from result  # yielding results from the current batch
            
        elif isinstance(x, str):
            all_files = [os.path.join(x, fname) for fname in os.listdir(x)]
            for result in Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_batch_files)(all_files[i:i+self.batch_size]) 
                for i in tqdm(range(0, len(all_files), self.batch_size), disable=(not self.verbose))
            ):
                yield from result              
        else:
            raise AttributeError
    
    def _process_batch(self, batch):        
        embeddings = []        
        for audio in batch:
            audio = audio[None, :]
            _, embd = self.at.inference(audio)
            embeddings.append(embd)
        return embeddings
    
    def _process_batch_files(self, batch_files):        
        embeddings = []        
        for file in batch_files:
            audio, _ = sf.read(file)
            audio = audio[None, :]
            _, embd = self.at.inference(audio)
            embeddings.append(embd)
        return embeddings

    def cleanup(self):
        return 0

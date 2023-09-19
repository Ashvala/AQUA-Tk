from extractor import Extractor
from tqdm import tqdm
import soundfile as sf


class JukeMIR(Extractor):
    def __init__(self, verbose=False, layers=[36], meanpool=True, fp16=True, offset=0, fp16_out=True):
        super(JukeMIR, self).__init__()
        self.verbose = verbose
        self.layers = layers
        self.meanpool = meanpool
        self.fp16 = fp16
        self.offset = offset
        self.fp16_out = fp16_out

    def get_embeddings(self, x, sr=16000):
        import jukemirlib
        emb_list = []
        if isinstance(x, list):
            try:
                for audio, sr in tqdm(x, disable=(not self.verbose)):
                    embd = jukemirlib.extract(audio, layers=self.layers, meanpool=self.meanpool, fp16=self.fp16,
                                              offset=self.offset, fp16_out=self.fp16_out)
                    emb_list.append(embd)
            except Exception as e:
                print("[Frechet Audio Distance] get_embeddings throw an exception: {}".format(str(e)))
        elif isinstance(x, str):
            try:
                for fname in tqdm(os.listdir(x), disable=(not self.verbose)):
                    audio = jukemirlib.load_audio(os.path.join(x, fname))
                    embd = jukemirlib.extract(audio, layers=self.layers, meanpool=self.meanpool, fp16=self.fp16,
                                              offset=self.offset, fp16_out=self.fp16_out)
                    emb_list.append(embd)
            except Exception as e:
                print("[Frechet Audio Distance] get_embeddings throw an exception: {}".format(str(e)))
        else:
            raise AttributeError
        return np.concatenate(emb_list, axis=0)

    def cleanup(self):
        # clear the gpu memory
        import torch
        torch.cuda.empty_cache()
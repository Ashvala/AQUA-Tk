from .extractor import Extractor
from tqdm import tqdm
import soundfile as sf


class PANNs(Extractor):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def get_embeddings(self, x, sr=16000):
        from panns_inference import AudioTagging
        emb_list = []
        at = AudioTagging(checkpoint_path=None, device="cuda")
        if isinstance(x, list):
            try:
                for audio in tqdm(x, disable=(not self.verbose)):
                    audio = audio[None, :]
                    _, embd = at.inference(audio)

                    emb_list.append(embd)
            except Exception as e:
                print("[Embedding Extractor] get_embeddings throw an exception: {}".format(str(e)))
        elif isinstance(x, str):
            try:
                for fname in tqdm(os.listdir(x), disable=(not self.verbose)):
                    audio, sr = sf.read(os.path.join(x, fname))
                    audio = audio[None, :]
                    _, embd = at.inference(audio)
                    emb_list.append(embd)
            except Exception as e:
                print("[Embedding Extractor] get_embeddings throw an exception: {}".format(str(e)))
        else:
            raise AttributeError
        return np.concatenate(emb_list, axis=0)

    def cleanup(self):
        return 0

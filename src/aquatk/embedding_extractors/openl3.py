from .extractor import Extractor
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os

class OpenL3(Extractor):
    def __init__(self, emb_size=512, content_type="music", verbose=True):
        super(OpenL3, self).__init__()
        assert emb_size in [512, 6144]
        self.emb_size = emb_size
        assert content_type in ["music", "env"]
        self.content_type = content_type
        self.verbose = verbose

    def _load_audio(self, fname, sr=16000):
        audio, sr = sf.read(fname, )
        return audio

    def get_embeddings(self, x, sr=16000):
        emb_list = []
        print("[OpenL3] Extracting embeddings...")
        import openl3
        if isinstance(x, list):
            # this implies that the input is a list of audio signals
            try:
                for audio in tqdm(x, disable=(not self.verbose)):
                    embd = openl3.get_audio_embedding(audio, sr, content_type=self.content_type,
                                                      embedding_size=self.emb_size)
                    emb_list.append(embd)
            except Exception as e:
                print("[Embedding Extraction] get_embeddings throw an exception: {}".format(str(e)))
        elif isinstance(x, str):
            # this implies that the input is a directory of audio files
            try:
                audio_list = []
                for fname in tqdm(os.listdir(x), disable=(not self.verbose)):
                    # we load audios into an array
                    audio_list.append(self._load_audio(os.path.join(x, fname), sr))
                # batch audio_list into groups of 32
                batch_size = 32
                for i in tqdm(range(0, len(audio_list), batch_size), disable=(not self.verbose)):
                    audios = audio_list[i:i + batch_size]
                    embs, _ = openl3.get_audio_embedding(audios, sr, embedding_size=self.emb_size,
                                                         content_type=self.content_type, batch_size=32, verbose=False)

                    # add to the list of embeddings without making it a list of lists so that it's of size (n, embedding_size)
                    emb_list.extend(embs)



            except Exception as e:
                print("[Frechet Audio Distance] get_embeddings throw an exception: {}".format(str(e)))
        else:
            raise AttributeError
        return np.concatenate(emb_list, axis=0)

    def cleanup(self):
        # ask tensorflow to cleanup
        import tensorflow as tf
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

from embedding_extractors import OpenL3, PANNs, VGGish
from functools import partial
from multiprocessing import Pool
from metrics import *
import os
import soundfile as sf
import librosa

class Task:
    def __init__(self, func, args, kwargs, name):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __call__(self):
        return self.func(*self.args, **self.kwargs)

    def execute(self):
        return self.func(*self.args, **self.kwargs)


class Pipeline:
    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self):
        for task in self.tasks:
            task.execute()

    def execute(self):
        for task in self.tasks:
            task.execute()


# multiprocess runner
class Runner:
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def run(self, tasks):
        if self.n_jobs == 1:
            for task in tasks:
                task.execute()
        else:
            pool = Pool(self.n_jobs)
            pool.map(lambda task: task.execute(), tasks)
            pool.close()
            pool.join()


def run_pipeline(pipeline, n_jobs=1):
    runner = Runner(n_jobs=n_jobs)
    runner.run(pipeline.tasks)


embedding_map = {
    "openl3": OpenL3,
    "panns": PANNs,
    "vggish": VGGish,
}

metric_map = {
    "cosine": cosine_similarity,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "fad": frechet_audio_distance,
    "kd": kernel_distance,
    "kld": kl_divergence
}

spec_dists = ["mse", "mae", "cosine"]
embedding_distances = ["fad", "kd"]


def get_embedding_extractor(name, **kwargs):
    return embedding_map[name](**kwargs)


def get_metric(name, **kwargs):
    return metric_map[name]


def generic_embedding_distance(embedding_extractor="panns", metric="fad", reference_dir=None, gen_dir=None, kwargs={}):
    embedding_extractor = get_embedding_extractor(embedding_extractor)
    metric = get_metric(metric)
    ref_feats = np.array(
        list(Task(embedding_extractor.get_embeddings, [reference_dir], kwargs, "extract_ref").execute()))
    ref_feats = ref_feats.reshape(-1, ref_feats.shape[2])
    gen_feats = np.array(list(Task(embedding_extractor.get_embeddings, [gen_dir], kwargs, "extract_gen").execute()))
    gen_feats = gen_feats.reshape(-1, gen_feats.shape[2])

    distance_task = Task(metric, [ref_feats, gen_feats], {}, "distance")
    distance = distance_task.execute()
    return distance


class JSONObject:
    def __init__(self, d):
        self.__dict__ = d
    def __repr__(self):
        return self.__dict__.__repr__()
    
def create_conf(original_config):
    smaller_jsons = []
    reference_dir = original_config.get("reference_dir", "")
    metrics = original_config.get("metrics", [])
    embeddings = original_config.get("embeddings", [])
    evaluate = original_config.get("evaluate", {})
    
    for key, value in evaluate.items():
        gen_dir = value.get("gen_dir", "")
        out = f"{key}_output.json"
        
        smaller_json = {
            "reference_dir": reference_dir,
            "sample_rate": 16000,  # Assuming a default value here
            "metrics": metrics,
            "embeddings": embeddings,
            "gen_dir": gen_dir,
            "out": out
        }
        smaller_jsons.append(smaller_json)
        
    return smaller_jsons

def get_melspec(file_path):
    mel_params = {
        "sr": 16000,
        "n_mels": 80,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 256 * 4,
        "fmin": 20.0,
        "fmax": 8000.0,
        "power": 1.0,
        "normalized": True,
    }
    audio, sr = sf.read(file_path)
    audio = audio[:64000]
    audio = np.clip(audio, -1, 1)

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=mel_params["n_mels"],
        n_fft=mel_params["n_fft"],
        hop_length=mel_params["hop_length"],
        win_length=mel_params["win_length"],
        fmin=mel_params["fmin"],
        fmax=mel_params["fmax"],
        power=mel_params["power"],
    )

    spec = 20 * np.log10(np.clip(spec, a_min=1e-5, a_max=np.inf)) - 20
    spec = np.clip((spec + 100) / 100, 0.0, 1.0)
    return spec

def get_melspecs(dir_path): 
    files = os.listdir(dir_path)
    for file in files: 
        yield get_melspec(os.path.join(dir_path, file))

def generic_spectral_dist(metric="mse", reference_dir=None, gen_dir=None, kwargs={}):
    metric = get_metric(metric)
    ref_feats = np.array(
        list(Task(get_melspecs, [reference_dir], kwargs, "extract_ref").execute()))
    # print(ref_feats.shape)
    # ref_feats = ref_feats.reshape(-1, ref_feats.shape[2])
    gen_feats = np.array(list(Task(get_melspecs, [gen_dir], kwargs, "extract_gen").execute()))
    # print(gen_feats.shape)
    # gen_feats = gen_feats.reshape(-1, gen_feats.shape[2])

    distance_task = Task(metric, [ref_feats, gen_feats], {}, "distance")
    distance = distance_task.execute()
    return distance


def config_parser(cfg_json):
    cfgs = create_conf(cfg_json)
    return [JSONObject(cfg) for cfg in cfgs]        

def run(configuration, st=False):
    # print configuration
    if not st: 
        print(f'''
        Configuration:
        reference_dir: {configuration.reference_dir}\n
        gen_dir: {configuration.gen_dir}\n
        output: {configuration.out}\n
        metrics: {configuration.metrics}\n
        embeddings to extract: {configuration.embeddings}\n
        ''')
    else: 
        st.write(f'''
        Configuration:
        reference_dir: {configuration.reference_dir}\n
        gen_dir: {configuration.gen_dir}\n
        output: {configuration.out}\n
        metrics: {configuration.metrics}\n
        embeddings to extract: {configuration.embeddings}\n
        ''') 
    metric_outs = []
    for metric in configuration.metrics:   
        if metric in spec_dists:
            out = generic_spectral_dist(metric=metric, reference_dir=configuration.reference_dir, gen_dir=configuration.gen_dir)    
            if not st: 
                print(f"{metric} {out}")
            else: 
                st.write(f"{metric} {out}")
        if metric in embedding_distances: 
            for emb in configuration.embeddings:
                out = generic_embedding_distance(embedding_extractor=emb, metric=metric, reference_dir=configuration.reference_dir, gen_dir=configuration.gen_dir)
                if not st: 
                    print(f"{metric} {out}")
                else: 
                    st.write(f"{metric} {emb} {out}")
                metric_outs.append({"metric": metric, "embedding": emb, "output": out})
        elif metric == "PEAQ":
            print("PEAQ is unstable. Use PEAQ through the peaq_basic command in the PEAQ folder.")

    return metric_outs     
from embedding_extractors import OpenL3, PANNs
from functools import partial
from multiprocessing import Pool
from metrics import * 

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
    "panns": PANNs
}

metric_map = {
    "cosine": cosine_similarity,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "fad": frechet_audio_distance,
    "kd": kernel_distance,
    "kld": kl_divergence
}


embedding_distances = ["fad", "kd"]

def get_embedding_extractor(name, **kwargs):
    return embedding_map[name](**kwargs)

def get_metric(name, **kwargs):
    return metric_map[name]

def generic_embedding_distance(embedding_extractor="panns", metric="fad", reference_dir=None, gen_dir=None, kwargs={}):
    embedding_extractor = get_embedding_extractor(embedding_extractor)
    metric = get_metric(metric)
    ref_feats = np.array(list(Task(embedding_extractor.get_embeddings, [reference_dir], kwargs, "extract_ref").execute()))
    ref_feats = ref_feats.reshape(-1, ref_feats.shape[2])    
    gen_feats = np.array(list(Task(embedding_extractor.get_embeddings, [gen_dir], kwargs, "extract_gen").execute()))
    gen_feats = gen_feats.reshape(-1, gen_feats.shape[2])
    
    distance_task = Task(metric, [ref_feats, gen_feats], {}, "distance")
    distance = distance_task.execute()
    return distance

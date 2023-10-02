import streamlit.web.bootstrap
import argparse
from functools import partial
from runner import Task, Pipeline, run_pipeline, embedding_map, generic_embedding_distance, embedding_distances
from joblib import Parallel, delayed
from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument("--web", action="store_true", help="Run in web mode")
parser.add_argument("--config", type=str, help="Path to config file")


args = parser.parse_args()

if args.web:
    streamlit.web.bootstrap.run("app.py", '', [], flag_options={})
    # block

        
    
    
# don't accept any other args if web is true
config = args.config

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


def config_parser(cfg_json):
    cfgs = create_conf(cfg_json)
    return [JSONObject(cfg) for cfg in cfgs]        

def run(configuration):
    # print configuration
    print(f'''
    Configuration:
    reference_dir: {configuration.reference_dir}
    gen_dir: {configuration.gen_dir}
    output: {configuration.out}
    metrics: {configuration.metrics}
    embeddings to extract: {configuration.embeddings}   
    ''')
    metric_outs = []
    for metric in configuration.metrics:        
        if metric in embedding_distances:
            for emb in configuration.embeddings:
                out = generic_embedding_distance(embedding_extractor=emb, metric=metric, reference_dir=configuration.reference_dir, gen_dir=configuration.gen_dir)
                print(out)
                metric_outs.append({"metric": metric, "embedding": emb, "output": out})
        else:
            pass

    with open(configuration.out, "w") as f:
        import json
        json.dump(metric_outs, f)

    
            
    

if config:
    import json
    with open(config) as f:
        cfg = json.load(f)
    cfgs = config_parser(cfg)
    for cfg in cfgs:
        run(cfg)
        



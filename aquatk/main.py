import streamlit.web.bootstrap
import argparse
from functools import partial
from runner import *
from joblib import Parallel, delayed
from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument("--web", action="store_true", help="Run in web mode")
parser.add_argument("--config", type=str, help="Path to config file")


args = parser.parse_args()

# # don't accept any other args if web is true

if args.web:
    streamlit.web.bootstrap.run("app.py", '', [], flag_options={})
    # block


config = args.configa
if config:
    import json
    with open(config) as f:
        cfg = json.load(f)
    cfgs = config_parser(cfg)
    for cfg in cfgs:
        run(cfg)
        



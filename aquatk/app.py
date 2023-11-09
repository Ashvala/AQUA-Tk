import streamlit as st
from runner import * 
st.title("AQUA*Tk*")

# set font to JetBrainsMono
st.markdown(
    """
    <style>
    *{
        font-family: 'JetBrains Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Options")

dir1 = st.sidebar.text_input("Reference directory")
dir2 = st.sidebar.text_input("Generated directory")

metrics = st.sidebar.multiselect("Select metrics", ["KL Divergence", "MSE", "MAE", "FAD", "KD", "PEAQ"])
embeddings = st.sidebar.multiselect("Select neural embeddings", ["PANNs", "OpenL3", "VGGish", "JukeMIR"])

metric_map = { 
    "KL Divergence": "kd", 
    "FAD": "fad", 
    "MSE": "mse", 
    "MAE": "mae",    
}

test_table = [{"embedding": "panns", "metric": "fad", "output": 0.1}, {"embedding": "", "metric": "fad", "output": 0.1}, {"embedding": "vggish", "metric": "fad", "output": 0.1}]

def create_conf(dirs, metrics, embeddings):
    reference_dir = dirs[0]
    sr = 16000
    metrics = metrics
    embeddings = embeddings
    evaluate = { f"network_{i}": { "gen_dir": dirs[i] } for i in range(1, len(dirs)) }    
    return {
        "reference_dir": reference_dir,
        "sr": sr,
        "metrics": metrics,
        "embeddings": embeddings,
        "evaluate": evaluate
    }

if st.sidebar.button("Evaluate!"):
    import json
    with open("config.json", "w") as f:
        json.dump(create_conf([dir1, dir2], metrics, embeddings), f)
    st.write("Running evaluation...")

#    st.write("Done!")
    with open("example_config.json") as f:
        output = json.load(f)
    cfg = config_parser(output)
    st.write(cfg)
    cfg_outs = []
    for cfg in cfg:
        cfg_outs.append(run(cfg, st=st))    
    # compile table from array of json objects.
    st.write(cfg_outs)
    metric_table = {}
    for cfg in cfg_outs:
        for metric in cfg.metrics:
            if metric not in metric_table:
                metric_table[metric] = []
            metric_table[metric].append(cfg.metrics[metric])

    st.write(metric_table)
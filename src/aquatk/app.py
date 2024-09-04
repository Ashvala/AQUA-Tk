import streamlit as st
from runner import * 
import pandas as pd
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

metrics = st.sidebar.multiselect("Select metrics", ["KL Divergence", "MSE", "MAE", "FAD", "KD"])
embeddings = st.sidebar.multiselect("Select neural embeddings", ["PANNs", "OpenL3", "VGGish", "JukeMIR"])

metric_map = { 
    "KL Divergence": "kd", 
    "FAD": "fad", 
    "MSE": "mse", 
    "MAE": "mae",    
}


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
    # st.write(cfg)
    cfg_outs = []
    for cfg in cfg:
        cfg_outs.append(run(cfg, st=st))    
    # compile table from array of json objects.
    # st.write(cfg_outs)
    metric_table = {}
    for i in cfg_outs:
        for j in i:  
            st.write(j["metric"])
            if j["metric"] not in metric_table: 
                metric_table[j["metric"]] = {}
            metric_table[j["metric"]][j["embedding"]] = j["output"]
    st.dataframe(pd.DataFrame(metric_table))

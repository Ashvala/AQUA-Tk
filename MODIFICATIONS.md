## MODIFICATIONS IN THIS FORK

### Goal
Make PEAQ usable as a pip import + runnable on audio data instead of just files

### Files

aquatk/__init__.py
aquatk/metrics/PEAQ/MOV.py
aquatk/metrics/PEAQ/__init__.py
aquatk/metrics/PEAQ/do_spreading.py
aquatk/metrics/PEAQ/fft_ear_model.py
aquatk/metrics/PEAQ/group_into_bands.py
aquatk/metrics/PEAQ/modulation.py
aquatk/metrics/PEAQ/peaq_basic.py
aquatk/metrics/PEAQ/threshold.py
aquatk/metrics/PEAQ/time_spreading.py
aquatk/metrics/__init__.py
setup.py
MODIFICATIONS.md

### Modifications
described in MODIFICATIONS.md

#### aquatk/metrics/PEAQ
all files in here modified for relative import for local imports, ex 
`from utils import` -> `from .utils import *`
peaq_basic.py, removed tqdm import and usage (loading bar)

#### __init__.py
all lines added to expose PEAQ functionality at the pip install level

aquatk/metrics/PEAQ/__init__.py: added, from . import peaq_basic
aquatk/metrics/__init__.py: from . import PEAQ
aquatk/__init__.py: from . import metrics

#### setup.py
switched packages=['aquatk'] to packages=find_packages() so pip install will pick up PEAQ

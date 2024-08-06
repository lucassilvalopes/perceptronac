
import pandas as pd

from complexity.glch_utils import get_trained_networks_history

tree_str = \
"""
032_010_010_001 !032_020_010_001 032_010_020_001 
032_020_010_001 !032_040_010_001 032_020_020_001 
032_040_010_001 032_080_010_001 !032_040_020_001 
032_040_020_001 032_080_020_001 !032_040_040_001 
032_040_040_001 032_080_040_001 032_040_080_001 
032_080_010_001 032_160_010_001 !032_080_020_001 
032_080_020_001 032_160_020_001 !032_080_040_001 
032_080_040_001 032_160_040_001 032_080_080_001 
032_160_010_001 032_320_010_001 !032_160_020_001 
032_160_020_001 032_320_020_001 !032_160_040_001 
032_160_040_001 032_320_040_001 032_160_080_001 
032_320_020_001 !032_640_020_001 032_320_040_001 
032_640_020_001 032_640_020_001 !032_640_040_001 
032_640_040_001 032_640_040_001 !032_640_080_001 
032_640_080_001 032_640_080_001 !032_640_160_001 
032_640_160_001 032_640_160_001 !032_640_320_001 
032_640_320_001 032_640_320_001 !032_640_640_001 
"""


data = pd.read_csv(
    "/home/lucas/Documents/perceptronac/complexity/data/rate-noisy-joules-time-params_hx-10-20-40-80-160-320-640.csv"
).set_index("topology")

hist = get_trained_networks_history(data,tree_str)
# print(hist.index)
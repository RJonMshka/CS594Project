import sys
import time
import os

from Ensemble.preprocessing import *
from config import *
from models import *
from dataPlotting import *

# default window sizes
rebalance_window = 63
validation_window = 63

window_size = {
    "half": 10,
    "one": 21,
    "three": 63,
    "default": 63
}


if len(sys.argv) >=1 and sys.argv[0] in window_size:
    rebalance_window = sys.argv[0]
    validation_window = sys.argv[0]

data = preprocess_data()
data = add_turbulence(data)

print(data.head())
print(data.size)

# 2015/10/01 is the date that validation starts
# 2016/01/01 is the date that real trading starts
# unique_trade_date needs to start from 2015/10/01 for validation purpose
unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()

model_used, a2c_list, ppo_list, ddpg_list = run_ensemble(
    df = data,
    unique_trade_date = unique_trade_date,
    rebalance_window = rebalance_window,
    validation_window = validation_window
)


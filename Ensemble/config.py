import datetime
import os

DATA_FILE = "Data/dow_30_2009_2020.csv"
now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)



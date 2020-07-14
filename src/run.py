import argparse
import logging
import warnings

try:
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from tqdm import tqdm
except:
	raise ModuleNotFoundError("Some modules could not be found. Try installing the `requirements.txt`")

from unsupervised_model import UnsupervisedModel

if __name__ == "__main__":
	data = pd.read_csv("../tp2_training_dataset.csv", header=None)

import argparse
import glob, os

try:
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from tqdm import tqdm
	import yaml
except:
	raise ModuleNotFoundError("Some modules could not be found. Try installing the `requirements.txt`")

from unsupervised_model import UnsupervisedModel
from utils import *

if __name__ == "__main__":
	data = pd.read_csv("../tp2_training_dataset.csv", header=None).to_numpy()
	config = yaml.load(open("./config.yml"), Loader=yaml.FullLoader)

	model = UnsupervisedModel(data, data.shape[-1], config["output"],
							lr=config["lr"], 
							algorithm=config["algorithm"],
							normalize=True,
							normal_params=(config["normal_params"]["mean"], config["normal_params"]["var"]))

	train = True
	model_name = config["model_name"]+"_"+config["algorithm"]
	for f in glob.glob("*.npy"):
		if model_name + ".npy" == f:
			train = False
			break
	
	if train:
		print(model)
		model.train()
		model.save_model(model_name)
	else:
		model.load_model(model_name)
		plot_data(model, data)
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

def plot_data(model, data, lims=(-100, 100)):
    # fig = plt.figure(figsize=(10, 7))
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # xyz = fig.add_subplot(111, projection='3d')
    xyz1 = fig.add_subplot(1, 3, 1, projection='3d')
    xyz1.set_xlim( -100, 100)
    xyz1.set_ylim( -100, 100)
    xyz1.set_zlim( -100, 100)

    xyz1.set_xlabel("Componente 0")
    xyz1.set_ylabel("Componente 1")
    xyz1.set_zlabel("Componente 2")
    xyz1.set_title("Componentes 0, 1 y 2")

    x, y, z = 0, 1, 2
    
    for t in data:
        prediction = model.predict(t)
        xyz1.scatter(prediction[x], prediction[y], prediction[z])

    xyz2 = fig.add_subplot(1, 3, 2, projection='3d')
    xyz2.set_xlabel("Componente 3")
    xyz2.set_ylabel("Componente 4")
    xyz2.set_zlabel("Componente 5")
    xyz2.set_title("Componentes 3, 4 y 5")

    x, y, z = 3, 4, 5
    
    for t in data:
        prediction = model.predict(t)
        xyz2.scatter(prediction[x], prediction[y], prediction[z])

    xyz3 = fig.add_subplot(1, 3, 3, projection='3d')
    xyz3.set_xlabel("Componente 6")
    xyz3.set_ylabel("Componente 7")
    xyz3.set_zlabel("Componente 8")
    xyz3.set_title("Componentes 6, 7 y 8")

    x, y, z = 6, 7, 8
    
    for t in data:
        prediction = model.predict(t)
        xyz3.scatter(prediction[x], prediction[y], prediction[z])

    plt.show()

if __name__ == "__main__":
	data = pd.read_csv("../tp2_training_dataset.csv", header=None).to_numpy()
	config = yaml.load(open("./config.yml"), Loader=yaml.FullLoader)

	model = UnsupervisedModel(data, data.shape[-1], config["output"],
							lr=config["lr"], 
							algorithm=config["algorithm"],
							normalize=True)

	train = True
	for f in glob.glob("*.npy"):
		if config["model_name"] + ".npy" == f:
			train = False
			break
	
	if train:
		model.train()
		model.save_model(config["model_name"])
	else:
		model.load_model(config["model_name"])
		plot_data(model, data)
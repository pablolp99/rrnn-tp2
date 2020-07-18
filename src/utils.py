import matplotlib.pyplot as plt
import numpy as np

from unsupervised_model import UnsupervisedModel

def plot_data(model, dataset, label):
    prediction = []
    for t in range(len(dataset)):
        pred = np.array([label[t]] + model.predict(dataset[t]).tolist())
        prediction.append(pred)
    
    prediction = np.array(prediction)
    
    fig = plt.figure(figsize=plt.figaspect(0.15))
    # xyz = fig.add_subplot(111, projection='3d')
    xyz1 = fig.add_subplot(1, 3, 1, projection='3d')

    xyz1.set_xlabel("Componente 1")
    xyz1.set_ylabel("Componente 2")
    xyz1.set_zlabel("Componente 3")
    xyz1.set_title("Componentes 1, 2 y 3")

    xyz1.scatter3D(prediction[:, 1], prediction[:, 2], prediction[:, 3], s=3, c=prediction[:, 0], alpha=0.7)

    xyz2 = fig.add_subplot(1, 3, 2, projection='3d')
    xyz2.set_xlabel("Componente 4")
    xyz2.set_ylabel("Componente 5")
    xyz2.set_zlabel("Componente 6")
    xyz2.set_title("Componentes 4, 5 y 6")

    xyz2.scatter3D(prediction[:, 4], prediction[:, 5], prediction[:, 6], s=3, c=prediction[:, 0], alpha=0.7)

    xyz3 = fig.add_subplot(1, 3, 3, projection='3d')
    xyz3.set_xlabel("Componente 7")
    xyz3.set_ylabel("Componente 8")
    xyz3.set_zlabel("Componente 9")
    xyz3.set_title("Componentes 7, 8 y 9")

    xyz3.scatter3D(prediction[:, 7], prediction[:, 8], prediction[:, 9], s=3, c=prediction[:, 0], alpha=0.7)

    plt.show()
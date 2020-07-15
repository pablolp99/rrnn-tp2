import matplotlib.pyplot as plt

from unsupervised_model import UnsupervisedModel

def plot_data(model, data, lims=(-100, 100)):
    # fig = plt.figure(figsize=(10, 7))
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # xyz = fig.add_subplot(111, projection='3d')
    xyz1 = fig.add_subplot(1, 3, 1, projection='3d')

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
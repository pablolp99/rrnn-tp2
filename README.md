# Trabajo Practico Nº2 - Redes Neuronales
## Aprendizaje No Supervisado

### Instruccion de ejecucion

---
#### Instalacion
Este modulo requiere de Python 3 y tener el paquete `tkinter`. Para verifica la instalacion del python ejecutar `python3 --version`, esto no deberia dar errores y deberia indicar una version del mismo. Lo ideal es tener de la version >=3.6. Para instalar `tkinter` (en sistemas Unix/BSD) utilizar `apt-get install python3-tk`

##### Pasos recomendados
Primero se crea un entorno virtual de python (Requiere `virtualenv`) y se accedera al mismo

`~/path/to/dir$ virtualenv py_env_name`

`$ source py_env_name/bin/activate`

A continuacion se instalan las librerias necesarias

`(py_env_name) $ pip install -r requirements.txt`

---
#### Configuracion
La configuracion del codigo se encuentra en el archivo `src/config.yml`, en el se encuentran las opciones de configuracion. La estructura es la siguiente

- 'force_train': hace que se fuerze el entrenamiento cada vez que se ejecute. `true` o `false`.
- 'algorithm': es el algoritmo/optimizacion que usara el modelo. Puede ser `oja_gen` o `sanger`.
- 'output': es la cantidad de neuronas de salida, es decir `M`. Default del tp es 9
- 'lr': es el learning rate que usara el modelo.
- 'model_name': nombre de como se guardara el modelo.
- 'plot_error': muestra, una vez finalizada la ejecucion, un grafico de la evolucion de la ortogonalidad del modelo. `true` o `false`
- 'normal_params': es donde se especifican los parametros con los cuales se crea la matriz `W`. Tanto como la media como la varianza

---
#### Ejecucion

El script que se debe de ejecutar es `run.py`. La primera vez que este se ejecute, este entrenara el modelo con los datos de entrenamiento, y guardara el modelo resultante. La segunda ejecucion, si la configuracion es la misma, este procedera a realizar un grafico con la activacion de los datos.

`(py_env_name) $ python run.py`

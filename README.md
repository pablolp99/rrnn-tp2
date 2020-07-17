# Trabajo Practico Nº2 - Redes Neuronales
## Aprendizaje No Supervisado

### Instruccion de ejecucion

---
#### Instalacion
Primero se crea un entorno virtual de python (Requiere `virtualenv`) y se accedera al mismo

`~/path/to/dir$ virtualenv py_env_name`

`$ source py_env_name/bin/activate`

A continuacion se instalan las librerias necesarias

`(py_env_name) $ pip install -r requirements.txt`

---
#### Configuracion
La configuracion del codigo se encuentra en el archivo `src/config.yml`, en el se encuentran las opciones de configuracion. La estructura es la siguiente

- 'algorithm' es el algoritmo/optimizacion que usara el modelo. Puede ser 'oja_gen' o 'sanger'
- 'output' es la cantidad de neuronas de salida, es decir `M`
- 'lr' es el learning rate que usara el modelo
- 'model_name' es donde se guardara el modelo
- En 'normal_params' es donde se especifican los parametros con los cuales se crea la matriz `W`. Tanto como la media como la varianza

---
#### Ejecucion

El script que se debe de ejecutar es `run.py`. La primera vez que este se ejecute, este entrenara el modelo con los datos de entrenamiento, y guardara el modelo resultante. La segunda ejecucion, si la configuracion es la misma, este procedera a realizar un grafico con la activacion de los datos.

`(py_env_name) $ python run.py`

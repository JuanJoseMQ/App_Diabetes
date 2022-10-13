# App_Diabetes

Este repositorio contiene un pequeña aplicación diseñada con Streamlit. La finalidad de esta aplicacion es determinar, 
a traves de algoritmos de predicción y una serie de parámetros, la probabilidad de padecer diabetes.


Este repositorio cuenta con dos partes:

- El fichero app_diabetes.py, donde encontraremos tanto el diseño de la app, como los algoritmos de predicción.
- EL fichero logs.txt crea un registro de los usuarios y los resultados obtenidos, para poder analizar posteriormente como
  se esta comportando mi modelo y si hay cosas que podemos mejorar.
- En la carpeta datasets, tenemos un fichero csv que contiene los datos de entrenamiento del modelo de predicción.
  Contine las carácteristicas que pedimos como referencia de 520 personas, las cuales 320 padecen diabetes y 200 no.
  Este dataset fue obtenido a traves de la web de Kaggle.
  
Para utilizar la app, hay que ejecutar el app_diabetes.py desde la terminal de prompt_anaconda siguiendo estos pasos:

- 1º Si no tenemos instalada la libreria Streamlit --> pip install streamlit
- 2º Una vez instalado, accedemos a la carpeta donde reside el app_diabetes.py que vamos a ejecutar
- 3º Usamos el comando --> run streamlit app_diabetes.py
- 4º Esto nos abrirá una host en la web y ya podremos interactuar con la app

Introduce los párametros deseados y comprueba si deberias cuidar un poco mas tu salud, o por el contrario todo va bien.

Se agradece el apoyo.
Un saludo.
Juanjo.

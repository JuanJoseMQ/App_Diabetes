import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time
import logging

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


def get_data():
	""".
	"""
	LOGGER.info(f'Getting data...')
	return pd.read_csv('datasets/diabetes_dataset.csv')


def train_and_predict_model(df, test_size, selected_model, input_data):
	""".
	"""

	# Aplicamos LabelEncoder a las variables categóricas (tipo objeto)
	LOGGER.info(f'Starting data transformation')
	object_columns = df.select_dtypes(include = 'object').columns

	lbl_enc = LabelEncoder()
	for col in object_columns:
		df[col] = lbl_enc.fit_transform(df[col])

	# Partimos datos entre variables X y variable y
	X = df.drop(["class"], axis=1)
	y = df["class"]


	# Split data entre test y train
	X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=test_size)


	column_names = ["Modelo Utilizado","Accuracy","Precision","Recall","F1",
	"Total_Positivos","Total_Negativos", "Falsos_Positivos", "Falsos_Negativos"]

	output_struct = pd.DataFrame(columns = column_names)

	LOGGER.info(f'{dt.now()} Model selected = {selected_model}')
	if selected_model == 'Logistic Regression': model = LogisticRegression(max_iter=2000)
	if selected_model == 'Random Forest': model = RandomForestClassifier(n_estimators=100)

	# Entrenamiento
	start_time = time.time()
	model.fit(X_train,y_train)
	time_train = time.time() - start_time

	# Testeo
	start_time = time.time()
	y_pred = model.predict(X_test)
	time_test = time.time() - start_time

	# Metricas
	acc = metrics.accuracy_score(y_test, y_pred)*100
	prc = metrics.precision_score(y_test, y_pred)*100
	rec = metrics.recall_score(y_test, y_pred)*100
	f1 = metrics.f1_score(y_test, y_pred)*100

	# Matriz de confusión
	conf_mtrx = confusion_matrix(y_test,y_pred)
	tn, fp, fn, tp = conf_mtrx.ravel()

	# Predict input data
	model_predict = model.predict(input_data)
	model_prob = model.predict_proba(input_data)
	max_prob = model_prob.max(axis=1)[0]

	LOGGER.info(f'Model result = {model_predict}')
	LOGGER.info(f'Model probability = {max_prob}')
	LOGGER.info(f'Model accuracy = {acc}')

	label_predict = "Positivo" if model_predict[0] == 1 else "Negativo"

	stats_df = pd.DataFrame({
	  'Modelo': [selected_model],
	  "Resultado": [label_predict],
	  'Prob()': [max_prob],
	  'Accuracy': [acc],
	  "Precisión": [prc],
	  'Recall': [rec],
	  'F1': [f1],
	  "TP": [tp],
	  "TN": [tn],
	  "FP": [fp],
  	  "FN": [fn],
	  "Tiempo Entrenamiento": [time_train],
	  "Tiempo Testeo": [time_test]
	})

	return stats_df, conf_mtrx


# Solicitamos variables de entrada para predecir/inferir
st.sidebar.subheader("Los atributos a utilizar en la predicción son:")


# Recogemos los inputs a través de la aplicación
input_edad =  st.sidebar.number_input("Edad", min_value=20,max_value=65,step=1)
input_gnr =  st.sidebar.selectbox("Género:", ["Masculino","Femenino"])
input_polyuria =  st.sidebar.selectbox("Polyuria:",["No","Sí"])
input_polydip =  st.sidebar.selectbox("Polydipsia:",["No","Sí"])
input_peso =  st.sidebar.selectbox("Perdida repentina de peso:",["No","Sí"])
input_debilidad =  st.sidebar.selectbox("Debilidad:",["No","Sí"])
input_polyphagia =  st.sidebar.selectbox("Polyphagia:",["No","Sí"])
input_llaga =  st.sidebar.selectbox("Genital thrush:",["No","Sí"])
input_vista =  st.sidebar.selectbox("Emborronamiento visual:",["No","Sí"])
input_picaz = st.sidebar.selectbox("Picazón:",["No","Sí"])
input_irrit = st.sidebar.selectbox("Irritabilidad:",["No","Sí"])
input_heridas = st.sidebar.selectbox("Las heridas tardan en curarse:",["No","Sí"])
input_paralisis = st.sidebar.selectbox("Parálisis parcial:",["No","Sí"])
input_rig = st.sidebar.selectbox("Rigidez muscular:",["No","Sí"])
input_alop = st.sidebar.selectbox("Alopecia:",["No","Sí"])
input_obs = st.sidebar.selectbox("Obesidad:",["No","Sí"])

input_model = st.sidebar.selectbox("Seleccione el modelo de ML:",["Logistic Regression","Random Forest"])


# Preguntamos el tamaño del dataset de entrenamiento que se quiere utilizar
test_size = st.sidebar.slider(label = 'Tamaño del set de entrenamiento (%):',
                            min_value=15,
                            max_value=30,
                            value=20,
                            step=1)

# Botón para ordenar el entrenamiento y predicción
btn_predict = st.sidebar.button("Adelante! Quiero ver los resultados")


# Una pequeña firma / datos
st.sidebar.write('**Immune Institute**')
st.sidebar.write('**Máster en Data Science**')

st.title("Sistema de clasificación predictivo sobre diabetes")

# subtítulo
st.write ('Aquí podemos insertar una pequeña definición del trabajo realizado....')
st.write("Los datos utilizados han sido extraidos de:")
st.markdown("Dataset: https://www.kaggle.com/ishandutta/early-stage-diabetes-risk-prediction-dataset")

# Comprobamos si se ha solicitado predecir resultados
if btn_predict:

	to_predict_dict = {
	  'Edad': [input_edad],
	  'Genero': [input_gnr],
	  "Poliuria": [input_polyuria],
	  "Polidipsia": [input_polydip],
	  "Perda_Peso": [input_peso],
	  "Fraqueza": [input_debilidad],
	  "Polifagia": [input_polyphagia],
	  "Tordo_genital": [input_llaga],
  	  "Embacamento_visual": [input_vista],
	  "Coceira": [input_picaz],
	  "Irritabilidade": [input_irrit],
  	  "Demora_cura": [input_heridas],
	  "Paresia_parcial": [input_paralisis],
	  "Rigidez_muscular": [input_rig],
	  "Alopecia": [input_alop],
	  "Obesidade": [input_obs]
	}

	to_predict_df = pd.DataFrame(to_predict_dict)

	LOGGER.info(f"Configuración del paciente = {to_predict_dict}")

	to_predict_df = to_predict_df.replace('Masculino', 1)
	to_predict_df = to_predict_df.replace('Femenino', 0)
	to_predict_df = to_predict_df.replace('Sí', 1)
	to_predict_df = to_predict_df.replace('No', 0)

	# Cargamos datos de entrenamiento
	df = get_data()

	# Entrenamos y predecimos
	model_stats, conf_mtrx = train_and_predict_model(df, test_size, input_model, to_predict_df)

	# Información que mostrar por pantalla
	st.subheader("Resultados")
	sub_df = model_stats[['Modelo', 'Resultado', 'Prob()']]
	st.table(sub_df)

	st.subheader("Métricas")
	met_df = model_stats[['Modelo', 'Accuracy', 'Precisión','Recall','F1', "Tiempo Entrenamiento", "Tiempo Testeo"]]
	st.table(met_df)

	st.subheader("Matriz de confusión - set de entrenamiento")
	st.write(conf_mtrx)

	st.subheader("Sobre los datos...")
	st.write(f"Los datos en los que se basa este trabajo tienen un registro de {len(df)} filas")
	st.write("Teniendo mayor número de personas diábeticas en el histórico como podemos ver en el siguiente gráfico")
	freq = df['class'].value_counts()
	fig, ax = plt.subplots()
	ax = freq.plot(kind='bar',
	                figsize = (10,5),
	                rot = 0,
	                grid = False)
	st.pyplot(fig)

	# histograma
	datos_para_hist = df['Age']
	datos_para_hist.hist()
	plt.show()
	st.set_option('deprecation.showPyplotGlobalUse', False)


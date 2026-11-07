import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
# Descargamos sklearn : python -m pip install scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def ejecutar_motor_quant(umbral=60): 

    print("=================")
    print(" MACHINE LEARNING FOR FACTOR TIMING  ")
    print("=================")

    # El primer paso es obtener todo los datos historicos del NASDAQ100
    print(" 1. Descargando datos del NASDAQ100 desde Yahoo Finance")

    # Elegimos las empresas mas representativas de los lideres del mercado, que son las que componen el NASDAQ100
    tickers_nasdaq = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", 
       "GOOGL", "TSLA", "AVGO", "COST", "NFLX", 
      "AMD", "QCOM", "INTC", "TXN", "AMAT"
    ]   

    fecha_hoy = datetime.today().strftime('%Y-%m-%d')
    datos = yf.download(tickers_nasdaq, start='2019-01-01', end=fecha_hoy, auto_adjust=True)['Close']
    print(f'Datos descargados con exito para {len(tickers_nasdaq)} activos del Nasdaq100')

    print("\n2. Constuimos los features")
    rendimientos = np.log(datos / datos.shift(1)).dropna()
    features_list = []

    for ticker in tickers_nasdaq:
        df_tickers = pd.DataFrame(datos[ticker]).rename(columns={ticker: 'Precio'})
        df_tickers['Ret_Diario'] = np.log(df_tickers['Precio'] / df_tickers['Precio'].shift(1))

        # Hacemos las caracterisitcas predictivas adaptadas al comportamiento de alta velocidad del Nasdaq
        # - Anualizamos la volatilidad a 21 dias (1 mes)
        # - Momentum a 21 dias (1 mes)
        df_tickers['Volatilidad_21d'] = df_tickers['Ret_Diario'].rolling(window=21).std() * np.sqrt(252)
        df_tickers['Momentum_21d'] = df_tickers['Precio'].pct_change(periods=21)

        # Creamos el target, y definimos si en 21 dias sera positivo con : 1=Si, 0=No
        df_tickers['Ret_Futuro_21d'] = df_tickers['Precio'].shift(-21) / df_tickers['Precio'] - 1
        df_tickers['Target'] = np.where(df_tickers['Ret_Futuro_21d'] > 0, 1, 0)

        df_tickers['Ticker'] = ticker
        features_list.append(df_tickers.dropna())

    # Lo unimos todo en un data set
    dataset = pd.concat(features_list)

    # Definimos X y Y, X = entrada, y Y = salida

    X = dataset[['Volatilidad_21d', 'Momentum_21d']]
    y = dataset['Target']

    print(f'Dataset constuido con {len(dataset)} observaciones para entrenamiento.')

    print("\n3. Entrenando el modelo de Random Forest")
    # Hacemos una division cronologica estricta para evitar el sesgo de info futura
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

    # Configuramos el lo que clasificara el machine learnging

    modelo = RandomForestClassifier(n_estimators = 150, max_depth = 5, random_state = 42)
    modelo.fit(X_train, y_train)

    # Evaluamos la precision con datos que el algoritno no conoce
    predicciones = modelo.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    print(f'Modelo generado sobr eel Nasdaq:')
    print(f'Accuracy : {precision * 100 :.2f}%')

    # Ampliamos que factor tiene mas peso para predecir el comportamiento.

    importances = modelo.feature_importances_
    print('Importancia relativa de los factores en el Nasdaq')
    for feature, imp in zip(['Volatilidad_21d', 'Momentum_21d'], importances):
        print(f'{feature}: {imp * 100 :.2f}%')


    print("=================")
    print(" LIVE INFERENCE ")
    print("=================")

    senales_hoy = []

    for ticker in tickers_nasdaq:
        serie_precio = datos[ticker].dropna()
        ret_diario = np.log(serie_precio / serie_precio.shift(1))

        # Obtenemos los valores mas recientes
        vol_actual = ret_diario.rolling(window=21).std().iloc[-1] * np.sqrt(252)
        mom_actual = (serie_precio.iloc[-1] / serie_precio.shift(21).iloc[-1]) - 1

        # Preparamos en vector de entrada actual para el modelo que entrenamos
        X_live = pd.DataFrame([[vol_actual, mom_actual]], columns=['Volatilidad_21d', 'Momentum_21d'])

        # Prediccion y probabilidad de exito
        prediccion = modelo.predict(X_live)[0]
        probabilidad = modelo.predict_proba(X_live)[0][1]

        Estado = "BUY" if prediccion == 1 else "SELL"

        senales_hoy.append ({
            'Ticker': ticker,
            'Momentum_1M_%': round(mom_actual * 100, 2),
            'Probabilidad_Subir_%': round(probabilidad * 100, 2),
            'Senal_Quant': Estado
        })

    # Convertimos el DataFrame y ordenamos de mayor a menor la probabilidad de subir
    df_senales = pd.DataFrame(senales_hoy).sort_values(by='Probabilidad_Subir_%', ascending=False).set_index('Ticker')
    print(df_senales[['Momentum_1M_%', 'Probabilidad_Subir_%', 'Senal_Quant']])


    print("=================")
    print(" POSITION SIZING ")
    print("=================")

    # Filtramos los activos para saber cuales cumplen con el umbral de confianza
    activos_aptos = df_senales[df_senales['Probabilidad_Subir_%'] >= umbral]

    if len(activos_aptos) > 0:
        # Seleccionamos los mejores activos aptos para diversificar el riezgo
        top_portafolio = activos_aptos.head(3)

        # Hacemos un Equal Weighting para saber cuanto porcentaje del portafolio se asigna a cada activo
        peso_por_activo = 100 / len(top_portafolio)
        top_portafolio['Peso_Portafolio_%'] = round(peso_por_activo, 2)

        print(f' Portafolio optimizado. Se seleccionaro {len(top_portafolio)} para operar')
        print(top_portafolio[['Probabilidad_Subir_%', 'Momentum_1M_%', 'Peso_Portafolio_%']])
    else: 
        print('No hay activos que cumplan con el umbral de confianza para operar hoy.')
        print('Desicion del sistema: Mantenerse fuera del mercado y esperar la proxima oportunidad de trading.')

    # 1. ¡AQUÍ ESTABA EL DETALLE CLAVE! Devolvemos el DataFrame para que la app web lo reciba
    return df_senales


print("=================")
print(" PAGINA INTERACTIVA ")
print("=================")

# Configuramos la pagina web

st.set_page_config(
    page_title="Plataforma Quant: Factor Timing & ML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Plataforma Quant: Factor Timing & Machine Learning")
st.markdown('Sistema automatizado de seleccion de activos del **Nasdaq100** basado en *Random Forest*, Inferencia en Vivo y Gestión de Riesgo Estricta')

# Agregamos el control interactivo en la barra lateral del umbral de confianza.
st.sidebar.header("Parámetros de Riesgo")
umbral_usuario = st.sidebar.slider(
    "Umbral de Confianza Mínimo (%)",
    min_value=40,
    max_value=90,
    value=60,  # Valor por defecto
    step=1,
    help="Filtra los activos que tengan una probabilidad igual o superior a este porcentaje."
)


# Configuramos el boton interactivo principal
if st.button (' Ejecturar Rebalanceo y Analisis de Mercado', type = 'primary'):
    with st.spinner('Descargando datos institucionales y entrenando el modelo predictivo'):
        df_senales = ejecutar_motor_quant(umbral_usuario)

    st.success('Motor ejecutado con exito')

    # Hacemos la visualicacion de la tabla de señales generadas por el motor cuantitativo
    st.subheader('Señales de Trading Generadas por el Motor Cuantitativo')
    st.dataframe(df_senales, use_container_width=True)

    # Hacemos visible la parte donde se asigna el capital y el control de riesgo
    st.subheader('Asignacion de portafolio y control de riesgo')

    # El umbral de confianza estricto para considerar un activo apto para operar
    
    activos_aptos = df_senales[df_senales['Probabilidad_Subir_%'] >= umbral_usuario]

    if len(activos_aptos) > 0:
        top_portafolio = activos_aptos.head(3).copy()
        peso_por_activo = 100 / len(top_portafolio)
        top_portafolio['Peso_Portafolio_%'] = round(peso_por_activo, 2)

        col1, col2, col3 = st.columns(3)
        col1.metric('Activos Seleccionados', len(top_portafolio))  # 2. Corregido de top_portafolios a top_portafolio
        col2.metric('Estrategia de Asignacion', 'Equal Weighting')
        col3.metric('Estado del Mercado', 'Oportunidades de compra')

        st.table(top_portafolio[['Probabilidad_Subir_%', 'Momentum_1M_%', 'Peso_Portafolio_%']])
    else:
        col1, col2 = st.columns(2)
        col1.metric('Posicion de capital sugerida', 'No realizar Posiciones')
        col2.metric('Estado del Mercado', 'Defensivo, sin oportunidades de BUY')

        st.warning('Alerta de riezgo: No hay activos que cumplan con el Minimo del Umbral de Confianza \n\n\n **Desicion del sistema**: Mantenerse fuera del Mercado y no operar hasta que se cumpla el minimo del Umbral de Confianza')
else:
    st.info('Haz clic en el boton superior para correr el motor y generar las senales de hoy')

# PARA CORRER LA PAGINA : 1. CORRER EL CODGIO, 2. python -m streamlit run FTR4.py en la terminal.
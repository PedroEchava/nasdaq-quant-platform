# Nasdaq 100 Factor Timing & Machine Learning Quant Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Una plataforma cuantitativa e interactiva diseñada para la selección de activos del índice Nasdaq100. Combina modelos de Aprendizaje Automático (*Machine Learning*) con una interfaz en
Streamlit para la toma de decisiones basada en datos y una gestión estricta del riesgo en tiempo real.

---

Esta herramienta automatiza el proceso de análisis financiero mediante:
1. Extracción y procesamiento de datos de mercado en tiempo real.
2. Entrenamiento de un modelo predictivo para calcular la probabilidad de apreciación de los activos.
3. Control interactivo de riesgo que permite filtrar oportunidades de inversión dinámicamente.
4. Construcción y asignación de portafolios bajo la metodología de ponderación equitativa (*Equal Weighting*).

---

## Arquitectura Técnica y Funcionalidades

* **Modelo Predictivo (*Random Forest Classifier*):** Entrenado con datos históricos para estimar la probabilidad individual de subida de las empresas componentes del Nasdaq 100.
* **Ingeniería de Características (*Feature Engineering*):** 
  * Cálculo del momentum relativo a 1 mes (`Momentum_1M_%`).
  * Análisis de factores macro y métricas clave de comportamiento técnico.
* **Gestión de Riesgo Interactiva:** Integración de un *slider* en la barra lateral de Streamlit (`st.sidebar.slider`) para ajustar el umbral de confianza predictiva (ej. 60% por defecto).
* **Generación de Señales Quantitative (`BUY` / `SELL`):** Clasificación automática e inferencia en vivo sobre el universo del mercado.
* **Ponderación de Portafolio:** Filtro automático que rebalancea el capital únicamente entre los activos que superan el parámetro de confianza seleccionado.

---

## Tech Stack

* **Lenguaje:** Python 3.10+
* **Machine Learning & Procesamiento:** `scikit-learn`, `pandas`, `numpy`
* **Datos Financieros:** `yfinance`
* **Despliegue e Interfaz:** Streamlit

---

## Cómo Ejecutar el Proyecto Localmente

Si deseas clonar y ejecutar este proyecto localmente en tu computadora:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU-USUARIO/nasdaq-quant-platform.git](https://github.com/TU-USUARIO/nasdaq-quant-platform.git)
   cd nasdaq-quant-platform

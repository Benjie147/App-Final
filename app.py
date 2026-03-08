import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Churn:

    def __init__(self, df):
        self.df = df

    def dataset_info(self):
        return self.df.dtypes

    def missing_values(self):
        return self.df.isnull().sum()

    def numeric_variables(self):
        return self.df.select_dtypes(include=np.number).columns.tolist()

    def categorical_variables(self):
        return self.df.select_dtypes(exclude=np.number).columns.tolist()

    def descriptive_stats(self):
        return self.df.describe()
    
    st.sidebar.title("Menú")

menu = st.sidebar.radio(
    "Navegación",
    ["🏠 Home", "📑 Carga de Dataset", "🔎 EDA", "📌 Conclusiones"]
)

# HOME

if menu == "🏠 Home":

    st.title("💻Análisis Exploratorio de Churn - Telecom")

    st.markdown("""
    ### Descripción del Proyecto

    En este proyecto se analiza el comportamiento de los clientes de una empresa de telecomunicaciones
    con el objetivo de identificar posibles patrones relacionados con la cancelación del servicio, conocida como Churn.
    Para lograrlo, se realiza un análisis exploratorio de datos (EDA), el cual permite comprender 
    mejor la información disponible y observar qué factores podrían estar influyendo en la decisión
    de los clientes de dejar la empresa.            

    """)

    st.subheader("👤 Autor")
    linkedin_url = "https://www.linkedin.com/in/victor-benjamin-rivas-jauregui-832a5b315/" 
    st.markdown(
        f'<a href="{linkedin_url}" target="_blank">'
        '<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20"> Víctor Benjamín Rivas Jáuregui'
        '</a>',
        unsafe_allow_html=True
    )
    st.write("Especialización: Python for Analytics")
    st.write("Año: 2026")

    st.subheader("📑 Dataset")

    st.write("""
    El dataset contiene información sobre clientes de telecomunicaciones,
    incluyendo servicios contratados, cargos mensuales, tipo de contrato
    y si el cliente abandonó la empresa.
    """)

    st.subheader("🐍 Tecnologías utilizadas")

    st.write("""
    ✅ Python
    ✅ Pandas
    ✅ NumPy
    ✅ Matplotlib
    ✅ Seaborn
    ✅ Streamlit
    """)

# CARGA DE DATASET

elif menu == "📑 Carga de Dataset":

    st.title("📑 Carga del Dataset")

    uploaded_file = st.file_uploader("Sube el archivo CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset cargado correctamente")

        st.subheader("🔍 Vista previa")

        st.dataframe(df.head())

        st.subheader("📏 Dimensiones del dataset")

        st.write(f"Filas: {df.shape[0]}")
        st.write(f"Columnas: {df.shape[1]}")

        st.session_state["data"] = df
        

    else:

        st.warning("Por favor sube un archivo para continuar.")

# EDA

elif menu == "🔎 EDA":

    st.title("🔎 Análisis Exploratorio de Datos")

    if "data" not in st.session_state:
        st.warning("Primero debes cargar el dataset")
        st.stop()

    df = st.session_state["data"]

    analyzer = Churn(df)

    tabs = [
        "1.Info",
        "2.Clasif.",
        "3.Estadísticas",
        "4.Faltantes",
        "5.Vars #",
        "6.Vars Cat",
        "7y8 A. bivariado",
        "9.Parametros",
        "10.Insights"
    ]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(tabs)

    # 1. Info general del dataset.

    with tab1:

        st.subheader("Información del dataset")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Tipos de datos")
            st.write(analyzer.dataset_info())

        with col2:
            total_nulos = df.isnull().sum().sum()

            st.write(f"Total de valores nulos en el dataset: {total_nulos}")

            st.write(analyzer.missing_values())

    # 2. Clasificacion de variables

    with tab2:

        st.subheader("Clasificación de variables")

        num_vars = analyzer.numeric_variables()
        cat_vars = analyzer.categorical_variables()

        col1, col2 = st.columns(2)

        with col1:
            st.write("Variables Numéricas")
            st.write(f"Cantidad: {len(num_vars)}")
            st.write(num_vars)

        with col2:
            st.write("Variables Categóricas")
            st.write(f"Cantidad: {len(cat_vars)}")
            st.write(cat_vars)

    # 3. Estadisticas descriptivas

    with tab3:

        st.subheader("Estadísticas descriptivas")

        st.dataframe(analyzer.descriptive_stats())

    # Ítem 4: Valores faltantes

    with tab4:
        st.subheader("Análisis de valores faltantes")

        missing_counts = df.isnull().sum()
        cols_con_nulos = missing_counts[missing_counts > 0]

        if len(cols_con_nulos) > 0:
            st.write("Columnas con valores faltantes:")
            st.dataframe(cols_con_nulos)
            st.bar_chart(cols_con_nulos)
        else:
            st.write("No hay valores faltantes en el dataset.")    

    # 5.Distribución de variables numéricas

    with tab5:

        st.subheader("Distribución de variables numéricas")

        numeric_cols = analyzer.numeric_variables()

        variable = st.selectbox(
            "Selecciona una variable numérica",
            numeric_cols
        )

        fig, ax = plt.subplots()

        sns.histplot(df[variable], kde=True, ax=ax)

        st.pyplot(fig)
        st.markdown(f"""
        **Interpretación:**  
        El gráfico muestra cómo se distribuyen los valores de **{variable}**.  
        dónde se concentran los datos y si hay picos o dispersión.
        """)

    # ===============================
    # ITEM 6
    # ===============================
    with tab6:
        st.subheader("Variables categóricas")

        cat_var = st.selectbox(
            "Selecciona variable categórica",
            analyzer.categorical_variables()
        )

        fig, ax = plt.subplots()

        df[cat_var].value_counts().plot(kind="bar", ax=ax)

        st.pyplot(fig)

    # ===============================
    # ITEM 7 y 8
    # ===============================

    with tab7:
    # Crear dos columnas
        col1, col2 = st.columns(2)

    # --- Columna 1: Numérico vs Categórico ---
        with col1:
        # Gráfico 1: MonthlyCharges vs Churn
            st.subheader("Análisis Bivariado (numérico vs categórico)")
            fig, ax = plt.subplots()
            sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax)
            ax.set_title("MonthlyCharges vs Churn")
            st.pyplot(fig)

        # Gráfico 2: Tenure vs Churn
            fig, ax = plt.subplots()
            sns.boxplot(x="Churn", y="tenure", data=df, ax=ax)
            ax.set_title("Tenure vs Churn")
            st.pyplot(fig)

    # --- Columna 2: Categórico vs Categórico ---
        with col2:
        # Gráfico 3: Contract vs Churn
            st.subheader("Análisis Bivariado (categórico vs categórico)")
            fig, ax = plt.subplots()
            sns.countplot(x="Contract", hue="Churn", data=df, ax=ax)
            ax.set_title("Contract vs Churn")
            st.pyplot(fig)

        # Gráfico 4: InternetService vs Churn
            fig, ax = plt.subplots()
            sns.countplot(x="InternetService", hue="Churn", data=df, ax=ax)
            ax.set_title("InternetService vs Churn")
            st.pyplot(fig)

    with tab8:
        st.subheader("Análisis basado en parámetros seleccionados")
        st.write("Selecciona una variable numérica y una variable categórica para analizar su relación.")

    # Selección de variables
        num_var = st.selectbox("Variable numérica", analyzer.numeric_variables())
        cat_var = st.selectbox("Variable categórica", analyzer.categorical_variables())

    # Generar gráfico dinámico
        fig, ax = plt.subplots()
        sns.boxplot(x=cat_var, y=num_var, data=df, ax=ax)
        ax.set_title(f"{num_var} vs {cat_var}")
        st.pyplot(fig)

    # ===============================
    # Ítem 10
    # ===============================
    with tab9:
        st.subheader("💡Hallazgos clave")
        st.write("""
        - Los contratos a largo plazo son efectivos para retener clientes y reducir la fuga.
        - Ofrecer servicios extra, como seguridad o soporte técnico, ayuda a que los clientes se queden.
        - Los clientes que tienen contratos mensuales tienden a dejar el servicio con más frecuencia.
        - Los cargos mensuales altos parecen estar relacionados con una mayor probabilidad de abandono.
        - Los que llevan más tiempo con la empresa suelen quedarse más tiempo y cancelar menos.
        """)

# ===============================
# CONCLUSIONES
# ===============================

elif menu == "📌 Conclusiones":

    st.title("📌 Conclusiones del Análisis")

    st.write("""
    1. Los contratos a largo plazo son efectivos para retener clientes y reducir la fuga..

    2. Ofrecer servicios extra, como seguridad o soporte técnico, ayuda a que los clientes se queden.

    3. Los clientes que tienen contratos mensuales tienden a dejar el servicio con más frecuencia.

    4. Los cargos mensuales altos parecen estar relacionados con una mayor probabilidad de abandono.

    5. Los que llevan más tiempo con la empresa suelen quedarse más tiempo y cancelar menos.
    """)
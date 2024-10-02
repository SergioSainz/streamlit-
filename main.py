import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv('claves.env')
chat = os.getenv('chat')


# Configurar la API key de OpenAI
client = OpenAI(api_key=chat)



# Función para generar datos de ejemplo
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'edad': np.random.randint(18, 80, n_samples),
        'ingreso_anual': np.random.normal(50000, 15000, n_samples),
        'puntuacion_credito': np.random.randint(300, 850, n_samples),
        'compras_ultimo_mes': np.random.exponential(100, n_samples),
        'tiempo_como_cliente': np.random.randint(1, 120, n_samples)  # en meses
    }
    df = pd.DataFrame(data)
    return df

# Aplicación Streamlit
st.title("Análisis de Clusters con Explicación IA")

# 1. Entender el Conjunto de Datos y Exploración Inicial
st.header("Paso 1: Entender el Conjunto de Datos")
data_option = st.radio("Seleccione la fuente de datos:", ('Usar datos de ejemplo', 'Cargar mi propio archivo'))

if data_option == 'Usar datos de ejemplo':
    data = generate_sample_data()
    st.success("Datos de ejemplo cargados exitosamente.")
else:
    uploaded_file = st.file_uploader("Carga tu archivo CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Por favor, carga un archivo CSV.")
        st.stop()

st.write("Primeras filas del archivo:")
st.write(data.head())
st.write("Información de las columnas:")
st.write(data.dtypes)

# Visualización de correlaciones para comprender mejor las relaciones entre las variables
st.subheader("Mapa de Correlación de las Variables")
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
st.pyplot(plt)

# Identificar las variables con mayor correlación
correlated_vars = corr.unstack().sort_values(kind="quicksort", ascending=False)
correlated_vars = correlated_vars[correlated_vars != 1]  # Filtrar las correlaciones que no sean la misma variable consigo misma
most_correlated = correlated_vars[::2].head(3)  # Seleccionar las tres correlaciones más altas, evitando duplicados
st.write("**Hallazgo**: Las variables más correlacionadas son:")
for index, value in most_correlated.items():
    st.write(f"- {index[0]} y {index[1]} con una correlación de {value:.2f}")

# 2. Preprocesamiento de Datos
st.header("Paso 2: Preprocesamiento de Datos")
st.write("Realizando la normalización de las características...")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

st.success("Datos normalizados.")

# 3. Selección del Algoritmo de Clustering
st.header("Paso 3: Selección del Algoritmo de Clustering")
algorithm = st.selectbox("Seleccione el algoritmo de clustering a usar:", ['K-means', 'DBSCAN'])

# 4. Determinación del Número Óptimo de Clusters (para K-means)
n_clusters = 3
if algorithm == 'K-means':
    st.subheader("Determinación del Número Óptimo de Clusters")
    st.write("Usando el método del codo y el análisis de silueta para determinar el número óptimo de clusters...")
    
    # Calcular el codo para determinar el número óptimo de clusters
    wcss = []
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_scores.append(score)
    
    plt.figure()
    plt.plot(range(2, 11), wcss, marker='o')
    plt.title("Método del Codo")
    plt.xlabel("Número de Clusters")
    plt.ylabel("WCSS")
    st.pyplot(plt)

    # Mostrar sugerencia del número de clusters basado en el método del codo y el Silhouette Score
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # Ajustar índice para el rango comenzando en 2
    st.write(f"**Sugerencia**: El número recomendado de clusters es {optimal_k}, basado en el método del codo y un análisis del Silhouette Score.")

    n_clusters = st.slider("Seleccione el número de clusters (K):", min_value=2, max_value=10, value=optimal_k)

# 5. Evaluación y Validación de Clusters
st.header("Paso 5: Evaluación y Validación de Clusters")
st.write("Realizando clustering...")

if algorithm == 'K-means':
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif algorithm == 'DBSCAN':
    st.write("DBSCAN es un algoritmo basado en densidad que agrupa puntos que están cerca entre sí y marca los puntos aislados como ruido.")
    eps = st.slider("Eps (Distancia máxima para agrupar puntos vecinos):", min_value=0.1, max_value=5.0, value=0.5)
    min_samples = st.slider("Min Samples (Número mínimo de puntos para formar un cluster):", min_value=1, max_value=20, value=5)
    st.write("**Contexto**:")
    st.write("- `eps` controla qué tan cerca deben estar los puntos para considerarse vecinos.")
    st.write("- `min_samples` define cuántos puntos deben estar cerca para definir un cluster.")
    model = DBSCAN(eps=eps, min_samples=min_samples)

clusters = model.fit_predict(scaled_data)
data['Cluster'] = clusters

# Calcular Silhouette Score si es K-means
if algorithm == 'K-means' and n_clusters > 1:
    score = silhouette_score(scaled_data, clusters)
    st.write(f"Silhouette Score: {score:.2f}")
    st.write("**Hallazgo**: Un Silhouette Score cercano a 1 indica una buena separación entre los clusters. Valores cercanos a 0 indican clusters solapados.")

# 6. Interpretación y Visualización de Resultados
st.header("Paso 6: Interpretación y Visualización de Resultados")
st.write("Selecciona la visualización para los clusters:")

visualization_option = st.selectbox(
    "Selecciona el tipo de visualización para los clusters:",
    ["PCA Scatterplot", "Pairplot", "Jointplot"]
)

if visualization_option == "PCA Scatterplot":
    # Mostrar PCA Scatterplot
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = clusters
    sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1')
    st.pyplot(plt)

elif visualization_option == "Pairplot":
    # Mostrar Pairplot
    sns.pairplot(data, hue='Cluster', palette='Set1')
    st.pyplot(plt)

elif visualization_option == "Jointplot":
    # Mostrar Jointplot
    var_x = st.selectbox("Selecciona la variable X:", data.columns)
    var_y = st.selectbox("Selecciona la variable Y:", data.columns)

    if var_x and var_y and var_x != var_y:
        g = sns.jointplot(data=data, x=var_x, y=var_y, hue='Cluster', kind="scatter", height=6, palette="Set1")
        g.plot_marginals(sns.kdeplot, shade=True, alpha=0.5)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"Jointplot de {var_x} y {var_y} por Cluster")
        st.pyplot(g.fig)

# 7. Generación de Explicación de los Clusters con GPT-4
st.header("Paso 7: Explicación de Clusters")
if st.button("Generar Explicación con GPT-4"):
    explanations = []
    for i in range(max(clusters) + 1):
        cluster_data = data[data['Cluster'] == i]
        prompt = f"Describe el siguiente cluster de datos:\n{cluster_data.describe()}\n\nExplicación concisa:"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un analista de datos experto en explicar resultados de clustering."},
                {"role": "user", "content": prompt}
            ]
        )
        explanations.append(response.choices[0].message.content.strip())
# Mostrar las explicaciones de los clusters generadas por GPT-4
    for i, exp in enumerate(explanations):
        st.subheader(f"Cluster {i}")
        st.write(exp)

# Sección de Ayuda
st.sidebar.title("Ayuda")
st.sidebar.markdown("""
## Estructura del archivo:

- Formato: CSV (valores separados por comas)
- Encabezado: La primera fila debe contener los nombres de las columnas.
- Tipos de datos: Al menos dos columnas deben ser numéricas.
- Evite valores faltantes.

Ejemplo de datos (clientes de una tienda en línea):
""")
st.sidebar.text("Desarrollado con Streamlit y OpenAI")

# Mensaje de cierre
st.write("Gracias por utilizar la aplicación de análisis de clusters. Esperamos que los resultados y las explicaciones hayan sido útiles para entender mejor los patrones en sus datos.")

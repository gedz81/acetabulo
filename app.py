import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Clasificador Morfológico de Pelvis de Guanaco")
st.write("Ingrese las medidas para clasificar el espécimen y visualizar su posición en el morfoespacio.")

# 1. Cargamos los datos limpios manteniendo el orden correcto de tus columnas
@st.cache_data
def cargar_y_procesar_datos():
    df = pd.read_csv('medidasAcetabulo.csv', sep=';')
    
    # Limpiamos comas por puntos y convertimos a float de forma directa
    df['HIS'] = df['HIS'].astype(str).str.replace(',', '.').astype(float)
    df['HAM'] = df['HAM'].astype(str).str.replace(',', '.').astype(float)
    
    return df

df_base = cargar_y_procesar_datos()

# 2. Calculamos los centroides automáticamente con K-medias
X = df_base[['HIS', 'HAM']]
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df_base['cluster'] = kmeans.fit_predict(X)

# Asignamos dinámicamente los centroides
centroides = kmeans.cluster_centers_
if centroides[0][1] < centroides[1][1]:
    centroide_hembra = centroides[0]
    centroide_macho = centroides[1]
else:
    centroide_hembra = centroides[1]
    centroide_macho = centroides[0]

# 3. Campos de entrada en la interfaz
col1, col2 = st.columns(2)
with col1:
    his_input = st.number_input("Medida HIS:", value=float(df_base['HIS'].mean()), format="%.2f")
with col2:
    ham_input = st.number_input("Medida HAM:", value=float(df_base['HAM'].mean()), format="%.2f")

if st.button("Clasificar y Graficar"):
    punto_nuevo = np.array([his_input, ham_input])
    
    # Distancia euclidiana
    dist_hembra = np.linalg.norm(punto_nuevo - centroide_hembra)
    dist_macho = np.linalg.norm(punto_nuevo - centroide_macho)
    
    st.subheader("Resultado:")
    if dist_hembra < dist_macho:
        st.success(f"🟢 El espécimen se clasifica como: **HEMBRA** (Distancia: {dist_hembra:.2f})")
    else:
        st.info(f"🔵 El espécimen se clasifica como: **MACHO** (Distancia: {dist_macho:.2f})")
    
    # 4. Generamos el gráfico del morfoespacio con los ejes bien ordenados
    fig, ax = plt.subplots(figsize=(8, 6))
    
    hembras_ref = df_base[df_base['Sexo'] == 'H']
    machos_ref = df_base[df_base['Sexo'] == 'M']
    
    ax.scatter(hembras_ref['HIS'], hembras_ref['HAM'], color='lightcoral', label='Hembras (referencia)', alpha=0.7, s=50)
    ax.scatter(machos_ref['HIS'], machos_ref['HAM'], color='skyblue', label='Machos (referencia)', alpha=0.7, s=50)
    
    ax.scatter(centroide_hembra[0], centroide_hembra[1], color='darkred', marker='X', s=150, label='Centroide Hembra')
    ax.scatter(centroide_macho[0], centroide_macho[1], color='darkblue', marker='X', s=150, label='Centroide Macho')
    
    # Estrellita dorada del nuevo espécimen
    ax.scatter(his_input, ham_input, color='gold', edgecolor='black', marker='*', s=300, label='Nuevo Espécimen', zorder=5)
    
    ax.set_xlabel('HIS')
    ax.set_ylabel('HAM')
    ax.set_title('Morfoespacio de Pelvis de Guanaco (HIS vs HAM)')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

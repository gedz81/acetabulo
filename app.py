import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Clasificador Morfológico de Pelvis de Guanaco")
st.write("Ingrese las medidas para clasificar el espécimen (HAM en eje X, HIS en eje Y).")

# 1. Cargamos los datos del CSV corregido
@st.cache_data
def cargar_datos():
    df = pd.read_csv('medidasAcetabulo.csv', sep=';')
    
    # Limpiamos comas por puntos y convertimos a float
    df['HIS'] = df['HIS'].astype(str).str.replace(',', '.').astype(float)
    df['HAM'] = df['HAM'].astype(str).str.replace(',', '.').astype(float)
    
    return df

df_base = cargar_datos()

# 2. Calculamos los centroides automáticamente con K-medias (usando las columnas reales)
X = df_base[['HIS', 'HAM']]
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df_base['cluster'] = kmeans.fit_predict(X)

# Asignamos dinámicamente los centroides
centroides = kmeans.cluster_centers_
if centroides[0][0] < centroides[1][0]:
    centroide_hembra = centroides[0] # [HIS, HAM]
    centroide_macho = centroides[1]
else:
    centroide_hembra = centroides[1]
    centroide_macho = centroides[0]

# 3. Campos de entrada en la interfaz (Eje X = HAM, Eje Y = HIS)
col1, col2 = st.columns(2)
with col1:
    ham_input = st.number_input("Medida HAM (Eje X):", value=float(df_base['HAM'].mean()), format="%.2f")
with col2:
    his_input = st.number_input("Medida HIS (Eje Y):", value=float(df_base['HIS'].mean()), format="%.2f")

if st.button("Clasificar y Graficar"):
    punto_nuevo = np.array([his_input, ham_input]) # [HIS, HAM] para el cálculo
    
    # Distancia euclidiana
    dist_hembra = np.linalg.norm(punto_nuevo - centroide_hembra)
    dist_macho = np.linalg.norm(punto_nuevo - centroide_macho)
    
    st.subheader("Resultado:")
    if dist_hembra < dist_macho:
        st.success(f"🟢 El espécimen se clasifica como: **HEMBRA** (Distancia: {dist_hembra:.2f})")
    else:
        st.info(f"🔵 El espécimen se clasifica como: **MACHO** (Distancia: {dist_macho:.2f})")
    
    # 4. Generamos el gráfico con Eje X = HAM y Eje Y = HIS
    fig, ax = plt.subplots(figsize=(8, 6))
    
    hembras_ref = df_base[df_base['Sexo'] == 'H']
    machos_ref = df_base[df_base['Sexo'] == 'M']
    
    # Eje X: HAM, Eje Y: HIS
    ax.scatter(hembras_ref['HAM'], hembras_ref['HIS'], color='lightcoral', label='Hembras (referencia)', alpha=0.7, s=50)
    ax.scatter(machos_ref['HAM'], machos_ref['HIS'], color='skyblue', label='Machos (referencia)', alpha=0.7, s=50)
    
    # Centroides (posición [1] es HAM, posición [0] es HIS)
    ax.scatter(centroide_hembra[1], centroide_hembra[0], color='darkred', marker='X', s=150, label='Centroide Hembra')
    ax.scatter(centroide_macho[1], centroide_macho[0], color='darkblue', marker='X', s=150, label='Centroide Macho')
    
    # Estrellita dorada del nuevo espécimen (X = HAM, Y = HIS)
    ax.scatter(ham_input, his_input, color='gold', edgecolor='black', marker='*', s=300, label='Nuevo Espécimen', zorder=5)
    
    ax.set_xlabel('HAM')
    ax.set_ylabel('HIS')
    ax.set_title('Morfoespacio de Pelvis de Guanaco (HAM vs HIS)')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

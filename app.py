import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Clasificador Morfológico de Pelvis de Guanaco")
st.write("Ingrese las medidas de **HIS** y **HAM** para clasificar el espécimen y visualizar su posición en el morfoespacio.")

# 1. Cargamos los datos reales para graficar el morfoespacio de fondo
@st.cache_data
def cargar_datos():
    df = pd.read_csv('medidasAcetabulo.csv', sep=';')
    df['HIS'] = df['HIS'].astype(str).str.replace(',', '.').astype(float)
    df['HAM'] = df['HAM'].astype(str).str.replace(',', '.').astype(float)
    return df

df_base = cargar_datos()

# Centroides exactos calculados de tu muestra
centroide_hembra = np.array([15.91645161, 34.77322581])
centroide_macho = np.array([19.49400000, 41.64600000])

# 2. Campos de entrada en la barra lateral o principal
col1, col2 = st.columns(2)
with col1:
    his_input = st.number_input("Medida HIS:", value=17.0, format="%.2f")
with col2:
    ham_input = st.number_input("Medida HAM:", value=37.0, format="%.2f")

if st.button("Clasificar y Graficar"):
    # Convertimos la entrada en un vector
    punto_nuevo = np.array([his_input, ham_input])
    
    # Calculamos la distancia euclidiana
    dist_hembra = np.linalg.norm(punto_nuevo - centroide_hembra)
    dist_macho = np.linalg.norm(punto_nuevo - centroide_macho)
    
    st.subheader("Resultado:")
    if dist_hembra < dist_macho:
        st.success(f"🟢 El espécimen se clasifica como: **HEMBRA** (Distancia: {dist_hembra:.2f})")
    else:
        st.info(f"🔵 El espécimen se clasifica como: **MACHO** (Distancia: {dist_macho:.2f})")
    
    # 3. Generamos el gráfico del morfoespacio
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Graficamos los datos de la muestra base según su sexo original registrado
    hembras_ref = df_base[df_base['Sexo'] == 'H']
    machos_ref = df_base[df_base['Sexo'] == 'M']
    
    ax.scatter(hembras_ref['HIS'], hembras_ref['HAM'], color='lightcoral', label='Hembras (referencia)', alpha=0.7, s=50)
    ax.scatter(machos_ref['HIS'], machos_ref['HAM'], color='skyblue', label='Machos (referencia)', alpha=0.7, s=50)
    
    # Graficamos los centroides
    ax.scatter(centroide_hembra[0], centroide_hembra[1], color='darkred', marker='X', s=150, label='Centroide Hembra')
    ax.scatter(centroide_macho[0], centroide_macho[1], color='darkblue', marker='X', s=150, label='Centroide Macho')
    
    # Graficamos el nuevo espécimen consultado con un color bien llamativo (ej. estrella amarilla o punto morado grande)
    ax.scatter(his_input, ham_input, color='gold', edgecolor='black', marker='*', s=300, label='Nuevo Espécimen', zorder=5)
    
    ax.set_xlabel('HIS')
    ax.set_ylabel('HAM')
    ax.set_title('Morfoespacio de Pelvis de Guanaco (HIS vs HAM)')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Mostramos el gráfico en Streamlit
    st.pyplot(fig)

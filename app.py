import streamlit as st
import numpy as np

st.title("Clasificador Morfológico de Pelvis de Guanaco")
st.write("Ingrese las medidas de **HIS** y **HAM** para clasificar el espécimen.")

# Centroides exactos calculados a partir de tu archivo 'medidasAcetabulo.csv'
# Centroide 0 (Grupo de menor tamaño) y Centroide 1 (Grupo de mayor tamaño)
centroide_hembra = np.array([15.91645161, 34.77322581])
centroide_macho = np.array([19.49400000, 41.64600000])

# Campos de entrada en la interfaz web
his_input = st.number_input("Medida HIS:", value=17.0, format="%.2f")
ham_input = st.number_input("Medida HAM:", value=37.0, format="%.2f")

if st.button("Clasificar espécimen"):
    # Convertimos la entrada en un vector
    punto_nuevo = np.array([his_input, ham_input])
    
    # Calculamos la distancia euclidiana a cada centroide
    dist_hembra = np.linalg.norm(punto_nuevo - centroide_hembra)
    dist_macho = np.linalg.norm(punto_nuevo - centroide_macho)
    
    st.subheader("Resultado de la Clasificación:")
    
    if dist_hembra < dist_macho:
        st.success(f"🟢 El espécimen se clasifica como: **HEMBRA** (Distancia al centroide: {dist_hembra:.2f})")
    else:
        st.info(f"🔵 El espécimen se clasifica como: **MACHO** (Distancia al centroide: {dist_macho:.2f})")
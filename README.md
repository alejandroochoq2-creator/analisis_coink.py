# analisis_coink.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(page_title="Análisis Coink", page_icon="🏦", layout="wide")
st.title("🏦 Análisis de Depósitos en Oinks")
st.markdown("Cálculo del **Coink Score** y clasificación de usuarios.")


# ============================================================
# CARGA DE DATOS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "depositos_oinks.csv"

if not CSV_PATH.exists():
    st.error(f"❌ No se encontró: {CSV_PATH}")
    st.info(f"📂 Archivos disponibles: {[f.name for f in BASE_DIR.iterdir()]}")
    st.stop()

@st.cache_data
def cargar():
    df = pd.read_csv(CSV_PATH)
    df['operation_date'] = pd.to_datetime(df['operation_date'])
    df['user_createddate'] = pd.to_datetime(df['user_createddate'])
    return df

df = cargar()
st.success(f"✅ Cargados {len(df):,} depósitos de {df['user_id'].nunique()} usuarios.")


# ============================================================
# MÉTRICAS POR USUARIO
# ============================================================
fecha_max = df['operation_date'].max()

usuarios = df.groupby('user_id').agg(
    num_depositos=('operation_value', 'count'),
    monto_total=('operation_value', 'sum'),
    monto_promedio=('operation_value', 'mean'),
    fecha_creacion=('user_createddate', 'first'),
    ultimo_deposito=('operation_date', 'max'),
    lugar_favorito=('maplocation_name', lambda x: x.mode().iloc[0])
).reset_index()

usuarios['antiguedad_dias'] = (fecha_max - usuarios['fecha_creacion']).dt.days

def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

usuarios['n_frecuencia'] = minmax(usuarios['num_depositos'])
usuarios['n_monto']      = minmax(usuarios['monto_total'])
usuarios['n_ticket']     = minmax(usuarios['monto_promedio'])
usuarios['n_antiguedad'] = minmax(usuarios['antiguedad_dias'])

usuarios['coink_score'] = (
    0.30 * usuarios['n_frecuencia'] +
    0.30 * usuarios['n_monto'] +
    0.20 * usuarios['n_ticket'] +
    0.20 * usuarios['n_antiguedad']
) * 100

def clasificar(s):
    if s >= 70: return 'Oro'
    elif s >= 40: return 'Plata'
    else: return 'Bronce'

usuarios['categoria'] = usuarios['coink_score'].apply(clasificar)


# ============================================================
# MÉTRICAS PRINCIPALES
# ============================================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Usuarios", f"{len(usuarios):,}")
col2.metric("💰 Monto total", f"${usuarios['monto_total'].sum():,.0f}")
col3.metric("📊 Score promedio", f"{usuarios['coink_score'].mean():.1f}")
col4.metric("🏆 Usuarios Oro", f"{(usuarios['categoria']=='Oro').sum()}")


# ============================================================
# GRÁFICAS
# ============================================================
st.markdown("---")
sns.set_style("darkgrid")

col_a, col_b = st.columns(2)

# Histograma score
with col_a:
    st.subheader("📈 Distribución del Coink Score")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(usuarios['coink_score'], bins=30, color='teal', edgecolor='black')
    ax1.set_xlabel('Score'); ax1.set_ylabel('N° usuarios')
    st.pyplot(fig1)

# Categorías
with col_b:
    st.subheader("📊 Usuarios por categoría")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    counts = usuarios['categoria'].value_counts().reindex(['Oro', 'Plata', 'Bronce']).fillna(0)
    colores = {'Oro': 'gold', 'Plata': 'silver', 'Bronce': '#cd7f32'}
    ax2.bar(counts.index, counts.values,
            color=[colores[c] for c in counts.index], edgecolor='black')
    ax2.set_ylabel('N° usuarios')
    st.pyplot(fig2)

# Scatter
st.subheader("🔵 Frecuencia vs Monto (color = Score)")
fig3, ax3 = plt.subplots(figsize=(12, 5))
sc = ax3.scatter(usuarios['num_depositos'], usuarios['monto_total'],
                 c=usuarios['coink_score'], cmap='viridis',
                 alpha=0.6, edgecolors='black')
ax3.set_xlabel('N° depósitos'); ax3.set_ylabel('Monto total ($)')
ax3.set_yscale('log')
plt.colorbar(sc, ax=ax3, label='Coink Score')
st.pyplot(fig3)

# Ubicaciones
st.subheader("📍 Monto total por ubicación")
lugar = df.groupby('maplocation_name')['operation_value'].sum().sort_values()
fig4, ax4 = plt.subplots(figsize=(12, 4))
ax4.barh(lugar.index, lugar.values, color='coral', edgecolor='black')
ax4.set_xlabel('Monto total ($)')
st.pyplot(fig4)


# ============================================================
# TOP USUARIOS
# ============================================================
st.markdown("---")
st.subheader("🏆 Top 10 usuarios por Coink Score")
top10 = usuarios.nlargest(10, 'coink_score')[
    ['user_id', 'num_depositos', 'monto_total', 'monto_promedio',
     'antiguedad_dias', 'coink_score', 'categoria']
].reset_index(drop=True)
st.dataframe(top10, use_container_width=True)


# ============================================================
# RESUMEN
# ============================================================
st.subheader("📋 Resumen por categoría")
resumen = usuarios.groupby('categoria').agg(
    num_usuarios=('user_id', 'count'),
    monto_promedio=('monto_total', 'mean'),
    depositos_promedio=('num_depositos', 'mean'),
    score_promedio=('coink_score', 'mean')
).round(2)
st.dataframe(resumen, use_container_width=True)


# ============================================================
# DESCARGA CSV
# ============================================================
st.markdown("---")
st.download_button(
    "⬇️ Descargar usuarios calificados (CSV)",
    data=usuarios.to_csv(index=False).encode('utf-8'),
    file_name='usuarios_calificados.csv',
    mime='text/csv'
)

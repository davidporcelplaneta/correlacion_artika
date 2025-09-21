# app.py
# Streamlit app: Cargar CSV, calcular correlaciones con "ventas" por meses y visualizar
# Ejecuta con: streamlit run app.py

import io
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Correlaciones con Ventas", layout="wide")
st.title("📈 Correlaciones de variables con **ventas**")
st.write(
    "Sube un CSV con una columna **ventas** y (opcionalmente) una columna de **mes**. "
    "La app detecta el separador y convierte comas decimales automáticamente."
)

# ------------------------------ Utils ------------------------------
def strip_bom(s: str) -> str:
    return s.lstrip("\ufeff").lstrip("﻿")

def normalize_col(c: str) -> str:
    c = strip_bom(str(c)).strip()
    c = "".join(ch for ch in unicodedata.normalize("NFKD", c) if not unicodedata.combining(ch))
    c = c.replace(" ", "_").replace("/", "_").replace("-", "_").replace("__", "_")
    return c

def maybe_to_float_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        # "1.234,56" o "1234,56"
        if s.str.contains(r"\d,\d", regex=True, na=False).any():
            s2 = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            return pd.to_numeric(s2, errors="coerce")
        # "1234.56" como texto
        if s.str.contains(r"\d\.\d", regex=True, na=False).any():
            return pd.to_numeric(s, errors="coerce")
    return s

MESES_ES = [
    "enero","febrero","marzo","abril","mayo","junio",
    "julio","agosto","septiembre","octubre","noviembre","diciembre"
]

def order_by_month(df: pd.DataFrame, col_mes: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp[col_mes] = tmp[col_mes].astype(str).str.strip()
    tmp["_mes_lower"] = tmp[col_mes].str.lower()
    cat_type = pd.CategoricalDtype(categories=MESES_ES, ordered=True)
    tmp["_mes_lower"] = tmp["_mes_lower"].astype(cat_type)
    if tmp["_mes_lower"].isna().all():
        tmp = tmp.drop(columns=["_mes_lower"])
        return df  # no se pudo ordenar
    tmp = tmp.sort_values("_mes_lower").drop(columns=["_mes_lower"])
    return tmp

def safe_barh(series: pd.Series, title: str, xlabel: str, ylabel: str):
    """Grafica barras horizontales solo si la serie no está vacía."""
    if series is None or len(series) == 0:
        st.info("No hay variables (aparte de ventas) para graficar.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    series.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig, clear_figure=True)

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.header("⚙️ Opciones")
    metodo = st.selectbox("Método de correlación", ["pearson", "spearman", "kendall"], index=0)
    separar_por_mes = st.checkbox("Calcular y mostrar correlaciones por mes", value=False)
    nombre_col_mes = st.text_input("Nombre de la columna de mes (si existe)", value="mes")
    nombre_col_ventas = st.text_input("Nombre de la columna de ventas", value="ventas")
    st.markdown("---")
    st.caption("Consejo: si tus decimales usan coma (1,23) la app lo convierte automáticamente.")

# ------------------------------ Carga del archivo ------------------------------
uploaded_file = st.file_uploader("📤 Sube tu archivo .csv", type=["csv"])

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()

    # Intentos de decodificación
    df = None
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            text = raw_bytes.decode(enc)
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            break
        except Exception:
            df = None

    if df is None or df.empty:
        st.error("No se pudo leer el CSV o está vacío. Revisa el archivo/codificación.")
        st.stop()

    # Limpiar nombres de columnas
    df.columns = [normalize_col(c) for c in df.columns]

    # Conversión de columnas numéricas con coma decimal
    for c in df.columns:
        if c not in (nombre_col_mes,):
            df[c] = maybe_to_float_series(df[c])

    # Vista previa (nueva API: width)
    st.subheader("👀 Vista previa de datos")
    st.dataframe(df.head(20), width="stretch")

    # Ordenar por mes si procede
    col_mes = nombre_col_mes if nombre_col_mes in df.columns else None
    if col_mes:
        df = order_by_month(df, col_mes)

    # Comprobaciones
    if nombre_col_ventas not in df.columns:
        st.error(f"No se encontró la columna de ventas: **{nombre_col_ventas}**")
        st.stop()

    # Selección de columnas numéricas
    num_df = df.select_dtypes(include=[np.number])
    if nombre_col_ventas not in num_df.columns:
        st.error(f"La columna **{nombre_col_ventas}** no es numérica después de la conversión.")
        st.stop()

    # ---------------------- Correlación global ----------------------
    st.subheader("📊 Correlaciones con ventas (global)")

    try:
        corr_serie = num_df.corr(method=metodo)[nombre_col_ventas].sort_values(ascending=False)
    except Exception as e:
        st.error(f"Error calculando correlaciones: {e}")
        st.stop()

    st.dataframe(corr_serie.to_frame("correlacion").round(4), width="stretch")

    # Serie para graficar (sin la propia columna de ventas)
    serie_graf = corr_serie.drop(labels=[nombre_col_ventas]) if nombre_col_ventas in corr_serie.index else corr_serie
    safe_barh(
        serie_graf,
        title=f"Correlación ({metodo}) con {nombre_col_ventas}",
        xlabel="Coeficiente de correlación",
        ylabel="Variable",
    )

    # Heatmap simple (solo si hay >= 2 columnas numéricas)
    st.subheader("🧯 Matriz de correlación (numéricas)")
    if num_df.shape[1] >= 2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        mat = num_df.corr(method=metodo)
        im = ax2.imshow(mat, aspect="auto")
        fig2.colorbar(im, ax=ax2, label="Correlación")
        ax2.set_xticks(range(len(mat.columns)))
        ax2.set_xticklabels(mat.columns, rotation=90)
        ax2.set_yticks(range(len(mat.columns)))
        ax2.set_yticklabels(mat.columns)
        ax2.set_title(f"Matriz de correlación ({metodo})")
        st.pyplot(fig2, clear_figure=True)
    else:
        st.info("Se necesita al menos 2 columnas numéricas para mostrar la matriz de correlación.")

    # Descarga de resultados
    csv_corr = corr_serie.to_frame("correlacion").to_csv(index=True).encode("utf-8")
    st.download_button(
        "⬇️ Descargar correlaciones (CSV)",
        data=csv_corr,
        file_name="correlaciones_ventas.csv",
        mime="text/csv",
    )

    # ---------------------- Correlaciones por mes (opcional) ----------------------
    if separar_por_mes and col_mes:
        st.subheader("🗓️ Correlaciones por mes")
        meses_presentes = df[col_mes].dropna().astype(str).unique().tolist()
        if not meses_presentes:
            st.info("No hay valores de mes válidos para separar.")
        for mes in meses_presentes:
            sub = df[df[col_mes].astype(str) == mes]
            sub_num = sub.select_dtypes(include=[np.number])

            # Necesitamos al menos ventas + 1 variable numérica
            if nombre_col_ventas not in sub_num.columns or sub_num.shape[1] < 2 or len(sub_num) < 2:
                continue

            try:
                corr_mes = sub_num.corr(method=metodo)[nombre_col_ventas].sort_values(ascending=False)
            except Exception:
                continue

            with st.expander(f"Mes: {mes}"):
                st.dataframe(corr_mes.to_frame("correlacion").round(4), width="stretch")
                serie_mes = corr_mes.drop(nombre_col_ventas) if nombre_col_ventas in corr_mes.index else corr_mes
                safe_barh(
                    serie_mes,
                    title=f"Correlaciones ({metodo}) con {nombre_col_ventas} - {mes}",
                    xlabel="Coeficiente",
                    ylabel="Variable",
                )
else:
    st.info("Sube un archivo CSV para comenzar.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Popularitas Lagu Spotify",
    page_icon="🎵",
    layout="wide",
)
 
st.title("🎵 Prediksi Popularitas Lagu Spotify")
st.markdown(
    """
Aplikasi ini memprediksi kategori popularitas lagu (**Low / Medium / High**)
berdasarkan fitur audio menggunakan algoritma **Decision Tree Classifier**.
"""
)
 
FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]
 
 
# FUNGSI TRAINING 
@st.cache_resource(show_spinner="Melatih model Decision Tree...")
def train_model(df: pd.DataFrame):
    """
    Melatih Decision Tree Classifier untuk klasifikasi popularitas lagu.
    Hasil di-cache oleh Streamlit, jadi tidak training ulang setiap
    ada interaksi user (slider, tombol, dll) selama dataset sama.
    """
    df = df.copy()
 
    # --- cleaning ---
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
 
    df = df.dropna(subset=["popularity"])
 
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"])
    elif "track_name" in df.columns and "artists" in df.columns:
        df = df.drop_duplicates(subset=["track_name", "artists"])
 
    # --- feature selection ---
    features = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=features)
 
    # --- binning popularity jadi 3 kelas pakai kuantil ---
    df["popularity_category"] = pd.qcut(
        df["popularity"], q=3, labels=["Low", "Medium", "High"], duplicates="drop"
    )
 
    # --- split data ---
    X = df[features]
    y = df["popularity_category"]
 
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
 
    # --- training ---
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)
 
    # --- evaluasi ---
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
 
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
 
    report = classification_report(
        y_test, y_test_pred, target_names=le.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_test_pred)
 
    importance_df = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
 
    return {
        "model": model,
        "label_encoder": le,
        "features": features,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importance": importance_df,
        "cleaned_df": df,
    }
 
 
# SIDEBAR NAVIGASI
menu = st.sidebar.radio(
    "Navigasi",
    ["📁 Upload & Eksplorasi Data", "📊 Visualisasi", "🔮 Prediksi", "📈 Evaluasi Model"],
)
 
if "df" not in st.session_state:
    st.session_state.df = None
if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None
 
# Upload dataset selalu tersedia di sidebar biar gampang diakses dari tab manapun
st.sidebar.markdown("---")
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("File CSV dataset Spotify", type=["csv"])
 
if uploaded_file is not None:
    df_loaded = pd.read_csv(uploaded_file)
    st.session_state.df = df_loaded
 
    # Cek apakah popularity tersedia, kalau ada langsung training
    if "popularity" in df_loaded.columns:
        st.session_state.model_bundle = train_model(df_loaded)
        st.sidebar.success("Dataset dimuat & model berhasil dilatih ✅")
    else:
        st.session_state.model_bundle = None
        st.sidebar.warning("Kolom 'popularity' tidak ditemukan, model tidak bisa dilatih.")
 
df = st.session_state.df
model_bundle = st.session_state.model_bundle

#1: UPLOAD & EKSPLORASI DATA
if menu == "📁 Upload & Eksplorasi Data":
    st.header("Eksplorasi Dataset")
 
    if df is None:
        st.warning("Silakan upload file CSV terlebih dahulu lewat sidebar.")
    else:
        st.success(f"Dataset aktif: {df.shape[0]} baris, {df.shape[1]} kolom")
 
        st.subheader("Cuplikan Data")
        st.dataframe(df.head(10))
 
        st.subheader("Informasi Dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", f"{df.shape[0]:,}")
        col2.metric("Jumlah Kolom", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
 
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe())
 

#2: VISUALISASI
elif menu == "📊 Visualisasi":
    st.header("Visualisasi Data")
 
    if df is None:
        st.warning("Silakan upload dataset terlebih dahulu lewat sidebar.")
    else:
        available_features = [f for f in FEATURES if f in df.columns]
 
        if "popularity" in df.columns:
            st.subheader("Distribusi Popularitas Lagu")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df["popularity"], bins=30, kde=True, ax=ax, color="#1DB954")
            ax.set_xlabel("Popularity Score")
            ax.set_ylabel("Jumlah Lagu")
            st.pyplot(fig)
 
        if available_features:
            st.subheader("Korelasi Antar Fitur Audio")
            corr_cols = available_features + (["popularity"] if "popularity" in df.columns else [])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="Greens", ax=ax)
            st.pyplot(fig)
 
            st.subheader("Distribusi Tiap Fitur")
            selected_feature = st.selectbox("Pilih fitur", available_features)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[selected_feature], bins=30, kde=True, ax=ax, color="#1DB954")
            st.pyplot(fig)
 
        if "track_genre" in df.columns:
            st.subheader("Top 10 Genre Terbanyak")
            top_genres = df["track_genre"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax, color="#1DB954")
            ax.set_xlabel("Jumlah Lagu")
            st.pyplot(fig)

#3: PREDIKSI
elif menu == "🔮 Prediksi":
    st.header("Prediksi Popularitas Lagu")
 
    if model_bundle is None:
        st.error("Upload dataset (dengan kolom 'popularity') lewat sidebar untuk melatih model terlebih dahulu.")
    else:
        model = model_bundle["model"]
        le = model_bundle["label_encoder"]
        features = model_bundle["features"]
 
        st.markdown("Masukkan nilai fitur audio lagu di bawah ini:")
 
        col1, col2 = st.columns(2)
        input_values = {}
 
        defaults = {
            "danceability": 0.5, "energy": 0.5, "loudness": -10.0,
            "speechiness": 0.1, "acousticness": 0.3, "instrumentalness": 0.0,
            "liveness": 0.2, "valence": 0.5, "tempo": 120.0, "duration_ms": 210000,
        }
 
        for i, feature in enumerate(features):
            col = col1 if i % 2 == 0 else col2
            if feature == "loudness":
                input_values[feature] = col.slider(feature, -60.0, 0.0, defaults.get(feature, -10.0))
            elif feature == "tempo":
                input_values[feature] = col.slider(feature, 0.0, 250.0, defaults.get(feature, 120.0))
            elif feature == "duration_ms":
                input_values[feature] = col.number_input(
                    feature, min_value=0, max_value=600000, value=int(defaults.get(feature, 210000))
                )
            else:
                input_values[feature] = col.slider(feature, 0.0, 1.0, defaults.get(feature, 0.5))
 
        if st.button("Prediksi Popularitas", type="primary"):
            input_df = pd.DataFrame([input_values])[features]
            prediction = model.predict(input_df)[0]
            prediction_label = le.inverse_transform([prediction])[0]
            probabilities = model.predict_proba(input_df)[0]
 
            st.subheader("Hasil Prediksi")
            color_map = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}
            st.markdown(f"## {color_map.get(prediction_label, '')} **{prediction_label}**")
 
            st.subheader("Confidence per Kategori")
            prob_df = pd.DataFrame(
                {"Kategori": le.classes_, "Probabilitas": probabilities}
            ).sort_values("Probabilitas", ascending=False)
 
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(data=prob_df, x="Probabilitas", y="Kategori", ax=ax, color="#1DB954")
            ax.set_xlim(0, 1)
            st.pyplot(fig)
 
#4: EVALUASI MODEL
elif menu == "📈 Evaluasi Model":
    st.header("Evaluasi Model")
 
    if model_bundle is None:
        st.error("Upload dataset (dengan kolom 'popularity') lewat sidebar untuk melatih model terlebih dahulu.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Akurasi Train", f"{model_bundle['train_accuracy']*100:.2f}%")
        col2.metric("Akurasi Test", f"{model_bundle['test_accuracy']*100:.2f}%")
 
        st.subheader("Feature Importance")
        importance_df = model_bundle["feature_importance"]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, color="#1DB954")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
 
        st.subheader("Confusion Matrix")
        cm = model_bundle["confusion_matrix"]
        le = model_bundle["label_encoder"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
 
        st.subheader("Classification Report")
        report_df = pd.DataFrame(model_bundle["classification_report"]).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))
 
        st.markdown(
            """
        **Catatan:** Feature importance menunjukkan fitur audio mana yang
        paling berpengaruh terhadap prediksi kategori popularitas lagu.
        """
        )
 
st.sidebar.markdown("---")
st.sidebar.caption("Project Akhir Praktikum Data Mining 2026")
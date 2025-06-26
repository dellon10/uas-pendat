import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo # Untuk mendapatkan data asli untuk fitting scaler

# Memuat model SVM yang telah dilatih
try:
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'svm_model.pkl' tidak ditemukan. Pastikan file model berada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan

# Mengambil dataset asli untuk fitting scaler
# Ini memastikan scaler yang digunakan untuk prediksi konsisten dengan scaler pelatihan
try:
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X_original = breast_cancer_wisconsin_diagnostic.data.features
except Exception as e:
    st.error(f"Error saat mengambil dataset untuk fitting scaler: {e}")
    st.stop()

# Inisialisasi dan fitting StandardScaler dengan fitur data pelatihan asli
# Ini sangat penting untuk penskalaan yang konsisten antara pelatihan dan prediksi
scaler = StandardScaler()
scaler.fit(X_original)

# Inisialisasi LabelEncoder untuk nama target
le = LabelEncoder()
# Fit dengan nilai target yang mungkin (0 dan 1 dari masalah sebelumnya)
le.fit([0, 1]) # 0 untuk Jinak (Benign), 1 untuk Ganas (Malignant), berdasarkan skrip sebelumnya

# Mendefinisikan nama fitur seperti yang muncul di dataset
# Ini penting untuk kolom input dan memastikan urutan yang benar
feature_names = [
    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
    'mean_smoothness', 'mean_compactness', 'mean_concavity',
    'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
    'radius_error', 'texture_error', 'perimeter_error', 'area_error',
    'smoothness_error', 'compactness_error', 'concavity_error',
    'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
    'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area',
    'worst_smoothness', 'worst_compactness', 'worst_concavity',
    'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
]

# Antarmuka Pengguna Streamlit
st.set_page_config(page_title="Prediksi Kanker Payudara", layout="wide")

st.title("🩺 Prediktor Diagnosis Kanker Payudara")
st.markdown("""
Aplikasi ini memprediksi apakah massa payudara bersifat Jinak (non-kanker) atau Ganas (kanker)
berdasarkan fitur-fitur yang dihitung dari gambar digital aspirasi jarum halus (FNA) massa payudara.

Mohon masukkan nilai untuk setiap fitur di bawah ini.
""")

# Kolom input untuk fitur
input_data = {}
st.header("Masukkan Nilai Fitur:")

# Gunakan kolom untuk tata letak yang lebih baik
num_cols = 3 # Jumlah kolom untuk kolom input
cols = st.columns(num_cols)

for i, feature in enumerate(feature_names):
    with cols[i % num_cols]:
        # Berikan nilai default yang masuk akal untuk menghindari kesalahan jika pengguna tidak mengubah
        # Untuk kesederhanaan, menggunakan 0.0 sebagai default, tetapi idealnya adalah mean/median
        input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=0.0, format="%.4f", key=feature)

# Tombol Prediksi
st.markdown("---")
if st.button("Prediksi Diagnosis"):
    # Ubah data input menjadi array NumPy dalam urutan yang benar
    features_array = np.array([input_data[feature] for feature in feature_names]).reshape(1, -1)

    # Skalakan fitur input
    scaled_features = scaler.transform(features_array)

    # Buat prediksi
    prediction = svm_model.predict(scaled_features)
    prediction_proba = svm_model.decision_function(scaled_features) # Untuk SVM, decision_function memberikan kepercayaan diri

    # Dekode prediksi
    predicted_class = le.inverse_transform(prediction)[0] # Akan menjadi 0 atau 1
    class_name = "Jinak" if predicted_class == 0 else "Ganas"

    st.subheader("Hasil Prediksi:")
    if predicted_class == 0:
        st.success(f"Diagnosis yang diprediksi adalah: **{class_name}**")
        st.balloons()
    else:
        st.error(f"Diagnosis yang diprediksi adalah: **{class_name}**")
    
    st.markdown(f"Skor kepercayaan (nilai fungsi keputusan mentah): `{prediction_proba[0]:.4f}`")
    st.info("Catatan: Nilai fungsi keputusan positif menunjukkan Ganas, negatif menunjukkan Jinak.")

st.markdown("---")
st.markdown("Dikembangkan menggunakan Streamlit dan scikit-learn.")
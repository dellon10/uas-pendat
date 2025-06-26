import streamlit as st
import pickle
import numpy as np
import pandas as pd # Import pandas untuk X_original.mean()
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo # Untuk mendapatkan data asli untuk fitting scaler

# Memuat model SVM yang telah dilatih
try:
    # Penting: Pastikan 'svm_model.pkl' berisi model yang dilatih dengan 30 fitur
    # (Lihat kembali instruksi sebelumnya untuk mengoreksi penyimpanan model jika perlu)
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'svm_model.pkl' tidak ditemukan. Pastikan file model berada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika model tidak ditemukan

# Mengambil dataset asli untuk fitting scaler DAN mendapatkan nilai rata-rata fitur
try:
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X_original = breast_cancer_wisconsin_diagnostic.data.features
    # Hitung rata-rata untuk setiap fitur dari dataset asli
    feature_means = X_original.mean().values # Dapatkan nilai rata-rata sebagai array NumPy
except Exception as e:
    st.error(f"Error saat mengambil dataset atau menghitung rata-rata fitur untuk scaler fitting: {e}")
    st.stop()

# Inisialisasi dan fitting StandardScaler dengan fitur data pelatihan asli (30 fitur)
scaler = StandardScaler()
scaler.fit(X_original)

# Inisialisasi LabelEncoder untuk nama target
le = LabelEncoder()
le.fit([0, 1]) # 0 untuk Jinak (Benign), 1 untuk Ganas (Malignant)

# Mendefinisikan semua nama fitur seperti yang muncul di dataset (total 30)
all_feature_names = X_original.columns.tolist() # Lebih baik ambil langsung dari DataFrame

# Pilih hanya 3 fitur yang akan ditampilkan di UI
selected_feature_names = ['mean_radius', 'mean_texture', 'mean_perimeter']

# Antarmuka Pengguna Streamlit
st.set_page_config(page_title="Prediksi Kanker Payudara", layout="wide")

st.title("🩺 Prediktor Diagnosis Kanker Payudara")
st.markdown("""
Aplikasi ini memprediksi apakah massa payudara bersifat Jinak (non-kanker) atau Ganas (kanker)
berdasarkan fitur-fitur yang dihitung dari gambar digital aspirasi jarum halus (FNA) massa payudara.

Mohon masukkan nilai untuk **3 fitur utama** di bawah ini. Fitur lainnya akan menggunakan **nilai rata-rata** dari dataset.
""")

# Kolom input untuk fitur
input_data = {}
st.header("Masukkan Nilai Fitur Utama:")

# Gunakan kolom untuk tata letak yang lebih baik
num_cols_display = 3 # Jumlah kolom untuk tampilan input
cols = st.columns(num_cols_display)

for i, feature in enumerate(selected_feature_names):
    with cols[i % num_cols_display]:
        # Mengambil nilai rata-rata asli sebagai nilai default yang lebih relevan
        # Gunakan X_original.loc[0, feature] atau cari di feature_means
        # Cara yang lebih aman adalah dengan mencari indeks fitur di all_feature_names
        default_value_for_input = X_original[feature].mean()

        input_data[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()}",
            value=float(default_value_for_input), # Pastikan ini float
            format="%.4f",
            key=feature
        )

st.markdown("---")
st.markdown("Tekan tombol 'Prediksi Diagnosis' untuk melihat hasilnya.")

# Tombol Prediksi
if st.button("Prediksi Diagnosis"):
    # Buat array 30 fitur penuh
    # Inisialisasi dengan nilai rata-rata dari semua fitur
    full_features_array = np.array([feature_means]) # Mulai dengan array 1x30 yang berisi rata-rata

    # Isi nilai untuk 3 fitur yang dipilih dari input pengguna
    for feature_name, value in input_data.items():
        if feature_name in all_feature_names:
            idx = all_feature_names.index(feature_name)
            full_features_array[0, idx] = value # Timpa nilai rata-rata dengan input pengguna

    # Skalakan semua 30 fitur
    scaled_features = scaler.transform(full_features_array)

    # Buat prediksi
    prediction = svm_model.predict(scaled_features)
    prediction_proba = svm_model.decision_function(scaled_features)

    # Dekode prediksi
    predicted_class = le.inverse_transform(prediction)[0]
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
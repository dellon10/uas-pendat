import streamlit as st
import pickle
import numpy as np
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

# Mengambil dataset asli untuk fitting scaler
# Ini memastikan scaler yang digunakan untuk prediksi konsisten dengan scaler pelatihan
try:
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X_original = breast_cancer_wisconsin_diagnostic.data.features
except Exception as e:
    st.error(f"Error saat mengambil dataset untuk fitting scaler: {e}")
    st.stop()

# Inisialisasi dan fitting StandardScaler dengan fitur data pelatihan asli (30 fitur)
# Ini sangat penting untuk penskalaan yang konsisten antara pelatihan dan prediksi
scaler = StandardScaler()
scaler.fit(X_original)

# Inisialisasi LabelEncoder untuk nama target
le = LabelEncoder()
# Fit dengan nilai target yang mungkin (0 dan 1 dari masalah sebelumnya)
le.fit([0, 1]) # 0 untuk Jinak (Benign), 1 untuk Ganas (Malignant)

# Mendefinisikan semua nama fitur seperti yang muncul di dataset (total 30)
all_feature_names = [
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

# Pilih hanya 3 fitur yang akan ditampilkan di UI
selected_feature_names = ['mean_radius', 'mean_texture', 'mean_perimeter']

# Antarmuka Pengguna Streamlit
st.set_page_config(page_title="Prediksi Kanker Payudara", layout="wide")

st.title("🩺 Prediktor Diagnosis Kanker Payudara")
st.markdown("""
Aplikasi ini memprediksi apakah massa payudara bersifat Jinak (non-kanker) atau Ganas (kanker)
berdasarkan fitur-fitur yang dihitung dari gambar digital aspirasi jarum halus (FNA) massa payudara.

Mohon masukkan nilai untuk **3 fitur utama** di bawah ini. Fitur lainnya akan menggunakan nilai default.
""")

# Kolom input untuk fitur
input_data = {}
st.header("Masukkan Nilai Fitur Utama:")

# Gunakan kolom untuk tata letak yang lebih baik
num_cols_display = 3 # Jumlah kolom untuk tampilan input
cols = st.columns(num_cols_display)

for i, feature in enumerate(selected_feature_names):
    with cols[i % num_cols_display]:
        # Mengatur nilai default yang lebih relevan untuk fitur ini
        if feature == 'mean_radius':
            default_value = 17.99
        elif feature == 'mean_texture':
            default_value = 10.38
        elif feature == 'mean_perimeter':
            default_value = 122.80
        else:
            # Fallback jika ada fitur lain yang ditambahkan ke selected_feature_names
            default_value = 0.0 # Ini seharusnya tidak terjadi dengan 3 fitur yang sudah ditetapkan

        # Hanya ada satu panggilan st.number_input per fitur terpilih
        # Gunakan nilai default yang sudah ditentukan di atas
        input_data[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()}",
            value=default_value,
            format="%.4f",
            key=feature # Menggunakan nama fitur sebagai kunci unik
        )


st.markdown("---")
st.markdown("Tekan tombol 'Prediksi Diagnosis' untuk melihat hasilnya.")

# Tombol Prediksi
if st.button("Prediksi Diagnosis"):
    # Buat array 30 fitur penuh
    # Inisialisasi dengan nilai 0.0 untuk semua 30 fitur
    full_features_array = np.zeros((1, len(all_feature_names)))

    # Isi nilai untuk 3 fitur yang dipilih dari input pengguna
    # Pastikan untuk mengisi nilai sesuai dengan indeks fitur di all_feature_names
    for feature_name, value in input_data.items():
        if feature_name in all_feature_names:
            idx = all_feature_names.index(feature_name)
            full_features_array[0, idx] = value


    # Skalakan semua 30 fitur (termasuk yang default)
    scaled_features = scaler.transform(full_features_array)

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

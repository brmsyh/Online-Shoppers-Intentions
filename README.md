# Online Shoppers Purchase Prediction Web App

## Repository Outline
Berikut adalah struktur file dalam repository ini:

1. **README.md**                    - Penjelasan gambaran umum project
2. **app.py**                       - Aplikasi Streamlit untuk EDA dan prediksi
3. **requirements.txt**             - Daftar dependencies Python yang digunakan
4. **best_model_dtc.pkl**           - Model Decision Tree terbaik yang telah dilatih
5. **online_shoppers_intention.csv** - Dataset utama dalam format CSV

## Problem Background
Pada bisnis e-commerce, tingkat kunjungan website yang tinggi tidak selalu berbanding lurus dengan angka penjualan. Banyak pengunjung yang hanya menjelajah tanpa melakukan pembelian. Oleh karena itu, perusahaan perlu memahami karakteristik dan perilaku pengguna yang berpotensi melakukan transaksi. Proyek ini bertujuan untuk membangun model prediktif berbasis data perilaku pengguna yang dapat digunakan untuk memprediksi kemungkinan pembelian secara real-time.

## Project Output
Output dari proyek ini berupa sebuah aplikasi web berbasis Streamlit yang terdiri dari dua bagian utama:

- Exploratory Data Analysis (EDA) untuk menggali insight dari perilaku pengguna.

- Halaman Prediksi untuk memprediksi kemungkinan pembelian dari input pengguna berdasarkan model machine learning yang telah dilatih.`

## Data
Dataset yang digunakan adalah Online Shoppers Intention Dataset dari UCI Machine Learning Repository. 

Dataset ini memiliki:

- Jumlah baris: 12.330 observasi

- Jumlah kolom: 18 fitur + 1 target (Revenue)

- Tipe data: kombinasi numerik dan kategorikal

- Target: Revenue (Boolean), apakah pengunjung melakukan pembelian atau tidak

- Sebelum masuk ke pelatihan model, tipe data kolom target (Revenue) sudah diubah ke tipe data integer

- Terdapat imbalance pada kelas target (Revenue = 15% True)

## Method
Proyek ini menggunakan pendekatan supervised learning untuk menyelesaikan masalah klasifikasi. Beberapa model diuji, seperti KNN, SVM, Random Forest, dan Gradient Boosting. Hasil evaluasi menunjukkan bahwa Decision Tree Classifier memiliki F1 Score tertinggi dan paling stabil untuk baseline model, dan dipilih sebagai model terbaik. Model dilatih menggunakan pipeline Scikit-Learn dan disimpan dalam format `.pkl` menggunakan pickle.

## Stacks
Berikut adalah teknologi dan tools yang digunakan dalam proyek ini:

- Bahasa Pemrograman: Python

    - Libraries:

        - `scikit-learn` (untuk modeling)

        - `pandas`, `numpy` (untuk manipulasi data)

        - `matplotlib`, `seaborn` (untuk visualisasi)

        - `streamlit` (untuk pembuatan web apps)

        - Platform Deployment: Hugging Face Spaces

## Reference
- [UCI Machine Learning Repository – Online Shoppers Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- [Geeks for Geeks](https://www.geeksforgeeks.org/python/python-program-to-convert-camel-case-string-to-snake-case/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Huggingface](https://huggingface.co/spaces/yunidobaheramsyah/online-shoppers-intention-deployment)

---

**Referensi tambahan:**
- [Basic Writing and Syntax on Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Contoh readme](https://github.com/fahmimnalfrzki/Swift-XRT-Automation)
- [Another example](https://github.com/sanggusti/final_bangkit) (**Must read**)
- [Additional reference](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
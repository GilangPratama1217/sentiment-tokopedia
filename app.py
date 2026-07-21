app_code = '''
import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import numpy as np

nltk.download('stopwords', quiet=True)

# Load model
with open('model_nb.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('chi2_selector.pkl', 'rb') as f:
    chi2_selector = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

df = pd.read_csv('tokopedia_preprocessed.csv')
stop_words = set(stopwords.words('indonesian'))

def preprocessing(teks):
    teks = teks.lower()
    teks = re.sub(r\'http\\S+\', \'\', teks)
    teks = re.sub(r\'@\\w+\', \'\', teks)
    teks = re.sub(r\'#\\w+\', \'\', teks)
    teks = re.sub(r\'\\d+\', \'\', teks)
    teks = re.sub(r\'[^\\w\\s]\', \'\', teks)
    teks = re.sub(r\'\\s+\', \' \', teks).strip()
    tokens = teks.split()
    tokens = [k for k in tokens if k not in stop_words]
    return \' \'.join(tokens)

st.set_page_config(page_title="Analisis Sentimen Tokopedia", layout="wide")
st.title("📊 Sistem Analisis Sentimen Ulasan Tokopedia")
st.markdown("Menggunakan metode **Naïve Bayes + TF-IDF + Chi-Square**")
st.divider()

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [
    "🏠 Beranda",
    "📈 Distribusi Sentimen",
    "☁️ WordCloud",
    "🔍 Prediksi Real-Time",
    "📋 Data Ulasan",
    "📊 Evaluasi Model"
])

if menu == "🏠 Beranda":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{len(df):,}")
    col2.metric("Positif", f"{len(df[df.sentimen==\'positif\']):,}")
    col3.metric("Negatif", f"{len(df[df.sentimen==\'negatif\']):,}")
    col4.metric("Netral", f"{len(df[df.sentimen==\'netral\']):,}")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Akurasi Model")
        st.metric("Naïve Bayes + Chi-Square", "83.25%")
        st.write("**Detail Evaluasi:**")
        eval_data = {
            "Kelas": ["Negatif", "Netral", "Positif"],
            "Precision": ["0.79", "0.00", "0.91"],
            "Recall": ["0.96", "0.00", "0.80"],
            "F1-Score": ["0.86", "0.00", "0.85"]
        }
        st.dataframe(pd.DataFrame(eval_data), hide_index=True)
    with col2:
        st.subheader("Informasi Dataset")
        info_data = {
            "Keterangan": ["Total Data", "Data Training", "Data Testing",
                           "Fitur TF-IDF", "Fitur Chi-Square"],
            "Nilai": [str(len(df)), str(int(len(df)*0.8)),
                      str(int(len(df)*0.2)), "5000", "1000"]
        }
        st.dataframe(pd.DataFrame(info_data), hide_index=True)

elif menu == "📈 Distribusi Sentimen":
    st.subheader("Distribusi Sentimen Ulasan Tokopedia")
    col1, col2 = st.columns(2)
    sentimen_count = df[\'sentimen\'].value_counts()
    with col1:
        fig, ax = plt.subplots()
        ax.bar(sentimen_count.index, sentimen_count.values,
               color=[\'#2ecc71\', \'#e74c3c\', \'#f39c12\'])
        ax.set_title("Bar Chart Distribusi Sentimen")
        ax.set_xlabel("Sentimen")
        ax.set_ylabel("Jumlah")
        for i, v in enumerate(sentimen_count.values):
            ax.text(i, v + 5, str(v), ha=\'center\', fontweight=\'bold\')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(sentimen_count.values, labels=sentimen_count.index,
               autopct=\'%1.1f%%\', colors=[\'#2ecc71\', \'#e74c3c\', \'#f39c12\'])
        ax.set_title("Pie Chart Distribusi Sentimen")
        st.pyplot(fig)

elif menu == "☁️ WordCloud":
    st.subheader("WordCloud per Sentimen")
    pilihan = st.selectbox("Pilih Sentimen:", ["positif", "negatif", "netral"])
    teks_gabung = \' \'.join(df[df[\'sentimen\']==pilihan][\'content_clean\'].tolist())
    if teks_gabung.strip():
        wc = WordCloud(width=800, height=400, background_color=\'white\',
                       colormap=\'Greens\' if pilihan==\'positif\'
                       else \'Reds\' if pilihan==\'negatif\' else \'Blues\').generate(teks_gabung)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation=\'bilinear\')
        ax.axis(\'off\')
        ax.set_title(f"WordCloud Sentimen {pilihan.capitalize()}", fontsize=16)
        st.pyplot(fig)
    else:
        st.warning("Tidak ada data untuk sentimen ini.")

elif menu == "🔍 Prediksi Real-Time":
    st.subheader("Prediksi Sentimen Ulasan Baru")
    teks_input = st.text_area("Masukkan ulasan Tokopedia:",
                               placeholder="Contoh: Aplikasinya bagus dan mudah digunakan...",
                               height=150)
    if st.button("🔍 Prediksi Sentimen"):
        if teks_input.strip():
            teks_bersih = preprocessing(teks_input)
            X_input = tfidf.transform([teks_bersih])
            X_input_chi2 = chi2_selector.transform(X_input)
            pred = model.predict(X_input_chi2)
            proba = model.predict_proba(X_input_chi2)[0]
            label = le.inverse_transform(pred)[0]
            emoji = "✅" if label=="positif" else "❌" if label=="negatif" else "➖"
            st.success(f"{emoji} Sentimen: **{label.upper()}**")
            st.divider()
            st.write("**Probabilitas per Kelas:**")
            proba_df = pd.DataFrame({
                "Kelas": le.classes_,
                "Probabilitas": [f"{p*100:.2f}%" for p in proba]
            })
            st.dataframe(proba_df, hide_index=True)
            st.write(f"**Teks setelah preprocessing:** `{teks_bersih}`")
        else:
            st.warning("Masukkan teks ulasan terlebih dahulu!")

elif menu == "📋 Data Ulasan":
    st.subheader("Data Ulasan Tokopedia")
    filter_sentimen = st.multiselect("Filter Sentimen:",
                                      options=[\'positif\', \'negatif\', \'netral\'],
                                      default=[\'positif\', \'negatif\', \'netral\'])
    df_filter = df[df[\'sentimen\'].isin(filter_sentimen)]
    st.write(f"Menampilkan {len(df_filter):,} ulasan")
    st.dataframe(df_filter[[\'content\', \'sentimen\', \'score\']].reset_index(drop=True),
                 height=400)

elif menu == "📊 Evaluasi Model":
    st.subheader("Evaluasi dan Perbandingan Model")
    st.markdown("Perbandingan performa model **Naïve Bayes tanpa Chi-Square** dengan **Naïve Bayes + Chi-Square** pada data uji.")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Naïve Bayes (Tanpa Chi-Square)")
        st.metric("Akurasi", "82.49%")
        eval1 = {
            "Kelas": ["Negatif", "Netral", "Positif"],
            "Precision": ["0.78", "0.00", "0.88"],
            "Recall": ["0.92", "0.00", "0.81"],
            "F1-Score": ["0.84", "0.00", "0.84"],
            "Support": ["189", "18", "187"]
        }
        st.dataframe(pd.DataFrame(eval1), hide_index=True)
    with col2:
        st.subheader("Naïve Bayes + Chi-Square")
        st.metric("Akurasi", "84.01%")
        eval2 = {
            "Kelas": ["Negatif", "Netral", "Positif"],
            "Precision": ["0.79", "0.00", "0.91"],
            "Recall": ["0.96", "0.00", "0.80"],
            "F1-Score": ["0.86", "0.00", "0.85"],
            "Support": ["189", "18", "187"]
        }
        st.dataframe(pd.DataFrame(eval2), hide_index=True)
    st.divider()
    st.success("✅ Chi-Square meningkatkan akurasi sebesar **1.78%** (81.47% → 83.25%)")
    st.write("### Grafik Perbandingan Akurasi")
    fig, ax = plt.subplots(figsize=(8, 5))
    models = [\'Model 1\\n(Tanpa Chi-Square)\', \'Model 2\\n(Dengan Chi-Square)\']
    akurasi = [81.47, 83.25]
    colors = [\'#e74c3c\', \'#2ecc71\']
    bars = ax.bar(models, akurasi, color=colors, width=0.4)
    ax.set_ylim(79, 85)
    ax.set_ylabel("Akurasi (%)")
    ax.set_title("Perbandingan Akurasi Model 1 vs Model 2")
    for bar, val in zip(bars, akurasi):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f\'{val}%\', ha=\'center\', fontweight=\'bold\')
    st.pyplot(fig)
    st.divider()
    st.write("### Confusion Matrix")
    col1, col2 = st.columns(2)
    try:
        import seaborn as sns
        with col1:
            st.write("**Model 1: Tanpa Chi-Square**")
            cm1 = np.array([
                [176, 0, 13],
                [14,  0,  4],
                [41,  0, 146]
            ])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm1, annot=True, fmt=\'d\', cmap=\'Blues\',
                        xticklabels=[\'negatif\', \'netral\', \'positif\'],
                        yticklabels=[\'negatif\', \'netral\', \'positif\'], ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_title("Confusion Matrix Model 1")
            st.pyplot(fig)
        with col2:
            st.write("**Model 2: Dengan Chi-Square**")
            cm2 = np.array([
                [182, 0,  7],
                [14,  0,  4],
                [39,  0, 148]
            ])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm2, annot=True, fmt=\'d\', cmap=\'Greens\',
                        xticklabels=[\'negatif\', \'netral\', \'positif\'],
                        yticklabels=[\'negatif\', \'netral\', \'positif\'], ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_title("Confusion Matrix Model 2")
            st.pyplot(fig)
    except ImportError:
        st.warning("Library seaborn tidak tersedia.")
'''

with open('app.py', 'w') as f:
    f.write(app_code)

print("app.py berhasil diupdate!")

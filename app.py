import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from PIL import Image


# -------------------------------------------
# Fungsi metric custom agar joblib.load sukses
# -------------------------------------------
def business_metric(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    total = tn + fp + fn + tp
    raw_score = ((tp*131 - (fp+tp)*18) - ((tp+fn)*131 - total*18)) / total
    min_score = -14.52
    max_score = 18.0
    normalized_score = (raw_score - min_score) / (max_score - min_score)
    return normalized_score

# ------------------------
# Load trained model
# ------------------------
model = joblib.load('model.sav')

# ------------------------
# Load dataset (for options)
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional-full.csv", sep=";")
    return df

df = load_data()

# ------------------------
# App title & description
# ------------------------
st.title("🎯 Prediksi Term Deposit")
image = Image.open("logo.jpg")
st.image(image, use_container_width=True)
st.write("Masukkan karakteristik nasabah di sidebar:")

# ------------------------
# Sidebar Inputs
# ------------------------
st.sidebar.header("Input Nasabah")

def get_unique(col):
    return sorted(df[col].dropna().unique().tolist())

age = st.sidebar.slider("Umur", int(df["age"].min()), int(df["age"].max()), 30)
job = st.sidebar.selectbox("Pekerjaan", get_unique("job"))
marital = st.sidebar.selectbox("Status Pernikahan", get_unique("marital"))
education = st.sidebar.selectbox("Pendidikan", get_unique("education"))
default = st.sidebar.selectbox("Kredit Macet", get_unique("default"))
housing = st.sidebar.selectbox("Pinjaman Rumah", get_unique("housing"))
loan = st.sidebar.selectbox("Pinjaman Pribadi", get_unique("loan"))
contact = st.sidebar.selectbox("Jenis Kontak", get_unique("contact"))
month = st.sidebar.selectbox("Bulan Kontak", get_unique("month"))
day_of_week = st.sidebar.selectbox("Hari Kontak", get_unique("day_of_week"))
campaign = st.sidebar.selectbox("Jumlah Kontak Kampanye Ini", sorted(df["campaign"].unique()))
pdays = st.sidebar.selectbox("Hari sejak kontak terakhir (999 = belum pernah)", sorted(df["pdays"].unique()))
previous = st.sidebar.selectbox("Jumlah Kontak Sebelumnya", sorted(df["previous"].unique()))
poutcome = st.sidebar.selectbox("Hasil Kampanye Sebelumnya", get_unique("poutcome"))
emp_var_rate = st.sidebar.selectbox("Variasi Tingkat Pekerjaan", sorted(df["emp.var.rate"].unique()))
cons_price_idx = st.sidebar.selectbox("Indeks Harga Konsumen", sorted(df["cons.price.idx"].unique()))
cons_conf_idx = st.sidebar.selectbox("Indeks Kepercayaan Konsumen", sorted(df["cons.conf.idx"].unique()))
euribor3m = st.sidebar.selectbox("Suku Bunga Euribor 3 Bulan", sorted(df["euribor3m"].unique()))
nr_employed = st.sidebar.selectbox("Jumlah Pekerja", sorted(df["nr.employed"].unique()))

# ------------------------
# Generate `pdays_grouped`
# ------------------------
pdays_grouped = "never_contacted" if pdays == 999 else "contacted_before"

# ------------------------
# Terapkan capping
# ------------------------
campaign_cap = df['campaign'].quantile(0.95)
previous_cap = df['previous'].quantile(0.95)
age_cap = df['age'].quantile(0.99)

campaign = min(campaign, campaign_cap)
previous = min(previous, previous_cap)
age = min(age, age_cap)

# ------------------------
# job 'admin.' → 'admin'
# ------------------------
job = 'admin' if job == 'admin.' else job

# ------------------------
# Final input dataframe
# ------------------------
input_df = pd.DataFrame([{
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'campaign': campaign,
    'previous': previous,
    'poutcome': poutcome,
    'emp.var.rate': emp_var_rate,
    'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx,
    'euribor3m': euribor3m,
    'nr.employed': nr_employed,
    'pdays_grouped': pdays_grouped
}])

# ------------------------
# Show Input
# ------------------------
st.subheader("📋 Data Input")
st.write(input_df)

# ------------------------
# Prediction Button
# ------------------------
if st.button("🔮 Prediksi"):
    try:
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("✅ Nasabah diprediksi akan **BERLANGGANAN** Term Deposit.")
        else:
            st.warning("❌ Nasabah diprediksi **TIDAK** akan berlangganan.")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

# ------------------------
# ABOUT THIS APP
# ------------------------
with st.expander("ℹ️ About this App"):
    st.markdown("""
    Aplikasi ini dibuat sebagai bagian dari proyek machine learning untuk memprediksi apakah nasabah akan berlangganan **Term Deposit** berdasarkan berbagai fitur demografis dan historis kampanye marketing.
    
    Model yang digunakan adalah **LightGBM Classifier** yang telah di-*tune* dengan RandomizedSearchCV dan menggunakan metrik bisnis yang disesuaikan dengan konteks perusahaan.

    Aplikasi ini bersifat edukatif dan ditujukan untuk pembelajaran analisis data prediktif.
    """)

# ------------------------
# FEATURE INFORMATION
# ------------------------
with st.expander("📊 Feature Information"):
    st.markdown("""
| Attribute | Data Type | Description |
| --- | --- | --- |
| age | Integer | Usia calon nasabah |
| job | String | Pekerjaan calon nasabah |
| marital | String | Status pernikahan calon nasabah |
| education | String | Status pendidikan calon nasabah |
| default | String | Pernahkah calon nasabah gagal bayar sebelumnya? |
| housing | String | Apakah calon nasabah memiliki cicilan rumah? |
| loan | String | Apakah calon nasabah memiliki pinjaman? |
| contact | String | Jenis komunikasi yang digunakan |
| month | String | Bulan dimana terakhir kali melakukan panggilan dengan nasabah pada tahun ini |
| day_of_week | String | Hari dimana terakhir kali melakukan panggilan dengan nasabah |
| campaign | Integer | Jumlah panggilan yang dilakukan selama kampanye untuk nasabah ini |
| pdays_grouped | String | Status apakah nasabah pernah dihubungi sebelumnya |
| previous | Integer | Jumlah panggilan yang dilakukan sebelum kampanye ini |
| poutcome | String | Hasil dari kampanye sebelumnya |
| emp.var.rate | Float | Employment Variation Rate: perubahan jumlah orang yang bekerja (dalam %) di Portugal |
| cons.price.idx | Float | Consumer Price Index: indikator perubahan harga barang dan jasa (indikator inflasi) |
| cons.conf.idx | Float | Consumer Confidence Index: tingkat kepercayaan masyarakat terhadap ekonomi |
| euribor3m | Float | Euribor 3 Month Rate: tingkat suku bunga antar bank Eropa |
| nr.employed | Float | Indikator tenaga kerja global di Portugal |
| y | String | Apakah calon nasabah memutuskan untuk berlangganan deposito atau tidak |
    """)

# ------------------------
# CREATOR
# ------------------------
with st.expander("👨‍💻 Creator"):
    st.markdown("""
- **Muhammad Rikal Renaldy** — [GitHub](https://github.com/Rikalmuhammad)
- **Akmal Raafid Taufiqurrahman** — [GitHub](https://github.com/akmalraafid25)
    """)

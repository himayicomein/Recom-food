import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# โหลดข้อมูล
@st.cache_data
def load_data():
    file_path = "Lineman_Shops_Final_Clean.csv"  # เปลี่ยนเป็น path ของคุณ
    df = pd.read_csv(file_path)
    df["combined_features"] = df["category"] + " " + df["price_level"]
    return df

df = load_data()

# สร้างโมเดล TF-IDF + Nearest Neighbors
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

nn_model = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="auto")
nn_model.fit(tfidf_matrix)

def format_url(name, url):
    if pd.isna(url) or url.strip() == "-" or url.strip() == "":
        return f"https://www.google.com/search?q={name} ร้านอาหาร"
    return url

def recommend_restaurants(category, price_level, top_n=5):
    filtered_df = df[df["category"].str.contains(category, na=False, case=False)]
    filtered_df = filtered_df[filtered_df["price_level"] == price_level]
    
    results = []
    for _, row in filtered_df.head(top_n).iterrows():
        paragraph = f"""
        <div style='background:#f9f9f9; padding:10px; border-radius:10px; margin-bottom:10px;'>
            <b>🍽 {row['name']}</b><br>
            <small>หมวดหมู่: {row['category']}</small><br>
            <small>ราคา: {row['price_level']}</small><br>
            <a href='{format_url(row['name'], row['url'])}' target='_blank'>🔗 ดูรายละเอียดเพิ่มเติม</a>
        </div>
        """
        results.append(paragraph)
    
    return results if results else ["❌ ไม่พบร้านที่ตรงกับเงื่อนไข"]

# ✅ สร้าง UI
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #16a085;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🍽️ ระบบแนะนำร้านอาหาร</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-title'>🔍 ค้นหาร้านอาหาร</div>", unsafe_allow_html=True)

# 🔹 ตัวเลือกค้นหาตามตัวกรอง
tab1 = st.tabs(["🔍 ค้นหาตามตัวกรอง"])[0]

with tab1:
    category = st.selectbox("🍜 เลือกประเภทอาหาร", df["category"].dropna().unique())
    price_level = st.selectbox("💰 เลือกระดับราคา", df["price_level"].unique())
    
    if st.button("🔍 ค้นหาร้านอาหาร"):
        results = recommend_restaurants(category, price_level)
        for res in results:
            st.markdown(res, unsafe_allow_html=True)

st.sidebar.markdown("### 📢 วิธีใช้")
st.sidebar.write("""
- ใช้ตัวกรองเพื่อค้นหาร้านอาหารตามประเภท และราคา
- หรือพิมพ์ชื่อร้านอาหารเพื่อหา "ร้านที่คล้ายกัน"
- คลิกที่ลิงก์ร้านเพื่อดูรายละเอียด
""")
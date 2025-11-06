import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ------------------ PAGE CONFIG & STYLING ------------------
st.set_page_config(page_title="Titanic Case Dashboard", layout="wide")
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #005c97, #363795);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-container {
    background: #fafafa;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 18px;
    border: 1px solid #ddd;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'> Titanic Data & Scenario Dashboard</h1>", unsafe_allow_html=True)

# ------------------ SIDEBAR NAVIGATION ------------------
st.sidebar.title(" Navigatie")
page = st.sidebar.radio("Ga naar:", [
    " Route Kaart",
    " Overlevingsanalyse",
    " Overlevingsvoorspelling",
    " Scenario Simulatie",
    " Heatmap Inzichten"
])

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")  # Zorg dat train.csv in dezelfde map staat
    # Houd alleen relevante kolommen en verwijder rijen met lege leeftijd
    df = df[['Survived','Pclass','Sex','Age']].dropna()
    return df

df = load_data()

# ------------------ PAGE LOGIC ------------------

# 1. ROUTE KAART
if page == " Route Kaart":
    st.subheader(" Route van de Titanic")
    southampton = [50.9097, -1.4043]
    new_york = [40.7128, -74.0060]
    sinking_point = [41.73, -49.95]

    m = folium.Map(location=[45, -50], zoom_start=3, tiles='cartodbpositron')
    folium.Marker(southampton, popup="Southampton (Start)", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(new_york, popup="New York (Bestemming)", icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker(sinking_point, popup="Zinkpunt", icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine([southampton, sinking_point, new_york], color="orange", weight=3, dash_array='5,10').add_to(m)

    st_folium(m, width=800, height=500)

# 2. OVERLEVINGSANALYSE
elif page == " Overlevingsanalyse":
    st.subheader(" Overlevingsanalyse")
    col1, col2 = st.columns(2)
    with col1:
        fig_class = px.histogram(df, x="Pclass", color="Survived", 
                                 title="Overleving per Reisklasse")
        st.plotly_chart(fig_class, use_container_width=True)
    with col2:
        fig_sex = px.histogram(df, x="Sex", color="Survived", 
                               title="Overleving per Geslacht")
        st.plotly_chart(fig_sex, use_container_width=True)
    fig_age = px.box(df, x="Survived", y="Age", title="Leeftijd vs Overlevingsstatus")
    st.plotly_chart(fig_age, use_container_width=True)

# 3. MACHINE LEARNING MODEL
elif page == " Overlevingsvoorspelling":
    st.subheader(" Voorspel of iemand zou overleven")
    ml_df = df.copy()
    X = ml_df[["Age","Pclass","Sex"]].copy()
    X["Sex"] = LabelEncoder().fit_transform(X["Sex"])
    y = ml_df["Survived"]

    model = RandomForestClassifier()
    model.fit(X, y)

    age = st.slider("Leeftijd", 1, 80, 25)
    sex = st.selectbox("Geslacht", ["male", "female"])
    pclass = st.selectbox("Reisklasse", [1, 2, 3])
    sex_val = 1 if sex == "male" else 0
    prob = model.predict_proba([[age, pclass, sex_val]])[0][1] * 100

    st.markdown(f"<div class='metric-container'> Kans op overleven: <b>{prob:.1f}%</b></div>", unsafe_allow_html=True)

# 4. SCENARIO SIMULATIE
elif page == " Scenario Simulatie":
    st.subheader(" Wat als de Titanic vandaag zou zinken?")
    lifeboats = st.slider("Aantal reddingsboten", 20, 60, 35)
    estimated = min((lifeboats / 20) * 38, 100)
    st.metric(" Geschatte overleving:", f"{estimated:.1f}%")
    st.progress(int(estimated))

# 5. HEATMAP
elif page == " Heatmap Inzichten":
    st.subheader(" Overleving per Leeftijdsgroep")
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,40,60,100],
                            labels=['Kind','Tiener','Volwassene','Middelbare leeftijd','Oudere'])
    heatmap = pd.crosstab(df['AgeGroup'], df['Survived'], normalize='index')*100
    fig = px.imshow(heatmap, text_auto=True, color_continuous_scale="Blues", 
                    title="Overlevingspercentages per Leeftijdsgroep")
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.success("✅ Dashboard actief!")


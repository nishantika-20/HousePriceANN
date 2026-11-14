import os, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Ames | House Price AI", page_icon="🏡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#0b1020;color:#eef2ff}
.block-container{max-width:1250px;padding-top:2rem}
.hero{padding:30px 34px;border:1px solid #334155;border-radius:24px;
background:linear-gradient(120deg,#172554,#312e81 55%,#164e63);margin-bottom:22px}
.hero h1{font-family:Manrope,sans-serif;color:white;font-size:2.4rem;margin:0}
.hero p{color:#c7d2fe;margin:8px 0 0}
.eyebrow{color:#67e8f9;font-size:.72rem;letter-spacing:.16em;font-weight:700}
.panel{background:#111a2e;border:1px solid #26344e;border-radius:18px;padding:20px}
div[data-testid="stForm"]{background:#111a2e;border:1px solid #26344e;border-radius:18px;padding:20px}
div[data-testid="stFormSubmitButton"] button{background:linear-gradient(90deg,#6366f1,#0891b2);color:white;border:0;border-radius:12px;font-weight:700}
.result{background:linear-gradient(135deg,#0f766e,#164e63);border:1px solid #2dd4bf;border-radius:22px;padding:25px;color:white}
.price{font-family:Manrope,sans-serif;font-size:2.5rem;font-weight:800}
.small{color:#a5b4cc;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_files():
    model = tf.keras.models.load_model("artifacts/house_price_ann.keras")
    prep = joblib.load("artifacts/preprocessor.joblib")
    with open("artifacts/metadata.json", encoding="utf-8") as f: meta = json.load(f)
    with open("artifacts/metrics.json", encoding="utf-8") as f: metrics = json.load(f)
    return model, prep, meta, metrics

st.markdown('<div class="hero"><div class="eyebrow">ARTIFICIAL NEURAL NETWORK · REGRESSION</div><h1>House Price Intelligence</h1><p>Explore property details, estimate a sale price, and inspect model performance.</p></div>', unsafe_allow_html=True)

required = ["artifacts/house_price_ann.keras","artifacts/preprocessor.joblib","artifacts/metadata.json","artifacts/metrics.json"]
if not all(os.path.exists(p) for p in required):
    st.error("Model files are missing. First add Kaggle train.csv to data/train.csv and run: python train_model.py")
    st.stop()

model, prep, meta, metrics = load_files()
defaults = meta["defaults"]

predict_tab, performance_tab, about_tab = st.tabs(["🏡 Price Predictor","📊 Model Performance","ℹ️ About"])
with predict_tab:
    a,b = st.columns([1.25,.85], gap="large")
    with a:
        st.subheader("Configure your property")
        st.caption("Selected high-impact features are editable. Other fields use training-set median/mode defaults.")
        with st.form("house_form"):
            c1,c2 = st.columns(2)
            with c1:
                quality = st.slider("Overall quality",1,10,int(round(defaults.get("OverallQual",5))))
                living = st.number_input("Living area (sq. ft.)",300,10000,int(defaults.get("GrLivArea",1500)),50)
                lot = st.number_input("Lot area (sq. ft.)",500,100000,int(defaults.get("LotArea",8000)),100)
                year = st.slider("Year built",int(meta["min_year"]),int(meta["max_year"]),int(defaults.get("YearBuilt",1970)))
                beds = st.number_input("Bedrooms above ground",0,10,int(defaults.get("BedroomAbvGr",3)))
            with c2:
                baths = st.number_input("Full bathrooms",0,8,int(defaults.get("FullBath",2)))
                garage = st.number_input("Garage capacity (cars)",0,5,int(defaults.get("GarageCars",2)))
                basement = st.number_input("Basement area (sq. ft.)",0,6000,int(defaults.get("TotalBsmtSF",800)),50)
                nhoods = meta.get("neighborhoods",[])
                nd = defaults.get("Neighborhood", nhoods[0] if nhoods else "")
                ni = nhoods.index(nd) if nd in nhoods else 0
                neighborhood = st.selectbox("Neighborhood",nhoods,index=ni) if nhoods else nd
                kitchens = meta.get("kitchen_qualities",[])
                kd = defaults.get("KitchenQual", kitchens[0] if kitchens else "")
                ki = kitchens.index(kd) if kd in kitchens else 0
                kitchen = st.selectbox("Kitchen quality",kitchens,index=ki) if kitchens else kd
            submitted = st.form_submit_button("✨ Estimate house price",use_container_width=True)
    with b:
        st.subheader("Your estimate")
        if submitted:
            row = defaults.copy()
            row.update({"OverallQual":quality,"GrLivArea":living,"LotArea":lot,"YearBuilt":year,
                        "BedroomAbvGr":beds,"FullBath":baths,"GarageCars":garage,
                        "TotalBsmtSF":basement,"Neighborhood":neighborhood,"KitchenQual":kitchen})
            frame = pd.DataFrame([row]).drop(columns=["Id","SalePrice"],errors="ignore")
            ready = prep.transform(frame).astype("float32")
            log_price = float(model.predict(ready,verbose=0).reshape(-1)[0])
            price = max(0,float(np.expm1(log_price)))
            st.markdown(f'<div class="result"><div class="eyebrow">ANN PREDICTED SALE PRICE</div><div class="price">${price:,.0f}</div><p>Estimated value based on the trained model.</p></div>',unsafe_allow_html=True)
            st.metric("Estimated price",f"${price:,.2f}")
            st.caption("Academic estimate only. Unentered features use training-set defaults; this is not a professional valuation.")
        else:
            st.markdown('<div class="panel"><div class="eyebrow">READY WHEN YOU ARE</div><h2>Enter property details</h2><p class="small">Submit the form to generate an ANN prediction.</p></div>',unsafe_allow_html=True)
            st.write("The model uses house quality, area, age, rooms, garage, basement, neighbourhood and additional dataset features.")

with performance_tab:
    st.subheader("Held-out test-set evaluation")
    st.caption("ANN and median baseline are evaluated on the same 20% test split.")
    x,y,z = st.columns(3)
    x.metric("ANN · MAE",f"${metrics['ann_mae']:,.0f}")
    y.metric("ANN · RMSE",f"${metrics['ann_rmse']:,.0f}")
    z.metric("ANN · R²",f"{metrics['ann_r2']:.3f}")
    st.markdown("#### Median-price baseline")
    x,y,z = st.columns(3)
    x.metric("Baseline · MAE",f"${metrics['baseline_mae']:,.0f}")
    y.metric("Baseline · RMSE",f"${metrics['baseline_rmse']:,.0f}")
    z.metric("Baseline · R²",f"{metrics['baseline_r2']:.3f}")
    st.caption(f"Epochs: {metrics['epochs_trained']} · Train rows: {metrics['training_rows']} · Test rows: {metrics['test_rows']} · Encoded features: {metrics['encoded_features']}")
    c1,c2 = st.columns(2)
    for col,path,title in [(c1,"artifacts/training_loss.png","Training and validation loss"),(c2,"artifacts/actual_vs_predicted.png","Actual vs predicted")]:
        with col:
            st.write(title)
            if os.path.exists(path): st.image(path,use_container_width=True)

with about_tab:
    st.subheader("Project workflow")
    st.markdown("""
    1. Load Kaggle Ames housing data.
    2. Separate `SalePrice` from the input features.
    3. Impute missing values, encode categories and scale numeric columns.
    4. Train a TensorFlow/Keras feed-forward ANN.
    5. Convert log-price predictions back to US dollars.
    6. Evaluate using MAE, RMSE and R²; compare against a median baseline.
    """)
    st.warning("Dataset geography: Ames, Iowa, USA. Prices are in US dollars. This is an academic demonstration, not a real-estate valuation service.")

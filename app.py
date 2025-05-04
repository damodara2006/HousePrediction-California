import streamlit as st
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import hf_hub_download , login

def dataload():
    df = pd.read_csv('housing.csv')
    df.dropna(inplace=True)
    df = df.join(pd.get_dummies(df["ocean_proximity"])).drop('ocean_proximity', axis=1)
    return df  

def main():
    st.set_page_config(
        page_title="Housing Price Estimator - California 1990",
        page_icon=":house:",
    )
    
    with st.container():
        st.title(":blue[Housing Price Estimator - California 1990]")

    data = dataload()
    input_dect = {}
    
    slider_list = [
        ("📍 Longitude", "longitude"),
        ("📍 Latitude", "latitude"),
        ("🏠 Median Age of Homes", "housing_median_age"),
        ("🏘️ Total Number of Rooms", "total_rooms"),
        ("🛏️ Total Number of Bedrooms", "total_bedrooms"),
        ("👨‍👩‍👧 Population in the Area", "population"),
        ("👪 Number of Households", "households"),
        ("💰 Median Income (×10,000)", "median_income")
    ]

    for label, key in slider_list:
        input_dect[key] = st.slider(
            label=label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        input_dect['<1H OCEAN'] = int(st.checkbox("🚗 <1H OCEAN"))
    with col2:
        input_dect['INLAND'] = int(st.checkbox("🌄 INLAND"))
    with col3:
        input_dect['ISLAND'] = int(st.checkbox("🏝️ ISLAND"))
    with col4:
        input_dect['NEAR BAY'] = int(st.checkbox("🌊 NEAR BAY"))
    with col5:
        input_dect['NEAR OCEAN'] = int(st.checkbox("🌅 NEAR OCEAN"))

    for lab in input_dect:
        if lab in ("total_rooms", "total_bedrooms", "population", "households"):
            input_dect[lab] = np.log(input_dect[lab] + 1)
            
    login(token="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")
    # ✅ Load model from Hugging Face
    model_path = hf_hub_download(
        repo_id="damodaraprakash/house-price-predictor",
        filename="model.pkl"
    )
    model = pickle.load(open(model_path, "rb"))

    if st.button("Predict"):
        prediction = model.predict(pd.DataFrame([input_dect]))
        st.write(f"{prediction[0]:,.2f}")

if __name__ == '__main__':
    main()

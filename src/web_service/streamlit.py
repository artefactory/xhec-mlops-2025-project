# app.py
import streamlit as st
import requests

# URL of your FastAPI backend
API_URL = "http://localhost:8080"

st.set_page_config(page_title="Prediction Dashboard", layout="wide")

st.title(" ML Prediction App")
st.markdown("Use the trained model to make predictions on unseen data.")

st.divider()

# Check API health (get request)
with st.spinner("Checking API health..."):
    try:
        health = requests.get(f"{API_URL}/").json()
        st.success(f" {health['health_check']}")
    except Exception as e:
        st.error(f"Could not reach API: {e}")
        st.stop()

st.divider()

# Trigger predictions (post request)
st.caption("Click the button below to run predictions using the backend model.")

if st.button("Run Predictions"):
    with st.spinner("Running model prediction..."):
        try:
            response = requests.post(f"{API_URL}/predict_all")
            if response.status_code == 201:
                # Just display the raw JSON response
                st.success(" Prediction completed!")
                st.info("🔍 Click the arrows below to expand and view each batch of predictions.")
                st.json(response.json())
            else:
                st.error(f"API returned {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Error calling API: {e}")

st.divider()
st.markdown("Thank you for using the app!")

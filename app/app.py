import streamlit as st
from sensor_har import predict_from_csv
from camera_har import webcam_activity

st.set_page_config(
    page_title="Human Activity Recognition",
    layout="centered"
)

st.title("📡 Human Activity Recognition System")

tab1, tab2 = st.tabs(["📊 Sensor-Based HAR", "📷 Camera-Based HAR"])

# TAB 1 
with tab1:
    st.subheader("Sensor-Based Activity Detection")

    uploaded_file = st.file_uploader(
        "Upload CSV with 561 sensor features",
        type=["csv"]
    )

    if uploaded_file:
        preds = predict_from_csv(uploaded_file)
        st.success("Prediction Complete")
        st.write(preds[:10])

# TAB 2 
with tab2:
    st.subheader("Live Camera Activity Detection")
    st.warning("A separate camera window will open. Press 'Q' to stop.")

    if st.button("Start Webcam"):
        webcam_activity()

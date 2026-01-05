import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sensor_har import predict_from_csv

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="WifMotionSense",
    layout="wide"
)

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 WiFi Human Activity Recognition</h1>
    <h3 style='text-align:center;'>MotionSense</h3>
    <h4 style='text-align:center; color:gray;'>
    Smart Activity Recognition from Wearable Sensors
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("📂 Upload Sensor Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **How it helps in real life**
    - Daily activity tracking
    - Sedentary behavior monitoring
    - Fitness & lifestyle insights
    - Elderly care support
    """
)

# ------------------ MAIN LOGIC ------------------
if uploaded_file:
    predictions = predict_from_csv(uploaded_file)
    df = pd.DataFrame({"Activity": predictions})

    st.success("Activity recognition completed successfully")

    # ------------------ KPI CARDS ------------------
    total_samples = len(df)
    dominant_activity = df["Activity"].value_counts().idxmax()
    sedentary = df["Activity"].isin(["SITTING", "LAYING"]).sum()
    active = total_samples - sedentary

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total_samples)
    col2.metric("Dominant Activity", dominant_activity)
    col3.metric("Sedentary Time", f"{sedentary}")
    col4.metric("Active Time", f"{active}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------ ACTIVITY DISTRIBUTION ------------------
    st.subheader("📊 Activity Distribution")

    activity_counts = df["Activity"].value_counts()

    fig, ax = plt.subplots()
    activity_counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Activity")
    ax.set_title("Recognized Activities")
    st.pyplot(fig)

    # ------------------ LIFESTYLE INSIGHTS ------------------
    st.subheader("💡 Lifestyle Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        if sedentary / total_samples > 0.6:
            st.warning(
                "High sedentary behavior detected. "
                "Consider incorporating more movement breaks."
            )
        else:
            st.success(
                "Balanced activity pattern observed. "
                "Good job maintaining movement!"
            )

    with insight_col2:
        if "WALKING" in activity_counts:
            st.info(
                "Walking activity detected. "
                "Consistent walking supports cardiovascular health."
            )
        else:
            st.info(
                "Limited walking detected. "
                "Light walking can improve overall wellness."
            )

    # ------------------ ACTIVITY TIMELINE PREVIEW ------------------
    st.subheader("🕒 Recent Activity Preview")
    st.dataframe(df.tail(15), use_container_width=True)

else:
    # ------------------ LANDING VIEW ------------------
    st.markdown(
        """
        <div style='text-align:center; padding:80px; color:gray;'>
        <h2>Upload wearable sensor data to get started</h2>
        <p>
        MotionSense transforms raw sensor signals into meaningful
        daily activity insights for health, fitness, and monitoring applications.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

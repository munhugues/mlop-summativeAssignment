import streamlit as st
import requests
import os
from PIL import Image
import io
import time
from src import analytics

# API URL
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Pest Detection System", layout="wide")

st.title("🌾 Agricultural Pest Detection System")
st.write("Upload an image of a pest to identify it.")

# Sidebar
st.sidebar.header("Options")
page = st.sidebar.radio("Navigate", ["Prediction", "Retraining", "Monitoring", "Analytics"])

if page == "Prediction":
    st.header("Pest Identification")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                try:
                    # Prepare file for API
                    # Reset pointer to beginning of file
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Prediction Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Predicted Class", result["class"])
                        with col2:
                            st.metric("Confidence", result["confidence"])
                            
                        # Show all scores if available
                        if "all_scores" in result:
                            st.subheader("Confidence Scores")
                            st.bar_chart(result["all_scores"])
                            
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}. Make sure the API is running.")

elif page == "Retraining":
    st.header("Model Retraining")
    st.write("Upload a ZIP file containing new training data to trigger model retraining.")
    st.info("The ZIP file should contain folders for each class (e.g., 'ants', 'bees') with images inside.")
    
    uploaded_file = st.file_uploader("Upload Training Data (ZIP)", type="zip")
    
    if st.button("Trigger Retraining"):
        if uploaded_file:
            # Step 1: Upload and Trigger
            with st.spinner("Uploading data..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/zip")}
                    response = requests.post(f"{API_URL}/retrain", files=files)
                    
                    if response.status_code == 200:
                        st.success("Upload successful! Retraining started...")
                    else:
                        st.error(f"Failed to start retraining: {response.text}")
                        st.stop()
                except Exception as e:
                     st.error(f"API connection failed: {e}")
                     st.stop()
            
            # Step 2: Poll for Status
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            while True:
                try:
                    status_response = requests.get(f"{API_URL}/retrain/status")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get("status")
                        message = status_data.get("message")
                        
                        if status == "training":
                            status_placeholder.info(f"Status: {message}")
                            progress_bar.progress(50) # Indeterminate progress
                        elif status == "completed":
                            status_placeholder.success(f"Success: {message}")
                            progress_bar.progress(100)
                            break
                        elif status == "failed":
                            status_placeholder.error(f"Error: {message}")
                            progress_bar.empty()
                            break
                        else:
                            status_placeholder.info("Waiting for status...")
                            
                    time.sleep(2) # Poll every 2 seconds
                    
                except Exception as e:
                    status_placeholder.error(f"Error checking status: {e}")
                    break
        else:
            st.warning("Please upload a ZIP file first.")

elif page == "Monitoring":
    st.header("System Monitoring")
    st.write("Real-time system metrics.")
    
    # Fetch metrics from API
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            total_predictions = metrics.get("total_predictions", 0)
            api_status = metrics.get("api_status", "Unknown")
            model_version = metrics.get("model_version", "Unknown")
        else:
            total_predictions = "Error"
            api_status = "Error"
            model_version = "Error"
    except Exception as e:
        st.error(f"Failed to fetch metrics: {e}")
        total_predictions = "N/A"
        api_status = "Offline"
        model_version = "N/A"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("API Status", api_status)
    col2.metric("Model Version", model_version)
    col3.metric("Total Predictions", total_predictions)
    
    if st.button("Refresh Metrics"):
        st.rerun()
    
    st.subheader("Response Time Latency")
    # Dummy data for visualization (Latency tracking would require more complex backend logic)
    chart_data = {"Time": [1, 2, 3, 4, 5], "Latency (ms)": [120, 115, 130, 125, 110]}
    st.line_chart(chart_data, x="Time", y="Latency (ms)")

elif page == "Analytics":
    st.header("📊 Dataset Analytics")
    st.write("Visual insights into the training dataset.")
    
    with st.spinner("Fetching dataset statistics from MongoDB... (This may take a moment)"):
        stats = analytics.fetch_dataset_stats(sample_size=100)
        
    if stats:
        # 1. Class Distribution
        st.subheader("1. Class Distribution")
        st.write("**Story:** This chart shows the balance of the dataset. An imbalanced dataset (some bars much higher than others) can lead to the model being biased towards the majority classes.")
        
        df_counts = stats["class_counts"]
        if not df_counts.empty:
            st.bar_chart(df_counts.set_index("Class"))
            
            # Interpretation
            max_class = df_counts.loc[df_counts['Count'].idxmax()]
            min_class = df_counts.loc[df_counts['Count'].idxmin()]
            st.info(f"💡 **Interpretation:** The dataset is dominated by **{max_class['Class']}** ({max_class['Count']} images), while **{min_class['Class']}** has the fewest ({min_class['Count']} images). Ideally, we should add more images for {min_class['Class']} to balance the training.")
        else:
            st.warning("No class data found.")

        # 2. Image Dimensions
        st.subheader("2. Image Dimensions (Width vs Height)")
        st.write("**Story:** This scatter plot reveals the consistency of image quality. Outliers (very small or very large points) might need preprocessing. A tight cluster indicates standardized input.")
        
        df_sample = stats["sample_data"]
        if not df_sample.empty:
            st.scatter_chart(df_sample, x="Width", y="Height", color="Class")
            
            # Interpretation
            avg_w = df_sample["Width"].mean()
            avg_h = df_sample["Height"].mean()
            st.info(f"💡 **Interpretation:** The average image size is **{int(avg_w)}x{int(avg_h)}**. Most images cluster around this size. If you see points near (0,0), those are low-resolution images that might confuse the model.")
        else:
            st.warning("No sample data available.")

        # 3. Color Analysis
        st.subheader("3. Color Analysis (Mean Red vs Mean Green)")
        st.write("**Story:** Do certain pests have a distinct color profile? For example, are beetles (often dark) distinguishable from aphids (often green) just by color?")
        
        if not df_sample.empty:
            st.scatter_chart(df_sample, x="Mean R", y="Mean G", color="Class")
            
            st.info("💡 **Interpretation:** Clusters in this plot suggest that color is a strong distinguishing feature for those classes. If classes overlap significantly, the model must rely on shape and texture rather than just color.")
            
    else:
        st.error("Could not fetch data. Ensure MongoDB is connected and contains data.")

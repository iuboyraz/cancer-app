import streamlit as st
import pandas as pd
# import numpy as np
# from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import joblib

# Load model
model = joblib.load("DT_final_model.pkl")

# Load image
st.image("https://sph.uth.edu/research/centers/chppr/features/img/coastal%20workflow%20visit%2010x8.jpg?language_id=1", caption="", use_container_width=True)

st.title("Cancer Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload the Excel file:", type=["xlsx"])

# Column names
column_names = [
    "BMP3_SET3_PR2", "TFPI2_SET1", "NDRG4_SET2", "SEPTIN9_SET1_R1",
    "ITGA4_SET1", "SDC2_SET3", "SFRP2_SET1", "ADHFE1_SET1", 
    "HOXA2_SET1", "BMP3_SET3_PR1", "MGMT_SET1", "NUMUNE"
]

if uploaded_file:
    try:
        # Read Excel file and prepare DataFrame
        df = pd.read_excel(uploaded_file, header=0)
        
        if len(df.columns) == len(column_names):
            df.columns = column_names
        else:
            st.error(f"Expected number of columns: {len(column_names)}, Number of columns in the uploaded file: {len(df.columns)}")
            st.stop()

        st.subheader("Uploaded Data")
        st.dataframe(df)

        # Fill NaN values
        df_processed = df.fillna("Unknown")

        # Debug information
        st.write("Debug Information:")
        st.write("DataFrame columns:", df_processed.columns.tolist())
        st.write("DataFrame shape:", df_processed.shape)
        st.write("First few rows:")
        st.write(df_processed.head())

        # Make predictions
        try:
            # Predict
            predictions = model.predict(df_processed)

            # Add predictions
            df['Prediction'] = predictions
            
            # Change the values 1 and 0 in the Prediction column to “Cancer” and "Check"
            df['Prediction'] = df['Prediction'].replace({1: 'Cancer', 0: 'Check'})

            # Get prediction probabilities
            prediction_probs = model.predict_proba(df_processed)

            # Take the highest probability, convert it to a percentage, round it to two digits and add the % sign
            df['Prediction Percentage'] = (prediction_probs.max(axis=1) * 100).round(0)

            st.subheader("Prediction Results")
            st.dataframe(df)

            # Download button
            st.download_button(
                label="Download Results",
                data=df.to_csv(index=False, encoding='utf-8-sig'),
                file_name="prediction_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            # Additional error information
            st.write("\nDetaylı Hata Bilgisi:")
            st.write("Model specifications:", model.feature_names_in_ if hasattr(model, 'feature_names_in_') else "Unknown")
            st.write("Data columns:", df_processed.columns.tolist())
            st.write("Data size:", df_processed.shape)
            st.write("Data types:", df_processed.dtypes)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
        st.write("Error detail:", str(e))
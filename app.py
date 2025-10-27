import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

# --- Configuration ---
MODEL_PATH = 'model.keras'
SCALER_PATH = 'scaler.pkl'

# FIX: Corrected feature names to match the column names in the sklearn breast cancer dataset.
# The format is generally 'STATISTIC FEATURE' or just 'FEATURE'
INPUT_FEATURES = [
    'mean concave points', 'worst concave points', 'worst perimeter', 
    'worst area', 'worst radius', 'mean concavity', 'mean perimeter', 
    'mean area', 'mean radius', 'area error', 'worst concavity', 
    'radius error', 'perimeter error'
]
NUM_FEATURES = len(INPUT_FEATURES) # Should be 13

# --- Artifact Loading and Training Function ---
@st.cache_resource
def load_and_train_model():
    """
    Loads pre-trained model and scaler if they exist. 
    Otherwise, loads the data, trains the model, and saves the artifacts.
    The model is now trained exclusively on the 13 specified features.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        st.success("Loading pre-trained model and scaler...")
        
        try:
            model = load_model(MODEL_PATH)
            # Basic validation check for input shape
            if model.input_shape[1] != NUM_FEATURES:
                st.warning(f"Loaded model input shape ({model.input_shape[1]}) does not match expected features ({NUM_FEATURES}). Re-training...")
                return train_and_save_new_model()
        except Exception:
            st.warning("Model file invalid or corrupted. Re-training...")
            return train_and_save_new_model()

        scaler = joblib.load(SCALER_PATH)
        
        # Load the full dataset, then select only the INPUT_FEATURES subset to get means
        data_all = load_breast_cancer(as_frame=True).frame[INPUT_FEATURES]
        data_means = data_all.mean().to_dict()
        return model, scaler, data_means
    
    # If artifacts are missing, train and save
    return train_and_save_new_model()

def train_and_save_new_model():
    # --- Training Phase (Executed only on first run or if files are missing) ---
    st.info(f"Artifacts not found. Training model now on {NUM_FEATURES} features (this may take a moment)...")

    # Load Data and select only the important features
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df['diagnosis'] = data.target

    # Separate features (X) and target (y) - X only contains the 13 important features
    X = df[INPUT_FEATURES]
    y = df['diagnosis']

    # Store means of the entire dataset for default input values
    data_means = X.mean().to_dict()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- Build the Keras ANN Model (Input shape must match the 13 features) ---
    model = Sequential([
        # Input layer with 13 features
        Dense(64, activation='relu', input_shape=(NUM_FEATURES,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        # Output layer (Sigmoid for binary classification)
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    # Note: Training verbosity is set to 0 to keep Streamlit output clean during training
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

    # Save artifacts
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    st.success(f"Model trained and saved to {MODEL_PATH} and {SCALER_PATH}!")

    return model, scaler, data_means

# --- Streamlit App UI and Logic ---

def main():
    st.set_page_config(page_title="Breast Cancer ANN Predictor", layout="wide")

    # Load/Train Model and Scaler
    model, scaler, data_means = load_and_train_model()

    # Title and Description
    st.title(f"Breast Cancer Classification with ANN ({NUM_FEATURES} Features)")
    st.markdown("""
        This app uses an Artificial Neural Network (ANN) trained on **13 specific cell characteristics** to predict whether a mass is **Malignant** (1) or **Benign** (0).
        
        Adjust the parameters below and click 'Predict' to see the diagnosis.
    """)

    # --- User Input Sidebar ---
    with st.sidebar:
        st.header(f"Input Features ({NUM_FEATURES} Characteristics)")
        st.markdown("Adjust the characteristics used for prediction:")

        input_data = {}

        # Create input sliders for all 13 features
        for feature in INPUT_FEATURES:
            # Use the actual mean of the feature as the default value
            default_value = data_means[feature]
            
            # Estimate a reasonable range for the slider based on data means
            # Setting the range to be 50% below and 50% above the mean value
            min_val = default_value * 0.5
            max_val = default_value * 1.5
            
            # Clean up feature name for display (e.g., 'mean concave points' -> 'Mean Concave Points')
            display_name = feature.replace('_', ' ').title()
            
            # Use a smaller step for better user control on float values
            input_data[feature] = st.slider(
                f"**{display_name}**",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_value),
                step=0.001, 
                key=f'slider_{feature}' 
            )
            
        st.markdown("---")
        if st.button("Run Prediction", type="primary"):
            run_prediction(model, scaler, input_data)


# --- Prediction Execution ---
def run_prediction(model, scaler, input_data):
    st.subheader("Prediction Result")
    
    # 1. Create the 13-feature input array in the correct order
    full_input_array = [input_data[feature] for feature in INPUT_FEATURES]

    # Convert to DataFrame (1 sample, 13 features)
    input_df = pd.DataFrame([full_input_array], columns=INPUT_FEATURES)
    
    # 2. Scale the input data
    scaled_input = scaler.transform(input_df)

    # 3. Make Prediction
    try:
        # Predict probability of Malignant (1)
        prediction_proba = model.predict(scaled_input)[0][0] 
        prediction_class = (prediction_proba > 0.5).astype(int)

        st.markdown("---")
        
        if prediction_class == 1:
            st.error(f"**Diagnosis: Malignant (Cancerous)**")
            st.markdown(f"**Confidence:** {prediction_proba * 100:.2f}%")
            
            st.balloons()
            st.markdown("The model predicts a malignant mass with high probability. This suggests more aggressive cell characteristics.")
        else:
            st.success(f"**Diagnosis: Benign (Non-Cancerous)**")
            # Confidence is 1 - probability of Malignant
            st.markdown(f"**Confidence:** {(1 - prediction_proba) * 100:.2f}%") 
            st.markdown("The model predicts a benign mass. The cell characteristics appear non-threatening.")

        st.markdown("---")
        st.markdown(f"Probability of Malignant (1): **{prediction_proba:.4f}**")
        st.markdown(f"Probability of Benign (0): **{1 - prediction_proba:.4f}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()


# Run the main function
if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Note: load_breast_cancer is only used as a fallback or for quick structure check, 
# but the main logic now uses the uploaded 'data.csv'.
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

# --- Configuration ---
MODEL_PATH = 'model.keras'
SCALER_PATH = 'scaler.pkl'
DATA_PATH = 'data.csv' # Path to the uploaded data file

# Features selected for the model (12 total)
# These features are highly correlated with 'diagnosis' and match the CSV's snake_case naming.
INPUT_FEATURES = [
    'radius_mean', 'perimeter_mean', 'area_mean', 
    'concave points_mean', 'radius_worst', 'perimeter_worst', 
    'area_worst', 'concave points_worst', 'concavity_mean', 
    'compactness_worst', 'concavity_worst', 'compactness_mean'
]
NUM_FEATURES = len(INPUT_FEATURES)

# --- Artifact Loading and Training Function ---
@st.cache_resource
def load_and_train_model():
    """
    Loads pre-trained model and scaler if they exist. 
    Otherwise, loads the data, trains the model, and saves the artifacts.
    Also calculates and returns the full min/max/mean ranges from the data.
    """
    
    try:
        df = pd.read_csv(DATA_PATH)
        # Standardize column names by removing trailing/leading spaces (Crucial for CSVs)
        df.columns = df.columns.str.strip() 
        
        # Check if all required features exist
        missing_features = [f for f in INPUT_FEATURES if f not in df.columns]
        if missing_features:
            st.error(f"Data file is missing required features: {missing_features}")
            st.stop()
            
        # Select only the features used for training
        X = df[INPUT_FEATURES]
        
        # Calculate min, max, and mean for slider ranges
        data_ranges = {
            feature: {
                'min': X[feature].min(),
                'max': X[feature].max(),
                'mean': X[feature].mean()
            } for feature in INPUT_FEATURES
        }
        
    except FileNotFoundError:
        st.error(f"Required data file '{DATA_PATH}' not found. Please ensure it is uploaded.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading or processing data.csv: {e}")
        st.stop()


    # Check for pre-trained model and scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        st.success("Loading pre-trained model and scaler...")
        
        try:
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            
            # Basic validation check for input shape
            if model.input_shape[1] != NUM_FEATURES:
                st.warning(f"Loaded model input shape ({model.input_shape[1]}) does not match expected features ({NUM_FEATURES}). Re-training...")
                return train_and_save_new_model(df, data_ranges)
                
            return model, scaler, data_ranges
            
        except Exception as e:
            st.warning(f"Model or scaler file invalid or corrupted ({e}). Re-training...")
            return train_and_save_new_model(df, data_ranges)

    # If artifacts are missing, train and save
    return train_and_save_new_model(df, data_ranges)

def train_and_save_new_model(df, data_ranges):
    # --- Training Phase (Executed only on first run or if files are missing) ---
    st.info(f"Artifacts not found. Training model now on {NUM_FEATURES} features (this may take a moment)...")

    # The target variable is 'diagnosis' (M=Malignant, B=Benign)
    # Convert 'diagnosis' to numerical: 1 for Malignant, 0 for Benign
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    X = df[INPUT_FEATURES]
    y = df['diagnosis']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- Build the Keras ANN Model ---
    model = Sequential([
        # Input layer with 12 features
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
    # epochs set higher for better performance
    model.fit(X_train_scaled, y_train, epochs=150, batch_size=32, verbose=0) 

    # Save artifacts
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    st.success(f"Model trained and saved to {MODEL_PATH} and {SCALER_PATH}!")

    return model, scaler, data_ranges

# --- Streamlit App UI and Logic ---

def main():
    # st.set_page_config must be the first Streamlit command
    st.set_page_config(page_title="Breast Cancer ANN Predictor", layout="wide")

    # Load/Train Model, Scaler, and Data Ranges
    model, scaler, data_ranges = load_and_train_model()

    # Title and Description
    st.title(f"Breast Cancer Classification with ANN ({NUM_FEATURES} Key Features)")
    st.markdown("""
        This app uses an Artificial Neural Network (ANN) trained on **12 highly correlated cell characteristics** (the most important features) from the dataset to predict whether a mass is **Malignant** (1) or **Benign** (0).
        
        Adjust the parameters in the sidebar and click 'Run Prediction'.
    """)
    st.markdown("---")


    # --- User Input Sidebar ---
    # NOTE: Inputs are created in the sidebar, but the prediction button is moved to the main area
    with st.sidebar:
        st.header(f"Input Features ({NUM_FEATURES} Characteristics)")
        st.markdown("Adjust the characteristics used for prediction:")

        input_data = {}

        # Create input sliders for all 12 features using calculated min/max/mean
        for feature in INPUT_FEATURES:
            feature_range = data_ranges[feature]
            default_value = feature_range['mean']
            
            # Use the actual MIN and MAX from the data for the slider range (The core fix)
            min_val = feature_range['min']
            max_val = feature_range['max']
            
            # Clean up feature name for display 
            display_name = feature.replace('_', ' ').title()
            
            # Determine appropriate step based on data range
            step = 0.001 if max_val < 1.0 else 0.01 if max_val < 10.0 else 0.1

            input_data[feature] = st.slider(
                f"**{display_name}**",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_value),
                step=step, 
                key=f'slider_{feature}',
                format="%.4f" # Ensure precision for display
            )
            
        st.markdown("---")

    # Main area button to trigger prediction
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Run Prediction", type="primary"):
            # Pass the data ranges to the prediction function for display purposes
            run_prediction(model, scaler, input_data, data_ranges)


# --- Prediction Execution ---
def run_prediction(model, scaler, input_data, data_ranges):
    
    # 0. Display Input Summary Table
    st.subheader("Input Feature Summary")
    summary_data = {
        "Feature": [],
        "Input Value": [],
        "Data Mean": [],
        "Data Range": []
    }
    
    for feature in INPUT_FEATURES:
        display_name = feature.replace('_', ' ').title()
        f_range = data_ranges[feature]
        
        summary_data["Feature"].append(display_name)
        summary_data["Input Value"].append(f"{input_data[feature]:.4f}")
        summary_data["Data Mean"].append(f"{f_range['mean']:.4f}")
        summary_data["Data Range"].append(f"[{f_range['min']:.4f} to {f_range['max']:.4f}]")
    
    # Create and display the DataFrame
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.subheader("Prediction Result")
    
    # 1. Create the 12-feature input array in the correct order
    full_input_array = [input_data[feature] for feature in INPUT_FEATURES]

    # Convert to DataFrame (1 sample, 12 features)
    # The feature order MUST be maintained
    input_df = pd.DataFrame([full_input_array], columns=INPUT_FEATURES)
    
    # 2. Scale the input data
    scaled_input = scaler.transform(input_df)

    # 3. Make Prediction
    try:
        # Predict probability of Malignant (1)
        prediction_proba = model.predict(scaled_input, verbose=0)[0][0] 
        prediction_class = (prediction_proba > 0.5).astype(int)

        st.markdown("---")
        
        # Display results
        if prediction_class == 1:
            st.error(f"**Diagnosis: Malignant (Cancerous)**")
            st.markdown(f"**Confidence:** {prediction_proba * 100:.2f}%")
            st.balloons()
            st.markdown("The model predicts a **malignant** mass with high probability. This suggests more aggressive cell characteristics.")
        else:
            st.success(f"**Diagnosis: Benign (Non-Cancerous)**")
            # Confidence is 1 - probability of Malignant
            st.markdown(f"**Confidence:** {(1 - prediction_proba) * 100:.2f}%") 
            st.markdown("The model predicts a **benign** mass. The cell characteristics appear non-threatening.")

        st.markdown("---")
        st.markdown(f"Probability of Malignant (1): **{prediction_proba:.4f}**")
        st.markdown(f"Probability of Benign (0): **{1 - prediction_proba:.4f}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()


# Run the main function
if __name__ == '__main__':
    main()

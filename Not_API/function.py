# ==============================================================================
# 1. IMPORT NECESSARY LIBRARIES
# ==============================================================================
import numpy as np
import pandas as pd
import streamlit as st
import scipy
import lime
import lime.lime_tabular
from datetime import date
import joblib

@st.cache_resource
def load_models():
    """
    Loads the trained machine learning model and LIME background data into memory.
    Uses @st.cache_resource so these heavy files are loaded only once during 
    server startup, preventing redundant memory resets on every UI interaction.
    """
    try:
        best_model = joblib.load("Maternal-Health-Risk-Model/best_hp_model.joblib")
        lime_training_data = np.load("Maternal-Health-Risk-Model/lime_training_data.npy")
        feature_names = joblib.load("Maternal-Health-Risk-Model/feature_names.joblib")

        return best_model, lime_training_data, feature_names
        
    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (STREAMLIT LEVEL)
    # ---------------------------------------------------------
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        st.error("💥 **[CRITICAL FAILURE]** Process aborted during Model Initialization!")

        # 1. Handling Missing Files/Features
        if error_type == "FileNotFoundError" or "file" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** Missing required model file. Details: `{error_raw}`")
            st.stop()

        # 2. Handling Data Type Mismatches
        elif error_type == "TypeError" or "type" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** Incompatible data type encountered during loading. Details: `{error_raw}`")
            st.stop()

        # 3. Handling Value/Shape Mismatch
        elif error_type == "ValueError" or "value" in error_msg or "shape" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** Array shape or value mismatch in model files. Details: `{error_raw}`")
            st.stop()

        # 4. Handling Corrupted Model Objects
        elif error_type == "AttributeError" or "attribute" in error_msg or "EOFError" in error_type:
            st.error(f"🚨 **[SYSTEM ERROR] {error_type}:** Model object is corrupted or missing attributes. Details: `{error_raw}`")
            st.stop()

        # 5. Handling Unfitted Models
        elif error_type == "NotFittedError" or "fitted" in error_msg:
            st.error(f"🚨 **[MODEL ERROR] {error_type}:** Attempting to load an untrained model. Details: `{error_raw}`")
            st.stop()

        # 6. Fallback for any other unknown errors
        else:
            st.error(f"🚨 **[UNKNOWN ERROR] {error_type}:** Unexpected failure during execution. Details: `{error_raw}`")
            st.stop()

best_model, lime_training_data, feature_names = load_models()

# ==============================================================================
# 2. GLOBAL CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42
TARGET_LABELS = ["low risk", "mid risk", "high risk"]
num_columns = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
cat_columns = ["RiskLevel"]

# ==============================================================================
# 3. DATA PREPARATION FUNCTIONS (FEATURE ENGINEERING)
# ==============================================================================
def prepare_data(
    age: int,
    systolic_bp: int,
    diastolic_bp: int,
    blood_glucose: float,
    body_temp: float,
    heart_rate: int
) -> pd.DataFrame:
    """
    Transforms raw patient health input parameters into a structured Pandas DataFrame.
    Prepares the data exactly as required by the preprocessing pipeline.
    """
    try:
        # Pandas requires scalar values to be wrapped in iterables (like lists) 
        # to construct a DataFrame successfully when an explicit index is not provided.
        df_testing = pd.DataFrame({
            "Age": [age],
            "SystolicBP": [systolic_bp],
            "DiastolicBP": [diastolic_bp],
            "BS": [blood_glucose],
            "BodyTemp": [body_temp],
            "HeartRate": [heart_rate]
        })
        
        return df_testing

    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (STREAMLIT LEVEL)
    # ---------------------------------------------------------
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        st.error("💥 **[CRITICAL FAILURE]** Process aborted during Data Preparation!")

        # 1. Handling Missing Columns/Features
        if error_type == "KeyError" or "key" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** Missing required feature/column. Details: `{error_raw}`")
            st.stop()

        # 2. Handling Data Type Mismatches
        elif error_type == "TypeError" or "type" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** Incompatible data type encountered. Details: `{error_raw}`")
            st.stop()

        # 3. Handling Value/Shape Mismatch
        elif error_type == "ValueError" or "value" in error_msg or "shape" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** Input data shape or value mismatch. Details: `{error_raw}`")
            st.stop()

        # 4. Handling Corrupted Model Objects
        elif error_type == "AttributeError" or "attribute" in error_msg:
            st.error(f"🚨 **[SYSTEM ERROR] {error_type}:** Data object is corrupted. Details: `{error_raw}`")
            st.stop()

        # 5. Handling Unfitted Models / State
        elif error_type == "NotFittedError" or "fitted" in error_msg:
            st.error(f"🚨 **[MODEL ERROR] {error_type}:** Attempting to process with an untrained state. Details: `{error_raw}`")
            st.stop()

        # 6. Fallback for any other unknown errors
        else:
            st.error(f"🚨 **[UNKNOWN ERROR] {error_type}:** Unexpected failure during execution. Details: `{error_raw}`")
            st.stop()

# ==============================================================================
# 4. INFERENCE FUNCTION (PREDICTION)
# ==============================================================================
def predict(df_testing: pd.DataFrame, best_model=best_model, target_labels: list = TARGET_LABELS) -> tuple:
    """
    Executes model inference on the prepared DataFrame.
    Extracts the predicted class label, overall confidence, and exact probabilities for all 3 risk levels.
    """
    try:
        # Generate predictions and probabilities using the loaded model
        y_pred = best_model.predict(df_testing)
        y_pred_proba = best_model.predict_proba(df_testing)
        
        # Bulletproof flattening: Different model libraries (like XGBoost vs CatBoost) 
        # return arrays in different dimensions (1D vs 2D). Flattening ensures consistent extraction.
        y_pred_flat = np.array(y_pred).flatten()
        
        # Extract specific values for the single patient instance (index 0)
        pred_index = int(y_pred_flat[0])
        prediction = target_labels[pred_index]
        
        # Extract the highest probability (confidence score) and format it as a percentage string (e.g., "95.5%")
        prediction_conf = ((y_pred_proba.max(axis=1) * 100).round(2).astype(str) + "%")[0]
        
        # Extract the raw floating-point probabilities for each specific risk level class for granular UI display
        low_risk_score  = y_pred_proba[:, 0].astype(float)[0]
        mid_risk_score  = y_pred_proba[:, 1].astype(float)[0]
        high_risk_score = y_pred_proba[:, 2].astype(float)[0]
        
        return prediction, prediction_conf, low_risk_score, mid_risk_score, high_risk_score

    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (STREAMLIT LEVEL)
    # ---------------------------------------------------------
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        st.error("💥 **[CRITICAL FAILURE]** Process aborted during Model Prediction!")

        # 1. Handling Missing Columns/Features
        if error_type == "KeyError" or "key" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** A required data field is missing from the processing pipeline. Details: `{error_raw}`")
            st.stop()

        # 2. Handling Data Type Mismatches
        elif error_type == "TypeError" or "type" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** Incorrect data type passed to the processing function. Details: `{error_raw}`")
            st.stop()

        # 3. Handling Value/Shape Mismatch
        elif error_type == "ValueError" or "value" in error_msg or "shape" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** The input data shape or value does not match the model's requirements. Details: `{error_raw}`")
            st.stop()

        # 4. Handling Corrupted Model Objects
        elif error_type == "AttributeError" or "attribute" in error_msg:
            st.error(f"🚨 **[SYSTEM ERROR] {error_type}:** Internal model architecture error during prediction (corrupted object). Details: `{error_raw}`")
            st.stop()

        # 5. Handling Unfitted Models
        elif error_type == "NotFittedError" or "fitted" in error_msg:
            st.error(f"🚨 **[MODEL ERROR] {error_type}:** The loaded machine learning model is not trained. Details: `{error_raw}`")
            st.stop()

        # 6. Fallback for any other unknown errors
        else:
            st.error(f"🚨 **[UNKNOWN ERROR] {error_type}:** An unexpected error occurred during prediction. Details: `{error_raw}`")
            st.stop()

# ==============================================================================
# 5. EXPLAINABILITY FUNCTION (LIME)
# ==============================================================================
def explain(df_testing: pd.DataFrame, feature_names: list = feature_names, best_model=best_model, X_train_processed: np.ndarray=lime_training_data, target_labels: list = TARGET_LABELS, random_seed: int = RANDOM_SEED) -> str:
    """
    Generates a LIME (Local Interpretable Model-agnostic Explanations) HTML output.
    Bypasses background data transformation as X_train_processed is already encoded.
    """
    try:
        # Isolate the preprocessing step and model from the pipeline 
        # to process background data and predictions separately for LIME compatibility.
        preprocessor = best_model.named_steps["Preprocessing"]
        ml_model = best_model.named_steps["Model"]

        # LIME tabular explainer strictly requires dense numpy arrays; sparse matrices 
        # (often produced by OneHotEncoder) must be explicitly converted to prevent type errors.
        if scipy.sparse.issparse(X_train_processed):
            X_train_processed = X_train_processed.toarray()

        # Use the injected feature names metadata to accurately label the LIME output chart
        features = feature_names

        # Initialize the Tabular Explainer with the pre-processed background data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_processed,
            feature_names=features,
            class_names=target_labels,
            mode="classification",
            random_state=random_seed
        )

        # Isolate the single row and apply the exact same preprocessing pipeline used during training
        status_raw = df_testing.iloc[[0]]
        status_processed = preprocessor.transform(status_raw)

        # Convert the processed single instance to a dense array to match the background data format
        if scipy.sparse.issparse(status_processed):
            status_processed = status_processed.toarray()

        # LIME's explain_instance strictly requires a 1D array for the data_row parameter
        status_data_1d = status_processed[0]

        # Generate the explanation using ONLY the isolated ML model's predict_proba method,
        # bypassing the full pipeline since the data is already preprocessed.
        explanation = explainer.explain_instance(
            data_row=status_data_1d,
            predict_fn=ml_model.predict_proba,
            num_features=10,
            top_labels=len(target_labels) if len(target_labels) > 2 else 1
        )
        
        return explanation.as_html()

    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (STREAMLIT LEVEL)
    # ---------------------------------------------------------
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        st.error("💥 **[CRITICAL FAILURE]** Process aborted during LIME explanation!")

        # 1. Handling Missing Columns/Features
        if error_type == "KeyError" or "key" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** A required data field is missing from the processing pipeline. Details: `{error_raw}`")
            st.stop()

        # 2. Handling Data Type Mismatches
        elif error_type == "TypeError" or "type" in error_msg:
            st.error(f"🚨 **[DATA ERROR] {error_type}:** Incorrect data type passed to the processing function. Details: `{error_raw}`")
            st.stop()

        # 3. Handling LIME Data Mismatch (Crucial for Explainable AI)
        elif error_type == "ValueError" or "value" in error_msg or "shape" in error_msg:
            st.error(f"🚨 **[LIME ERROR] {error_type}:** The input data shape does not match the LIME background dataset. Details: `{error_raw}`")
            st.stop()

        # 4. Handling Corrupted Model/Explainer Objects
        elif error_type == "AttributeError" or "attribute" in error_msg:
            st.error(f"🚨 **[SYSTEM ERROR] {error_type}:** Internal model architecture error during explanation generation. Details: `{error_raw}`")
            st.stop()

        # 5. Handling Unfitted Models
        elif error_type == "NotFittedError" or "fitted" in error_msg:
            st.error(f"🚨 **[MODEL ERROR] {error_type}:** The loaded machine learning model is not trained. Details: `{error_raw}`")
            st.stop()

        # 6. Fallback for any other unknown errors
        else:
            st.error(f"🚨 **[UNKNOWN ERROR] {error_type}:** An unexpected server error occurred during LIME generation. Details: `{error_raw}`")
            st.stop()
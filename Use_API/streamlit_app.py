# ==============================================================================
# 1. IMPORT NECESSARY LIBRARIES
# ==============================================================================
import streamlit as st
import requests
import streamlit.components.v1 as components

# ==============================================================================
# 2. GLOBAL CONFIGURATION
# ==============================================================================
# Remote FastAPI backend URL handling the core machine learning inference operations.
MATERNAL_HEALTH_RISK_BASE_URL = "https://silvio0-maternal-health-api.hf.space"

# ==============================================================================
# 3. STREAMLIT PAGE CONFIGURATION & UI SETUP
# ==============================================================================
st.set_page_config(
    page_title="Maternal Health AI Predictor", 
    page_icon="🩺", 
    layout="centered"
)

# Custom CSS defining the visual hierarchy and color palette for the Hero Section.
# Utilizes a dark theme with Medical Cyan gradient accents to establish a clinical dashboard aesthetic.
st.markdown(
    """
    <style>
    .hero-container { 
        text-align: center; 
        padding-bottom: 2rem; 
    }

    .gradient-text { 
        font-size: 2.8rem; 
        font-weight: 800; 
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 0.5rem; 
    }

    .sub-hook { 
        font-size: 1.2rem; 
        font-weight: 500; 
        color: #A0AEC0; 
        margin-bottom: 2rem; 
    }

    .description-box { 
        background-color: #1E1E2E; 
        padding: 1.5rem 2rem; 
        border-radius: 8px; 
        border-left: 4px solid #00C9FF; 
        text-align: left; 
        font-size: 1rem; 
        line-height: 1.6; 
        color: #E2E8F0; 
        margin-top: 1.5rem; 
    }
    </style>

    <div class="hero-container">
        <div class="gradient-text">🩺 Maternal Health Dashboard</div>
        <div class="sub-hook">AI-powered risk assessment for pregnancy health.</div>
        <div class="description-box">
            Input the patient's clinical metrics below. Our Machine Learning model will 
            evaluate the data to predict potential maternal health risks, ensuring timely and accurate medical insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 4. PATIENT DATA INPUT FORM
# ==============================================================================
# Context manager batching input components into a single submission payload 
# to prevent redundant API calls on every keystroke.
with st.form("maternal_health_form"):
    st.markdown("### 🏥 Patient Clinical Metrics")

    # Layout division for visual symmetry across the form interface.
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            label="Patient Age (Years)",
            value=25,
            min_value=10,
            max_value=70,
            help="Enter the patient's age."
        )

        systolic_bp = st.number_input(
            label="Systolic BP (mmHg)",
            value=120,
            min_value=70,
            max_value=160,
            help="Upper blood pressure metric."
        )

        diastolic_bp = st.number_input(
            label="Diastolic BP (mmHg)",
            value=80,
            min_value=49, 
            max_value=100,
            help="Lower blood pressure metric."
        )

    with col2:
        blood_glucose = st.number_input(
            label="Blood Glucose (mmol/L)",
            value=7.5,
            min_value=6.0,
            max_value=19.0,
            help="Patient's blood sugar level."
        )

        body_temp = st.number_input(
            label="Body Temperature (°F)",
            value=98.6,
            min_value=98.0,
            max_value=103.0,
            help="Core body temperature in Fahrenheit."
        )

        heart_rate = st.number_input(
            label="Heart Rate (BPM)",
            value=75,
            min_value=60, 
            max_value=90,
            help="Resting heart rate in beats per minute."
        )

    submitted = st.form_submit_button("Analyze Health Risk 🚀", use_container_width=True)
    
if submitted:
    # Standardized dictionary mirroring the Pydantic schema required by the remote FastAPI endpoint.
    payload = {
        "age": age,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "blood_glucose": blood_glucose,
        "body_temp": body_temp,
        "heart_rate": heart_rate
    }

    try:
        # ---------------------------------------------------------
        # A. FETCH PREDICTION
        # ---------------------------------------------------------
        with st.spinner("🤖 AI is analyzing patient metrics for health risk..."):
            response_pred = requests.post(f"{MATERNAL_HEALTH_RISK_BASE_URL}/predict", data=payload, timeout=15)

            # Strict validation ensuring the HTTP request completed successfully before JSON parsing.
            if response_pred and response_pred.status_code == 200:
                pred_data = response_pred.json()

                # .get() ensures safe extraction; variables fall back to None if keys are absent.
                prediction = pred_data.get("prediction", "")
                prediction_conf = pred_data.get("prediction_conf", "")
                low_risk_score = pred_data.get("low_risk_score")
                mid_risk_score = pred_data.get("mid_risk_score")
                high_risk_score = pred_data.get("high_risk_score")

                # Explicit 'is not None' checks prevent a legitimate 0.0 probability score 
                # from being erroneously evaluated as a missing metric.
                if not all([prediction, prediction_conf, low_risk_score is not None, mid_risk_score is not None, high_risk_score is not None]):
                    st.error("🚨 **[API ERROR]** The API returned an incomplete prediction response. Missing key metrics.")
                    st.stop()

                # Conversion of raw float probabilities into formatted percentage integers for UI rendering.
                low_risk_score_percentage = round((low_risk_score * 100), 2)
                mid_risk_score_percentage = round((mid_risk_score * 100), 2)
                high_risk_score_percentage = round((high_risk_score * 100), 2)

                # HTML structure defining the visual presentation of the prediction payload.
                st.markdown(
                    f"""
                    <div style='background-color: #1E1E1E; padding: 25px; border-radius: 12px; border-left: 6px solid #00C9FF; box-shadow: 0 4px 8px rgba(0,0,0,0.2); margin-top: 20px;'>
                        <h3 style='color: #00C9FF; margin-top: 0; font-family: sans-serif;'>🎉 Analysis Complete!</h3>
                        <p style='font-size: 16px; color: #E0E0E0; margin-bottom: 5px; font-family: sans-serif;'>Predicted Risk Level:</p>
                        <p style='font-size: 32px; font-weight: bold; color: #92FE9D; margin-top: 0; margin-bottom: 15px; font-family: monospace; text-transform: uppercase;'>
                            {prediction}
                        </p>
                        <hr style='border-color: #333333;'>
                        <p style='color: #A0A0A0; margin-bottom: 5px; font-size: 14px; font-family: sans-serif;'>
                            📊 <strong>Probability Breakdown:</strong> Low: {low_risk_score} | Mid: {mid_risk_score} | High: {high_risk_score}
                        </p>
                        <p style='color: #A0A0A0; margin-bottom: 0; font-size: 14px; font-family: sans-serif;'>
                            🤖 <strong>Model Confidence Score:</strong> {prediction_conf}
                        </p>
                    </div>
                    <br><br>
                    """,
                    unsafe_allow_html=True
                )

                # Visual progress bars representing the continuous probability distribution.
                # Streamlit st.progress inherently requires float arguments strictly bounded between 0.0 and 1.0.
                st.progress(low_risk_score, text=f"Low Risk Probability: {low_risk_score_percentage}%")
                st.progress(mid_risk_score, text=f"Mid Risk Probability: {mid_risk_score_percentage}%")
                st.progress(high_risk_score, text=f"High Risk Probability: {high_risk_score_percentage}%")
            
            else:
                st.error(f"🚨 **API Error [{response_pred.status_code if response_pred else 'Unknown'}]:** {response_pred.text if response_pred else 'No response'}")
                st.stop()

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # B. FETCH LIME EXPLANATION
        # ---------------------------------------------------------
        with st.spinner("🔍 AI is generating the reasoning behind this prediction..."):
            response_explain = requests.post(f"{MATERNAL_HEALTH_RISK_BASE_URL}/explain", data=payload, timeout=30)

            if response_explain and response_explain.status_code == 200:
                explain_data = response_explain.json()
                html_data = explain_data.get("explanation_html", "")

                if not html_data:
                    st.error("🚨 **[EXPLANATION ERROR]** The API successfully responded, but the LIME HTML string is empty.")
                    st.stop()

                # CSS block overriding default LIME styles for compatibility with the Dark Mode layout.
                lime_html_with_bg = f"""
                <div style='background-color: #1E1E1E; padding: 25px; border-radius: 12px; 
                            box-shadow: 0 8px 16px rgba(0,0,0,0.5); border-top: 6px solid #00C9FF; 
                            font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;'>
                    <style>
                        svg text {{ fill: #E0E0E0 !important; }}
                        table {{ color: #E0E0E0 !important; }}
                        .lime-table th, .lime-table td {{ border-color: #444444 !important; }}
                    </style>
                    <h2 style='color: #FFFFFF; margin-top: 0; margin-bottom: 5px; font-weight: 700;'>
                        🧠 AI Decision Breakdown
                    </h2>
                    <p style='color: #A0AEC0; font-size: 15px; margin-top: 0; margin-bottom: 25px; font-weight: 500;'>
                        A transparent view of which specific health metrics positively or negatively impacted the risk prediction.
                    </p>
                    {html_data}
                </div>
                """
                
                # Streamlit component rendering the raw JavaScript-dependent HTML within an isolated iframe architecture.
                components.html(lime_html_with_bg, height=850, scrolling=True)

            else:
                st.error(f"🚨 **Explanation API Error [{response_explain.status_code if response_explain else 'Unknown'}]:** {response_explain.text if response_explain else 'No response'}")
                st.stop()

    # ---------------------------------------------------------
    # EXCEPTION HANDLING & ERROR ROUTING (NETWORK / API LEVEL)
    # ---------------------------------------------------------
    
    # 1. Handling DNS failures, connection refusals, or offline remote servers.
    except requests.exceptions.ConnectionError:
        st.error(f"🚨 **[NETWORK ERROR]** Could not connect to the API server at `{MATERNAL_HEALTH_RISK_BASE_URL}`. Please verify server status.")
        st.stop()
        
    # 2. Handling requests that exceed the designated synchronous timeout bounds.
    except requests.exceptions.Timeout:
        st.error("🚨 **[TIMEOUT ERROR]** The API request exceeded the maximum allowed response time and was safely aborted.")
        st.stop()
        
    # 3. Handling API responses that break JSON format specifications (e.g., 502 HTML error pages).
    except requests.exceptions.JSONDecodeError:
        st.error("🚨 **[DATA ERROR]** Received an invalid JSON response format from the API server.")
        st.stop()
        
    # 4, 5, 6. Routing internal dictionary parsing failures and generalized unhandled exceptions.
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e).lower()
        error_raw = str(e)

        if error_type == "KeyError" or "key" in error_msg:
            st.error(f"🚨 **[PAYLOAD ERROR]** A required data key is missing in the JSON response. Details: `{error_raw}`")
        elif error_type == "TypeError" or "type" in error_msg:
            st.error(f"🚨 **[DATA ERROR]** Incompatible data type encountered during API JSON parsing. Details: `{error_raw}`")
        else:
            st.error(f"🚨 **[UNKNOWN ERROR]** An unexpected system error occurred during API execution. Type: `{error_type}`. Details: `{error_raw}`")
        
        st.stop()

# ==============================================================================
# FOOTER CONFIGURATION
# ==============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown(
    f"""
    <div style='text-align: center; color: #A0AEC0; font-size: 14px; font-family: sans-serif; padding-bottom: 20px;'>
        <p style='margin-bottom: 8px;'>
            🛠️ <strong>Built With:</strong> Frontend UI (Streamlit) | Backend API (FastAPI) | Machine Learning (Scikit-Learn) | Explainable AI (LIME) | Data Processing (Pandas & Numpy)
        </p>
        <p style='margin-bottom: 8px;'>
            👨‍💻 Developed by <strong>Silvio Christian Joe</strong> &nbsp;|&nbsp; <a href='https://github.com/viochris' target='_blank' style='color: #00C9FF; text-decoration: none;'>GitHub (@viochris)</a>
        </p>
        <p style='margin-bottom: 0; font-size: 12px; color: #718096;'>
            API Target: <code>{MATERNAL_HEALTH_RISK_BASE_URL}</code>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
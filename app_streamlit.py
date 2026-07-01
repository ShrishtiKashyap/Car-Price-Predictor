"""
ValuAuto — Car Price Predictor
A polished, single-file Streamlit app for a scikit-learn car resale
price model loaded from a pickle file.

Run with:
    streamlit run app.py
"""

import pickle
import numpy as np
import streamlit as st

# =========================================================
# 1. PAGE CONFIG  (must be the first Streamlit call)
# =========================================================
st.set_page_config(
    page_title="ValuAuto | Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================================================
# 2. MODEL LOADING (cached so the pickle is read only once)
# =========================================================
MODEL_PATH = "car_price_model.pkl"


@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model = None
    model_loaded = False


# =========================================================
# 3. ENCODING MAP
#    Adjust these to EXACTLY match how the categorical
#    columns were encoded when the model was trained.
# =========================================================
FUEL_MAP = {"Petrol": 0, "Diesel": 1, "CNG": 2}
SELLER_MAP = {"Dealer": 0, "Individual": 1}
TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}

# Feature order expected by the model.
# Reorder this list if your model was trained on a different column order.
FEATURE_ORDER = [
    "Present_Price",
    "Kms_Driven",
    "Owner",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Car_Age",
]


# =========================================================
# 4. GLOBAL STYLE
#    A single CSS block injected once, matching a dashboard /
#    instrument-cluster visual language (midnight + amber + teal).
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* ---- App background ---- */
    .stApp {
        background:
            radial-gradient(circle at 10% -10%, rgba(242, 166, 90, 0.16), transparent 40%),
            radial-gradient(circle at 95% 110%, rgba(94, 234, 212, 0.12), transparent 45%),
            linear-gradient(160deg, #0B1120 0%, #10182C 45%, #131C31 100%);
        color: #F5F3EE;
    }

    /* ---- Hide default Streamlit chrome ---- */
    #MainMenu, footer, header {visibility: hidden;}

    /* ---- Header block ---- */
    .hero-eyebrow {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #F2A65A;
        margin-bottom: 6px;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        line-height: 1.2;
        color: #F5F3EE;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #96A1B8;
        margin-bottom: 1.6rem;
    }

    /* ---- Section labels above each group ---- */
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #F5F3EE;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0.4rem 0 0.9rem 0;
    }
    .section-label .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #F2A65A;
        box-shadow: 0 0 8px rgba(242, 166, 90, 0.8);
        display: inline-block;
    }

    /* ---- Card look for bordered containers ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 18px !important;
        padding: 6px 6px 2px 6px;
        backdrop-filter: blur(18px);
        box-shadow: 0 20px 40px -20px rgba(0,0,0,0.55);
        margin-bottom: 1.2rem;
    }

    /* ---- Inputs ---- */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: rgba(11, 17, 32, 0.55) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #F5F3EE !important;
    }
    label {
        color: #C9D1E0 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* ---- Slider accent ---- */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #F2A65A !important;
    }

    /* ---- Predict button ---- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #F2A65A 0%, #E8934A 100%);
        color: #0B1120;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        box-shadow: 0 14px 30px -10px rgba(242, 166, 90, 0.55);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 36px -8px rgba(242, 166, 90, 0.7);
        color: #0B1120;
    }

    /* ---- Result box ---- */
    .result-box {
        margin-top: 1.4rem;
        padding: 1.6rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(94, 234, 212, 0.12), rgba(242, 166, 90, 0.08));
        border: 1px solid rgba(94, 234, 212, 0.28);
        text-align: center;
    }
    .result-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #5EEAD4;
        margin-bottom: 6px;
    }
    .result-value {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: #F5F3EE;
    }

    .footer-note {
        text-align: center;
        color: #5C6785;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. HEADER
# =========================================================
st.markdown('<p class="hero-eyebrow">Resale Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-title">🚗 ValuAuto — Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Enter your vehicle\'s details below and get an instant, '
    'data-driven resale value estimate.</p>',
    unsafe_allow_html=True,
)

if not model_loaded:
    st.warning(
        f"⚠️ Could not find **{MODEL_PATH}** in the app directory. "
        "The form will still render, but predictions are disabled until the "
        "model file is available."
    )

# =========================================================
# 6. INPUT FORM (grouped into two logical cards)
# =========================================================
with st.form("prediction_form"):

    # ---- Group 1: Pricing & Usage ----
    with st.container(border=True):
        st.markdown(
            '<p class="section-label"><span class="dot"></span>Pricing &amp; Usage</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            present_price = st.number_input(
                "Present Price (₹ Lakhs)",
                min_value=0.0,
                step=0.1,
                format="%.2f",
                help="Current ex-showroom price of a new, equivalent car.",
            )
            owner = st.number_input(
                "Owner (0–3)",
                min_value=0,
                max_value=3,
                step=1,
                help="Number of previous owners.",
            )
        with col2:
            kms_driven = st.number_input(
                "Kilometers Driven",
                min_value=0,
                step=500,
                format="%d",
            )
            car_age = st.number_input(
                "Car Age (years)",
                min_value=0,
                step=1,
                format="%d",
            )

    # ---- Group 2: Configuration ----
    with st.container(border=True):
        st.markdown(
            '<p class="section-label"><span class="dot"></span>Configuration</p>',
            unsafe_allow_html=True,
        )
        col3, col4, col5 = st.columns(3)
        with col3:
            fuel_type = st.selectbox("Fuel Type", list(FUEL_MAP.keys()))
        with col4:
            seller_type = st.selectbox("Seller Type", list(SELLER_MAP.keys()))
        with col5:
            transmission = st.selectbox("Transmission", list(TRANSMISSION_MAP.keys()))

    submitted = st.form_submit_button("Estimate My Car's Value")


# =========================================================
# 7. PREDICTION
# =========================================================
if submitted:
    if not model_loaded:
        st.error("Prediction unavailable — the model file was not loaded.")
    else:
        # Build the feature vector in the exact order the model expects.
        features = {
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Owner": owner,
            "Fuel_Type": FUEL_MAP[fuel_type],
            "Seller_Type": SELLER_MAP[seller_type],
            "Transmission": TRANSMISSION_MAP[transmission],
            "Car_Age": car_age,
        }
        input_vector = np.array([[features[col] for col in FEATURE_ORDER]])

        try:
            prediction = model.predict(input_vector)[0]
            prediction = max(prediction, 0)  # guard against negative estimates

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Estimated Resale Value</div>
                    <div class="result-value">₹ {prediction:.2f} Lakhs</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")

# =========================================================
# 8. FOOTER
# =========================================================
st.markdown(
    '<p class="footer-note">Estimates are model-generated guidance, not a guaranteed sale price.</p>',
    unsafe_allow_html=True,
)
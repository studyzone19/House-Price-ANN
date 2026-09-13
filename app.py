import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path
from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "house_price_model.keras"
SCALER_PATH = BASE_DIR / "scaler.pkl"
IMAGE_PATH = BASE_DIR / "assets" / "house.jpg"


# =========================================================
# LOAD MODEL + SCALER
# =========================================================

@st.cache_resource
def load_artifacts():

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


model, scaler = load_artifacts()


# =========================================================
# IMAGE TO BASE64
# =========================================================

def image_to_base64(path):

    if not path.exists():
        return None

    with open(path, "rb") as file:

        return base64.b64encode(
            file.read()
        ).decode("utf-8")


house_image = image_to_base64(IMAGE_PATH)


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

* {
    box-sizing: border-box;
}

.stApp {
    background: #f5f8fc;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* =========================================
   SIDEBAR
========================================= */

section[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #142d50 0%,
        #102641 100%
    );

    width: 280px !important;
}

section[data-testid="stSidebar"] > div {

    padding-top: 25px;
    padding-left: 15px;
    padding-right: 15px;
}


/* Sidebar title */

.sidebar-brand {

    display: flex;
    align-items: center;

    gap: 14px;

    color: white;

    font-size: 21px;
    font-weight: 700;

    line-height: 1.15;

    margin-bottom: 35px;

    padding: 0 10px;
}

.sidebar-brand-icon {

    font-size: 39px;
}


/* Radio navigation */

section[data-testid="stSidebar"]
div[role="radiogroup"] {

    gap: 8px;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] label {

    color: white !important;

    border-radius: 8px;

    padding: 13px 12px;

    font-size: 17px;

    margin-bottom: 4px;

    transition: 0.2s;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] label:hover {

    background: rgba(255,255,255,0.10);
}


/* Hide radio circles */

section[data-testid="stSidebar"]
div[role="radiogroup"]
label > div:first-child {

    display: none;
}


/* =========================================
   HERO
========================================= */

.hero {

    position: relative;

    height: 255px;

    overflow: hidden;

    display: flex;

    align-items: center;

    background: #dcecff;
}

.hero-content {

    position: relative;

    z-index: 5;

    width: 64%;

    padding-left: 52px;
}

.hero-title {

    font-size: 38px;

    font-weight: 750;

    color: #102542;

    line-height: 1.15;

    margin-bottom: 10px;
}

.hero-subtitle {

    font-size: 19px;

    color: #17365c;

    line-height: 1.35;

    max-width: 650px;

    margin-bottom: 17px;
}

.hero-badge {

    display: inline-block;

    background: #c9e1ff;

    color: #17365c;

    padding: 9px 18px;

    border-radius: 25px;

    font-size: 16px;
}


/* Hero image */

.hero-image {

    position: absolute;

    right: 0;

    top: 0;

    width: 52%;

    height: 100%;

    object-fit: cover;

    object-position: center;

    z-index: 1;
}


/* Gradient over image */

.hero-gradient {

    position: absolute;

    left: 38%;

    top: 0;

    width: 30%;

    height: 100%;

    z-index: 3;

    background: linear-gradient(
        90deg,
        #dcecff 0%,
        rgba(220,236,255,0.9) 30%,
        rgba(220,236,255,0) 100%
    );
}


/* =========================================
   CONTENT
========================================= */

.content {

    padding: 22px 26px;

    background: #f5f8fc;
}


/* =========================================
   CARD
========================================= */

.card {

    background: white;

    border: 1px solid #dfe6ef;

    border-radius: 10px;

    padding: 24px;

    box-shadow:
        0 2px 8px rgba(20,40,70,0.06);

    height: 100%;
}

.card-header {

    display: flex;

    align-items: center;

    gap: 13px;

    margin-bottom: 5px;
}

.card-icon {

    font-size: 34px;
}

.card-title {

    font-size: 25px;

    font-weight: 750;

    color: #14213d;
}

.card-subtitle {

    font-size: 15px;

    color: #53657d;

    margin-bottom: 22px;
}


/* =========================================
   INPUTS
========================================= */

label {

    color: #14213d !important;

    font-weight: 600 !important;
}

div[data-baseweb="input"] {

    border-radius: 7px !important;
}

div[data-baseweb="select"] {

    border-radius: 7px !important;
}


/* =========================================
   BUTTON
========================================= */

.stButton > button {

    width: 100%;

    height: 55px;

    background: linear-gradient(
        90deg,
        #1769e0,
        #2878e8
    );

    color: white;

    border: none;

    border-radius: 7px;

    font-size: 18px;

    font-weight: 600;

    margin-top: 12px;
}

.stButton > button:hover {

    background: linear-gradient(
        90deg,
        #125ac5,
        #1769e0
    );

    color: white;
}


/* =========================================
   RESULT
========================================= */

.result-card {

    background: linear-gradient(
        135deg,
        #f4fff7,
        #e9faef
    );

    border: 1px solid #d2eddb;

    border-radius: 10px;

    padding: 25px;

    margin-bottom: 20px;
}

.result-top {

    display: flex;

    align-items: center;

    gap: 18px;
}

.money {

    width: 68px;
    height: 68px;

    border-radius: 50%;

    background: #42b96b;

    display: flex;

    align-items: center;
    justify-content: center;

    color: white;

    font-size: 38px;

    font-weight: 700;
}

.result-title {

    font-size: 23px;

    font-weight: 750;

    color: #14213d;
}

.price {

    margin-left: 86px;

    margin-top: 8px;

    font-size: 42px;

    font-weight: 800;

    color: #075b2d;
}

.result-text {

    margin-left: 86px;

    margin-top: 12px;

    background: #d9f5e2;

    color: #12602e;

    border-radius: 7px;

    padding: 12px;

    font-size: 15px;

    line-height: 1.5;
}


/* =========================================
   SUMMARY
========================================= */

.summary-title {

    font-size: 17px;

    font-weight: 750;

    color: #14213d;

    margin-bottom: 10px;
}

.summary {

    width: 100%;

    border-collapse: collapse;

    background: white;

    border: 1px solid #d8e0ea;

    border-radius: 7px;

    overflow: hidden;
}

.summary td {

    padding: 8px 11px;

    border-bottom: 1px solid #d8e0ea;

    font-size: 15px;

    color: #172b4d;
}

.summary tr:last-child td {

    border-bottom: none;
}

.summary td:first-child {

    width: 48%;

    font-weight: 500;
}


/* =========================================
   FOOTER
========================================= */

.custom-footer {

    background: #10233e;

    height: 62px;

    color: white;

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 0 30px;

    font-size: 15px;
}

.footer-right {

    color: #d8dfeb;
}

.heart {

    color: #ff4545;
}


/* =========================================
   INFO PAGE
========================================= */

.info-card {

    background: white;

    border: 1px solid #dfe6ef;

    border-radius: 10px;

    padding: 30px;

    box-shadow:
        0 2px 8px rgba(20,40,70,0.05);
}

.info-title {

    font-size: 28px;

    font-weight: 750;

    color: #14213d;

    margin-bottom: 15px;
}

.info-text {

    color: #42546d;

    font-size: 16px;

    line-height: 1.7;
}

</style>
""")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.html("""
    <div class="sidebar-brand">

        <div class="sidebar-brand-icon">
            🏠
        </div>

        <div>
            House Price<br>
            Prediction
        </div>

    </div>
    """)

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Predict",
            "About",
            "Dataset Info"
        ],
        format_func=lambda x: {
            "Home": "🏠   Home",
            "Predict": "📈   Predict",
            "About": "ⓘ   About",
            "Dataset Info": "🗄️   Dataset Info"
        }[x],
        label_visibility="collapsed"
    )


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:

    st.session_state.prediction = None


if "input_values" not in st.session_state:

    st.session_state.input_values = None


# =========================================================
# HOME
# =========================================================

if page == "Home":

    # -----------------------------------------
    # HERO
    # -----------------------------------------

    if house_image:

        st.html(f"""
        <div class="hero">

            <div class="hero-content">

                <div class="hero-title">
                    Find the Value of Your Dream Home
                </div>

                <div class="hero-subtitle">
                    Enter the details of a house and get an estimated
                    price using Machine Learning (ANN)
                </div>

                <div class="hero-badge">
                    Accurate &nbsp; • &nbsp;
                    Fast &nbsp; • &nbsp;
                    Easy to Use
                </div>

            </div>

            <img
                class="hero-image"
                src="data:image/jpeg;base64,{house_image}"
            >

            <div class="hero-gradient"></div>

        </div>
        """)

    else:

        st.error(
            "House image nahi mili. assets/house.jpg check karo."
        )


    # -----------------------------------------
    # MAIN CONTENT
    # -----------------------------------------

    st.html('<div class="content">')

    left, right = st.columns(
        [1.08, 0.92],
        gap="large"
    )


    # =====================================================
    # LEFT SIDE
    # =====================================================

    with left:

        st.html("""
        <div class="card">

            <div class="card-header">

                <div class="card-icon">
                    🏠
                </div>

                <div class="card-title">
                    Enter House Details
                </div>

            </div>

            <div class="card-subtitle">
                Fill in the features below to predict the house price
            </div>

        """)

        # Row 1

        c1, c2 = st.columns(2)

        with c1:

            longitude = st.number_input(
                "Longitude",
                value=-122.23,
                format="%.2f"
            )

        with c2:

            latitude = st.number_input(
                "Latitude",
                value=37.88,
                format="%.2f"
            )


        # Row 2

        c1, c2 = st.columns(2)

        with c1:

            housing_median_age = st.number_input(
                "Housing Median Age (years)",
                min_value=0.0,
                value=25.0,
                step=1.0
            )

        with c2:

            total_rooms = st.number_input(
                "Total Rooms",
                min_value=0.0,
                value=3000.0,
                step=100.0
            )


        # Row 3

        c1, c2 = st.columns(2)

        with c1:

            total_bedrooms = st.number_input(
                "Total Bedrooms",
                min_value=0.0,
                value=500.0,
                step=50.0
            )

        with c2:

            population = st.number_input(
                "Population",
                min_value=0.0,
                value=1200.0,
                step=100.0
            )


        # Row 4

        c1, c2 = st.columns(2)

        with c1:

            households = st.number_input(
                "Households",
                min_value=0.0,
                value=400.0,
                step=50.0
            )

        with c2:

            median_income = st.number_input(
                "Median Income",
                min_value=0.0,
                value=4.5,
                step=0.1
            )


        # Ocean proximity

        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            [
                "-- Select --",
                "<1H OCEAN",
                "INLAND",
                "ISLAND",
                "NEAR BAY",
                "NEAR OCEAN"
            ]
        )


        # Button

        predict_button = st.button(
            "📊  Predict House Price",
            use_container_width=True
        )

        st.html("</div>")


    # =====================================================
    # RIGHT SIDE
    # =====================================================

    with right:

        # -----------------------------------------
        # PREDICT
        # -----------------------------------------

        if predict_button:

            if ocean_proximity == "-- Select --":

                st.warning(
                    "Please select Ocean Proximity."
                )

            else:

                # Input dataframe

                input_data = pd.DataFrame(
                    [{
                        "longitude": longitude,
                        "latitude": latitude,
                        "housing_median_age": housing_median_age,
                        "total_rooms": total_rooms,
                        "total_bedrooms": total_bedrooms,
                        "population": population,
                        "households": households,
                        "median_income": median_income,
                        "ocean_proximity": ocean_proximity
                    }]
                )


                # One-hot encoding

                input_data = pd.get_dummies(
                    input_data,
                    columns=["ocean_proximity"],
                    dtype=int
                )


                # Same columns used during training

                feature_columns = [

                    "longitude",
                    "latitude",
                    "housing_median_age",
                    "total_rooms",
                    "total_bedrooms",
                    "population",
                    "households",
                    "median_income",

                    "ocean_proximity_<1H OCEAN",
                    "ocean_proximity_INLAND",
                    "ocean_proximity_ISLAND",
                    "ocean_proximity_NEAR BAY",
                    "ocean_proximity_NEAR OCEAN"
                ]


                # Arrange columns

                input_data = input_data.reindex(
                    columns=feature_columns,
                    fill_value=0
                )


                # Scaling

                input_scaled = scaler.transform(
                    input_data
                )


                # ANN prediction

                prediction = model.predict(
                    input_scaled,
                    verbose=0
                )

                price = float(
                    prediction[0][0]
                )

                price = max(
                    0,
                    price
                )


                # Save result

                st.session_state.prediction = price

                st.session_state.input_values = {

                    "Longitude": longitude,

                    "Latitude": latitude,

                    "Housing Median Age": housing_median_age,

                    "Total Rooms": total_rooms,

                    "Total Bedrooms": total_bedrooms,

                    "Population": population,

                    "Households": households,

                    "Median Income": median_income,

                    "Ocean Proximity": ocean_proximity
                }


        # -----------------------------------------
        # DEFAULT RESULT
        # -----------------------------------------

        if st.session_state.prediction is None:

            display_price = 250000

        else:

            display_price = st.session_state.prediction


        # -----------------------------------------
        # RESULT CARD
        # -----------------------------------------

        st.html(f"""
        <div class="result-card">

            <div class="result-top">

                <div class="money">
                    $
                </div>

                <div class="result-title">
                    Predicted House Price
                </div>

            </div>

            <div class="price">
                ${display_price:,.0f}
            </div>

            <div class="result-text">

                This is an estimated price based on the
                input features using an Artificial Neural
                Network model.

            </div>

        </div>
        """)


        # -----------------------------------------
        # INPUT SUMMARY
        # -----------------------------------------

        st.html("""
        <div class="summary-title">
            Input Summary
        </div>
        """)


        # If prediction has happened

        if st.session_state.input_values:

            values = st.session_state.input_values

        else:

            values = {

                "Longitude": longitude,

                "Latitude": latitude,

                "Housing Median Age": housing_median_age,

                "Total Rooms": total_rooms,

                "Total Bedrooms": total_bedrooms,

                "Population": population,

                "Households": households,

                "Median Income": median_income,

                "Ocean Proximity": ocean_proximity
            }


        table = """
        <table class="summary">
        """


        for key, value in values.items():

            if isinstance(value, float):

                if key in [
                    "Longitude",
                    "Latitude",
                    "Median Income"
                ]:

                    value = f"{value:.2f}"

                else:

                    value = f"{value:,.0f}"


            table += f"""
            <tr>
                <td>{key}</td>
                <td>{value}</td>
            </tr>
            """


        table += "</table>"


        st.html(table)


    st.html("</div>")


# =========================================================
# PREDICT PAGE
# =========================================================

elif page == "Predict":

    st.html("""
    <div class="content">

        <div class="info-card">

            <div class="info-title">
                📈 House Price Prediction
            </div>

            <div class="info-text">

                Enter the required house features on the
                Home page and click the
                <b>Predict House Price</b> button.

                <br><br>

                The trained Artificial Neural Network
                will process the input and generate
                an estimated house price.

            </div>

        </div>

    </div>
    """)


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.html("""
    <div class="content">

        <div class="info-card">

            <div class="info-title">
                ℹ️ About This Project
            </div>

            <div class="info-text">

                <b>House Price Prediction</b> is a
                Machine Learning based application
                developed using an Artificial Neural
                Network (ANN).

                <br><br>

                The model uses different house features
                such as location, rooms, bedrooms,
                population, households and median income.

                <br><br>

                <b>Technologies Used</b>

                <br><br>

                🐍 Python<br>
                🧠 TensorFlow / Keras<br>
                📊 Pandas<br>
                📈 Scikit-learn<br>
                🎨 Streamlit

            </div>

        </div>

    </div>
    """)


# =========================================================
# DATASET PAGE
# =========================================================

elif page == "Dataset Info":

    st.html("""
    <div class="content">

        <div class="info-card">

            <div class="info-title">
                🗄️ Dataset Information
            </div>

            <div class="info-text">

                <b>Dataset:</b>
                California Housing Dataset

                <br><br>

                <b>Input Features:</b>

                <br><br>

                • Longitude<br>
                • Latitude<br>
                • Housing Median Age<br>
                • Total Rooms<br>
                • Total Bedrooms<br>
                • Population<br>
                • Households<br>
                • Median Income<br>
                • Ocean Proximity

                <br><br>

                <b>Target Variable:</b>

                <br>

                Median House Value

            </div>

        </div>

    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="custom-footer">

    <div>
        <b>House Price Prediction</b>
    </div>

    <div class="footer-right">
        Built with
        <span class="heart">❤️</span>
       Ram
    </div>

</div>
""")
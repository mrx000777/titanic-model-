import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

model = joblib.load("titanic_model.pkl")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 40px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 30px;
        background: linear-gradient(135deg, #172554, #1e40af);
        border: 1px solid #3b82f6;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
    }

    .hero h1 {
        color: white;
        font-size: 44px;
        margin-bottom: 10px;
        font-weight: 800;
    }

    .hero p {
        color: #bfdbfe;
        font-size: 18px;
        margin: 0;
    }

    .section-title {
        color: white;
        font-size: 27px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .stCaption {
        color: #cbd5e1 !important;
    }

    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div {
        background-color: #1e293b;
        color: white;
        border: 1px solid #475569;
        border-radius: 10px;
    }

    .stSelectbox input,
    .stNumberInput input {
        color: white !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.35);
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        transform: translateY(-2px);
    }

    .survived-card {
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    }

    .not-survived-card {
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2);
    }

    .result-title {
        color: white;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .probability {
        color: #f8fafc;
        font-size: 22px;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #475569;
        padding: 15px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: white !important;
    }

    hr {
        border-color: #475569;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Enter the passenger details below and let the machine learning model "
    "predict the survival outcome."
)

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    sibsp = st.number_input(
        "Number of Siblings/Spouses",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    cabin_known = st.selectbox(
        "Cabin Information Available?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=32.0,
        step=1.0
    )

    parch = st.number_input(
        "Number of Parents/Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["C", "Q", "S"],
        format_func=lambda x: {
            "C": "Cherbourg (C)",
            "Q": "Queenstown (Q)",
            "S": "Southampton (S)"
        }[x]
    )

st.divider()

sex_value = 0 if sex == "Male" else 1

embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_value],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "CabinKnown": [cabin_known],
    "Embarked_Q": [embarked_q],
    "Embarked_S": [embarked_s]
})

predict_button = st.button(
    "🔮 Predict Survival",
    use_container_width=True
)

if predict_button:
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    probability_percent = probability * 100

    st.divider()

    if prediction[0] == 1:
        st.markdown(
            f"""
            <div class="survived-card">
                <div class="result-title">
                    🎉 Predicted to Survive
                </div>
                <div class="probability">
                    Survival Probability: {probability_percent:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="not-survived-card">
                <div class="result-title">
                    ⚠️ Predicted Not to Survive
                </div>
                <div class="probability">
                    Survival Probability: {probability_percent:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📊 Model Confidence")

    st.progress(int(probability_percent))

    st.write(
        f"The model estimates a **{probability_percent:.1f}% probability of survival**."
    )

    st.markdown("### 📋 Passenger Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Passenger Class", pclass)
        st.metric("Age", age)
        st.metric("Sex", sex)

    with summary_col2:
        st.metric("Fare", f"${fare:.2f}")
        st.metric("Siblings / Spouses", sibsp)
        st.metric("Parents / Children", parch)

    with summary_col3:
        st.metric(
            "Cabin Known",
            "Yes" if cabin_known == 1 else "No"
        )

        st.metric(
            "Embarkation",
            embarked
        )

        st.metric(
            "Prediction",
            "Survived" if prediction[0] == 1 else "Not Survived"
        )

st.divider()

st.caption(
    "Titanic Survival Prediction By Manish Thakur"
)
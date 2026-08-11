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
    .stApp {
        background-color: #f8fafc;
    }

    .hero {
        padding: 35px;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 30px;
        background-color: #0f172a;
        color: white;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        margin: 0;
        opacity: 0.8;
    }

    .survived-card {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        background-color: #dcfce7;
        border: 1px solid #86efac;
        margin-top: 25px;
    }

    .not-survived-card {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        background-color: #fee2e2;
        border: 1px solid #fca5a5;
        margin-top: 25px;
    }

    .result-title {
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .probability {
        font-size: 22px;
        font-weight: bold;
    }

    .section-title {
        font-size: 25px;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>🚢 Titanic Survival Predictor</h1>
        <p>Machine Learning powered passenger survival prediction</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">👤 Passenger Information</div>',
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
    "Titanic Survival Prediction | Logistic Regression | "
    "Python • Pandas • Scikit-learn • Streamlit"
)
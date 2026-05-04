import streamlit as st
st.set_page_config(page_title="Injury Type Prediction", layout="centered")

import numpy as np
import pandas as pd
import urllib.request
import joblib

# -----------------------------
# Load model + preprocessing
# -----------------------------
def load_model(url, filename):
    urllib.request.urlretrieve(url, filename)
    return joblib.load(filename)

final_lr = load_model(
    "https://raw.githubusercontent.com/dinushan1998/Model_3/main/lr_model.pkl",
    "final_lr"
)
le = load_model(
    "https://raw.githubusercontent.com/dinushan1998/Model_3/main/m3_label_encoder_injury_type.pkl",
    "m3_label_encoder_injury_type.pkl"
)


# -----------------------------
# Mappings
# -----------------------------
X_train_columns = ['main_activity_Construction of buildings',
       'main_activity_Specialised activities',
       'Kind_group_Contact with electricity',
       'Kind_group_Contact with machinery', 'Kind_group_Exposed to explosion',
       'Kind_group_Exposed to fire',
       'Kind_group_Exposure to harmful substance',
       'Kind_group_Fall from height', 'Kind_group_Injured by an animal',
       'Kind_group_Lifting and handling injuries',
       'Kind_group_Physical assault', 'Kind_group_Slip, trip, fall same level',
       'Kind_group_Struck against', 'Kind_group_Struck by moving vehicle',
       'Kind_group_Struck by object',
       'Kind_group_Trapped by something collapsing', 'body_part_Back',
       'body_part_Ear', 'body_part_Eye', 'body_part_Finger or fingers',
       'body_part_Foot', 'body_part_General locations', 'body_part_Hand',
       'body_part_Head', 'body_part_Lower limb', 'body_part_Neck',
       'body_part_Other parts of face', 'body_part_Several head locations',
       'body_part_Several locations', 'body_part_Several lower limb locations',
       'body_part_Several torso locations',
       'body_part_Several upper limb locations', 'body_part_Toe',
       'body_part_Trunk', 'body_part_Unknown locations',
       'body_part_Upper limb', 'body_part_Wrist']


# -----------------------------
# UI
# -----------------------------
st.title("❤️‍🩹 Injury Type Prediction")

main_activity = st.selectbox("Main Activity", [
    'Construction of buildings', 'Civil engineering', 'Specialised activities'
])

Kind_group = st.selectbox("Kind_group", ['Contact with electricity',
       'Contact with machinery', 'Exposed to explosion',
       'Exposed to fire', 'Exposure to harmful substance',
       'Fall from height', 'Lifting and handling injuries',
       'Physical assault', 'Slip, trip, fall same level',
       'Struck against', 'Struck by moving vehicle', 'Struck by object',
       'Trapped by something collapsing', 'Injured by an animal', 'Another kind of accident'])

body_part = st.selectbox("Body Part", [
    'Ankle','Back','Ear','Eye','Finger or fingers','Foot','General locations','Hand','Head','Lower limb','Neck',
    'Other parts of face','Several head locations','Several locations','Several lower limb locations','Several torso locations',
    'Several upper limb locations','Toe','Trunk','Upper limb','Wrist','Unknown locations'])


# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):

    ## Step 1: build input DataFrame (RAW format)
    input_data = {
        "main_activity": main_activity,
        "Kind_group": Kind_group,
        "body_part": body_part
    }


    feature_cols = X_train_columns  # must be saved before training

    row = pd.DataFrame(columns=feature_cols)
    row.loc[0] = 0  # initialize all zeros


    ## Mapping

    # helper function
    def set_feature(prefix, value):
        col = f"{prefix}_{value}"
        if col in row.columns:
            row[col] = 1

    set_feature("main_activity", input_data["main_activity"])
    set_feature("Kind_group", input_data["Kind_group"])
    set_feature("body_part", input_data["body_part"])


    # Step 2: preprocess
    X_input = row

    # Step 3: prediction
    prediction = final_lr.predict(X_input)
    proba = max(final_lr.predict_proba(X_input)[0])

    # Step 4: decode label
    pred_label = le.inverse_transform([prediction[0]])
    confidence = proba * 100

    # -----------------------------
    # Output
    # -----------------------------
    st.success(f"Prediction: {pred_label[0]}")
    st.info(f"Confidence: {confidence:.2f}%")

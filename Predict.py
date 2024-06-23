import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from streamlit_option_menu import option_menu
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
diabetes_model = pickle.load(
    open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
dia = pd.read_csv(
    'C:/Users/heman/OneDrive/Desktop/Disease Prediction/dataset/diabetes_dataset.csv')
dia = dia.fillna(0)

# Ensure the columns match what the model expects
dia_x = dia.drop(columns='diabetes', axis=1)

heart_disease_model = pickle.load(
    open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
heart = pd.read_csv(
    'C:/Users/heman/OneDrive/Desktop/Disease Prediction/dataset/heart.csv')
heart = heart.fillna(0)
heart_x = heart.drop(columns='target', axis=1)

parkinsons_model = pickle.load(
    open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
parkinsons = pd.read_csv(
    'C:/Users/heman/OneDrive/Desktop/Disease Prediction/dataset/parkinsons.csv')
parkinsons = parkinsons.fillna(0)
parkinsons_x = parkinsons.drop(columns='status', axis=1)

typhoid_model = pickle.load(
    open(f'{working_dir}/saved_models/typhoid_model.sav', 'rb'))
typhoid = pd.read_csv(
    'C:/Users/heman/OneDrive/Desktop/Disease Prediction/dataset/Typhoid1.csv')
typhoid = typhoid.fillna(0)
typhoid_x = typhoid.drop(columns='Output', axis=1)

covid19_model = pickle.load(
    open(f'{working_dir}/saved_models/covid19_model.sav', 'rb'))
dfc = pd.read_csv(
    'C:/Users/heman/OneDrive/Desktop/Disease Prediction/dataset/covid19-dataset.csv')
dfc = dfc.fillna(0)
dfc_x = dfc.drop(columns='Outcome', axis=1)

# breast_cancer_model = pickle.load(
#    open(f'{working_dir}/saved_models/breast_cancer.sav', 'rb'))
# dfc = pd.read_csv(
#    'C:/Users/heman/OneDrive/Desktop/Disease Prediction/dataset/covid19-dataset.csv')
# dfc = dfc.fillna(0)
# dfc_x = dfc.drop(columns='Outcome', axis=1)


lung_cancer_model = pickle.load(
    open(f'{working_dir}/saved_models/lung_cancer_model.sav', 'rb'))
dfl = pd.read_csv(
    'C:/Users/heman/OneDrive/Desktop/Disease Prediction/dataset/lung cancer.csv')
dfl = dfl.fillna(0)
dfl_x = dfl.drop(columns='Outcome', axis=1)

bp_model = pickle.load(
    open(f'{working_dir}/saved_models/blood_pressure.sav', 'rb'))
bp = pd.read_csv(
    'C:/Users/heman/OneDrive/Desktop/Disease Prediction/dataset/blood_pressure.csv')
bp = bp.fillna(0)
bp_x = bp.drop(columns='Blood_Pressure_Abnormality', axis=1)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction',
                            'Parkinsons Prediction', 'Typhoid Prediction', 'Blood Pressure', 'Lung Cancer Prediction', 'Covid-19 Prediction',],  # 'Breast Cancer Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart',
                                  'person', 'thermometer-half', 'droplet-fill', 'lungs', 'virus',],  # 'bullseye'],
                           default_index=0)


# Define the prediction function


def predict_result(input_data, model):
    input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_as_numpy_array)
    prediction = model.predict(std_data)
    return prediction[0]

# Fit the scaler on the whole dataset once to avoid fitting for each prediction
# Assuming `full_data` is a dataframe containing all the combined data for all models
# scaler.fit(full_data)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    scaler = StandardScaler()
    scaler.fit_transform(dia_x)
    st.title('Diabetes Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0)
    with col2:
        hypertension = st.selectbox('Hypertension', [0, 1])
    with col3:
        heart_disease = st.selectbox('Heart Disease', [0, 1])
    with col1:
        bmi = st.number_input('BMI', min_value=0.0)
    with col2:
        hba1c_level = st.number_input('HbA1c Level', min_value=0.0)
    with col3:
        blood_glucose_level = st.number_input(
            'Blood Glucose Level', min_value=0.0)

    if st.button('Predict'):
        input_data = [age, hypertension, heart_disease,
                      bmi, hba1c_level, blood_glucose_level]
        result = predict_result(input_data, diabetes_model)
        st.write(f"Input data: {input_data}")

        if result == 0:
            st.success('The person is not Diabetic')
        else:
            st.success('The person is Diabetic')
            st.text("Treatment")
            st.text("Insulin Biguanide\nThiazolidinedione\nDipeptidyl peptides IV inhibitors\nMet forming\nMeglitinide\nDiabetes pills\nSGLT2 inhibitor\nSulfonylurea\nPhysical activity")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    scaler = StandardScaler()
    scaler.fit_transform(heart_x)
    st.title('Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0)
    with col2:
        sex = st.selectbox('Sex', [0, 1])
    with col3:
        cp = st.number_input('Chest Pain types', min_value=0)
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0)
    with col2:
        chol = st.number_input('Serum Cholesterol in mg/dl', min_value=0)
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    with col1:
        restecg = st.number_input(
            'Resting Electrocardiographic results', min_value=0)
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0)
    with col3:
        exang = st.selectbox('Exercise Induced Angina', [0, 1])
    with col1:
        oldpeak = st.number_input(
            'ST depression induced by exercise', min_value=0.0)
    with col2:
        slope = st.number_input(
            'Slope of the peak exercise ST segment', min_value=0)
    with col3:
        ca = st.number_input(
            'Major vessels colored by fluoroscopy', min_value=0)
    with col1:
        thal = st.selectbox('thal', [0, 1, 2])

    if st.button('Predict'):
        input_data = [age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = predict_result(input_data, heart_disease_model)
        st.write(f"Input data: {input_data}")

        if result == 0:
            st.success('The person is Healthy')
        else:
            st.success('The person has Heart Disease')
            st.text("Treatment")
            st.text(
                "Blood thinking medicine\n low dose aspirin\nClopidogrel\nRivaroxaban\nTicagrelor\n")
            st.text("Satins:\nAtorvastatin\nSimvastatin\nRosuvastatin")
            st.text("Beta blockers:\nAtenolol\nIsopropyl\nMeoprolol")
            st.text(
                "Anginotensin converting enzyme inhibitors:\nRamipril\n\nLisinopril\nAnginotensin2 receptor blockers\nDiuretics")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction")
    scaler = StandardScaler()
    scaler.fit_transform(parkinsons_x)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo (Hz)', min_value=0.0)
    with col2:
        fhi = st.number_input('MDVP:Fhi (Hz)', min_value=0.0)
    with col3:
        flo = st.number_input('MDVP:Flo (Hz)', min_value=0.0)
    with col4:
        jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0)
    with col5:
        jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0)
    with col1:
        rap = st.number_input('MDVP:RAP', min_value=0.0)
    with col2:
        ppq = st.number_input('MDVP:PPQ', min_value=0.0)
    with col3:
        ddp = st.number_input('Jitter:DDP', min_value=0.0)
    with col4:
        shimmer = st.number_input('MDVP:Shimmer', min_value=0.0)
    with col5:
        shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0)
    with col1:
        apq3 = st.number_input('Shimmer:APQ3', min_value=0.0)
    with col2:
        apq5 = st.number_input('Shimmer:APQ5', min_value=0.0)
    with col3:
        apq = st.number_input('MDVP:APQ', min_value=0.0)
    with col4:
        dda = st.number_input('Shimmer:DDA', min_value=0.0)
    with col5:
        nhr = st.number_input('NHR', min_value=0.0)
    with col1:
        hnr = st.number_input('HNR', min_value=0.0)
    with col2:
        rpde = st.number_input('RPDE', min_value=0.0)
    with col3:
        dfa = st.number_input('DFA', min_value=0.0)
    with col4:
        spread1 = st.number_input('spread1', min_value=0.0)
    with col5:
        spread2 = st.number_input('spread2', min_value=0.0)
    with col1:
        d2 = st.number_input('D2', min_value=0.0)
    with col2:
        ppe = st.number_input('PPE', min_value=0.0)

    if st.button('Predict'):
        input_data = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer,
                      shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        result = predict_result(input_data, parkinsons_model)
        st.write(f"Input data: {input_data}")

        if result == 0:
            st.success('The person does not have Parkinson\'s Disease')
        else:
            st.success('The person has Parkinson\'s Disease')
            st.text("Treatment")

# Typhoid Disease prediction
if selected == "Typhoid Prediction":
    st.title("Typhoid Disease Prediction")
    scaler = StandardScaler()
    scaler.fit_transform(typhoid_x)
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input('Age', min_value=0)
    with col2:
        Gender = st.selectbox('Gender', [0, 1])
    with col3:
        Hemoglobin_g_dL = st.number_input('Hemoglobin', min_value=0.0)
    with col1:
        Platelet_Count = st.number_input('Platelet Count', min_value=0.0)
    with col2:
        Calcium_mg_dL = st.number_input('Calcium', min_value=0.0)
    with col3:
        Potassium_mmol_L = st.number_input('Potassium', min_value=0.0)

    if st.button('Predict'):
        input_data = [Age, Gender, Hemoglobin_g_dL,
                      Platelet_Count, Calcium_mg_dL, Potassium_mmol_L]
        result = predict_result(input_data, typhoid_model)
        st.write(f"Input data: {input_data}")

        if result == 0:
            st.success('The person does not have Typhoid')
        else:
            st.success('The person has Typhoid')
            st.text("Treatment")
            st.text('1. Treat typhoid fever with antibiotics, which may include:\n'
                    ' Ciprofloxacin \n Levofloxacin or ofloxacin. \n Ceftriaxone \n Cefotaxime or cefixime. \n Azithromycin. \n'
                    ' Fluoroquinolones \n Cephalosporins \n Macrolides \n Carbapenems \n\n'
                    '2. Treated with steroids: dexamethasone \n\n'
                    '3. Other treatments:\n Drinking fluids \n Surgery')

# Covid-19 Prediction Page
if selected == 'Covid-19 Prediction':
    # Page title
    st.title("Covid-19 Prediction using ML")
    scaler = StandardScaler()
    scaler.fit(dfc_x)

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        dry_cough = st.number_input('Dry Cough', min_value=0.0)
    with col2:
        high_fever = st.number_input('High Fever', min_value=0.0)
    with col1:
        sore_throat = st.number_input('Sore Throat', min_value=0.0)
    with col2:
        difficulty_in_breathing = st.number_input(
            'Difficulty in Breathing', min_value=0.0)

    if st.button('Cov_Predict'):
        input_data = [dry_cough, high_fever,
                      sore_throat, difficulty_in_breathing]
        result = predict_result(input_data, covid19_model)
        st.write(f"Input data: {input_data}")

        if result == 0:
            st.success('The person is not affected with COVID-19')
        else:
            st.success('The person is affected with COVID-19')
            st.text("Treatment")
            st.text('1. Isolation and rest\n 2. Hydration and nutritious food\n 3. Medications as prescribed\n 4. Monitoring symptoms and seeking medical help if needed')

# Breast Cancer Prediction Page
# if selected == 'Breast Cancer Prediction':

#     # Page title
#     st.title("Breast Cancer Prediction using ML")

#     # getting the input data from the user
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         radius_mean = st.number_input('Radius Mean', min_value=0.0)
#     with col2:
#         texture_mean = st.number_input('Texture Mean', min_value=0.0)
#     with col3:
#         perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0)
#     with col1:
#         area_mean = st.number_input('Area Mean', min_value=0.0)
#     with col2:
#         smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0)
#     with col3:
#         compactness_mean = st.number_input('Compactness Mean', min_value=0.0)
#     with col1:
#         concavity_mean = st.number_input('Concavity Mean', min_value=0.0)
#     with col2:
#         concave_points_mean = st.number_input(
#             'Concave Points Mean', min_value=0.0)
#     with col3:
#         symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0)
#     with col1:
#         fractal_dimension_mean = st.number_input(
#             'Fractal Dimension Mean', min_value=0.0)
#     with col2:
#         radius_worst = st.number_input('Radius Worst', min_value=0.0)
#     with col3:
#         texture_worst = st.number_input('Texture Worst', min_value=0.0)
#     with col1:
#         perimeter_worst = st.number_input('Perimeter Worst', min_value=0.0)
#     with col2:
#         area_worst = st.number_input('Area Worst', min_value=0.0)
#     with col3:
#         smoothness_worst = st.number_input('Smoothness Worst', min_value=0.0)
#     with col1:
#         compactness_worst = st.number_input('Compactness Worst', min_value=0.0)
#     with col2:
#         concavity_worst = st.number_input('Concavity Worst', min_value=0.0)
#     with col3:
#         concave_points_worst = st.number_input(
#             'Concave Points Worst', min_value=0.0)
#     with col1:
#         symmetry_worst = st.number_input('Symmetry Worst', min_value=0.0)
#     with col2:
#         fractal_dimension_worst = st.number_input(
#             'Fractal Dimension Worst', min_value=0.0)

#     if st.button('BR_Predict'):
#         input_data = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_worst,
#                       texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]
#         result = predict_result(input_data, breast_cancer_model)
#         st.write(f"Input data: {input_data}")

#         if result == 0:
#             st.success("The person does not have Breast Cancer")
#         else:
#             st.success('The person has Breast Cancer')
#             st.text("Treatment")
#             st.text(
#                 "Systemic therapy FOR breast cancer\n Chemotherapy.\n Hormone therapy.\nImmunotherapy.\nTargeted therapy\nRadiation therapy\n a. External beam partial breast radiation \nb. Intraoperative radiation therapy[IORT]\n Surgery for breast cancer\n Lumpectomy (partial mastectomy).\n Mastectomy.\n Modified radical mastectomy \n Breast Reconstruction Surgery\n a. Implant-based reconstruction\n b. Nipple reconstruction\n c. Tissue-based reconstruction (flap reconstruction)\n Advanced techniques : \n Nipple-sparing mastectomy\n Skin-sparing mastectomy  ")


# Lung cancer prediction
if selected == 'Lung Cancer Prediction':
    scaler = StandardScaler()
    scaler.fit_transform(dfl_x)
    # Page title
    st.title("Lung Cancer Prediction")
    st.write("Fill in the details below to predict if a person has Lung Cancer.")

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('AGE', min_value=0)
    with col2:
        smoking = st.number_input('SMOKING', min_value=0)
    with col1:
        yellow_fingers = st.number_input('YELLOW_FINGERS', min_value=0)
    with col2:
        anxiety = st.number_input('ANXIETY', min_value=0)
    with col1:
        peer_pressure = st.number_input('PEER_PRESSURE', min_value=0)
    with col2:
        chronic_disease = st.number_input('CHRONIC_DISEASE', min_value=0)
    with col1:
        fatigue = st.number_input('FATIGUE', min_value=0)
    with col2:
        allergy = st.number_input('ALLERGY', min_value=0)
    with col1:
        wheezing = st.number_input('WHEEZING', min_value=0)
    with col2:
        alcohol_consuming = st.number_input('ALCOHOL_CONSUMING', min_value=0)
    with col1:
        coughing = st.number_input('COUGHING', min_value=0)
    with col2:
        shortness_of_breath = st.number_input(
            'SHORTNESS_OF_BREATH', min_value=0)
    with col1:
        swallowing_difficulty = st.number_input(
            'SWALLOWING_DIFFICULTY', min_value=0)
    with col2:
        chest_pain = st.number_input('CHEST_PAIN', min_value=0)

    if st.button('Lung_Predict'):
        input_data = [age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
                      allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]
        result = predict_result(input_data, lung_cancer_model)
        st.write(f"Input data: {input_data}")

        if result == 0:
            st.success("The person is healthy")
        else:
            st.success('The person has Lung Cancer')
            st.text("Treatment")
            st.text(
                "surgery \n chemotherapy \n radiation  therapy \n targeted therapy")


# BP prediction
if selected == 'Blood Pressure':
    scaler = StandardScaler()
    scaler.fit_transform(bp_x)
    st.title('Blood Pressure Prediction')
    st.write(
        'This model predicts the blood pressure of a person based on the input data.')

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        Level_of_Hemoglobin = st.number_input(
            'Level of Hemoglobin', min_value=0.0)
    with col2:
        Genetic_Pedigree_Coefficient = st.number_input(
            'Genetic Pedigree Coefficient', min_value=0.0)
    with col1:
        Age = st.number_input('Age', min_value=0)
    with col2:
        bmi = st.number_input('BMI', min_value=0.0)
    with col1:
        Sex = st.selectbox('Sex', [0, 1])  # Assuming 0 for female, 1 for male
    with col2:
        # Assuming 0 for no, 1 for yes
        Pregnancy = st.number_input('Pregnancy', min_value=0, max_value=1)
    with col1:
        # Assuming 0 for no, 1 for yes
        smoking = st.number_input('Smoking', min_value=0, max_value=1)
    with col2:
        Physical_activity = st.number_input('Physical Activity', min_value=0.0)
    with col1:
        salt_content_in_the_diet = st.number_input(
            'Salt Content in the Diet', min_value=0.0)
    with col2:
        alcohol_consumption = st.number_input(
            'Alcohol Consumption', min_value=0.0)
    with col1:
        Level_of_Stress = st.number_input('Level of Stress', min_value=0.0)
    with col2:
        Chronic_kidney_disease = st.number_input(
            'Chronic Kidney Disease', min_value=0, max_value=1)  # Assuming 0 for no, 1 for yes
    with col1:
        Adrenal_and_thyroid_disorders = st.number_input(
            'Adrenal and Thyroid Disorders', min_value=0, max_value=1)  # Assuming 0 for no, 1 for yes

    # Predict button
    if st.button('BP Predict'):
        input_data = [Level_of_Hemoglobin, Genetic_Pedigree_Coefficient, Age, bmi, Sex, Pregnancy, smoking, Physical_activity, salt_content_in_the_diet,
                      alcohol_consumption, Level_of_Stress, Chronic_kidney_disease, Adrenal_and_thyroid_disorders]

        # Assuming predict_result is a function that returns 0 for healthy, 1 for has BP
        result = predict_result(input_data, bp_model)

        st.write(f"Input data: {input_data}")

        if result == 0:
            st.success("The person is healthy.")
        else:
            st.success('The person has Blood Pressure.')
            st.text("Treatment")
            st.text("HYPERTENSION\n"
                    "Water pills(diuretics)"
                    "Angiotension converting enzyme inhibitors: Lisinopril,Benazepril,Captopril"
                    "Angiotension II receptor blockers:Candesartan,losartan"
                    "Calcium channel blockers: Amlodipine, diltiazem\n"
                    "HYPOTENSION\n"
                    "Use more salt"
                    "Drink more water"
                    "Medications:Fludrocortisone, Midodrine")

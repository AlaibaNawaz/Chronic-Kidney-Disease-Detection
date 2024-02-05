# Import necessary libraries
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Page title and navigation bar
st.title('Chronic Kidney Disease')
# Sidebar navigation
nav_choice = st.sidebar.radio('Navigation', ['Home', 'Data Visualizations', 'Machine Learning Model'])
# Load the data
data = pd.read_csv("chronic_kidney_disease.csv" , decimal = '.',
header = 0,
names = ['specific_gravity', 'albumin', 'sugar',
        'red_blood_cells', 'pus_cell',
        'pus_cell_clumps', 'bacteria',
        'blood_glucose_random', 'blood_urea',
        'serum_creatinine', 'sodium',
        'potassium', 'haemoglobin',
        'packed_cell_volume', 'white_cell_count',
        'red_cell_count', 'hypertension',
        'diabetes_mellitus',
        'coronary_artery_disease', 'appetite',
        'pedal_edema', 'anaemia',
        'classification'])
data.info()

#looking for missing values
data.isna().sum()
data.head()

data["red_cell_count"].unique()
#converting red_cell_count to numeric as we can see \t?
data["red_cell_count"] = pd.to_numeric(data["red_cell_count"] , errors ="coerce")

data["white_cell_count"].unique()
#converting white_cell_count to numeric as we can see \t? and \t8400
data["white_cell_count"] = pd.to_numeric(data["white_cell_count"] , errors= "coerce")

data["packed_cell_volume"].unique()
#converting packed_cell_volume to numeric as we can see \t? and \t43'
data["packed_cell_volume"] = pd.to_numeric(data["packed_cell_volume"] ,
errors = "coerce")

#checking the categorical variables now
cat_var = []
num_var = []
for var in data.columns:
    if data[var].dtype == 'object':
        cat_var.append(var)
    elif data[var].dtype == 'float64':
        num_var.append(var)
for var in cat_var:
    print(f"{var} unique values: {data[var].unique()}")

#we can see that diabetes_mellitus , coronary_artery_disease and classification have errors
#Fixing errors of 'diabetes_mellitus'
data.diabetes_mellitus = data.diabetes_mellitus.str.strip()
data['diabetes_mellitus'].replace({'/tno','/tyes'}, {'no','yes'}, inplace= True)

#Fixing errors of 'coronary_artery_disease'
data['coronary_artery_disease'].replace('\tno', 'no', inplace = True)
# Fixing errors of 'classification'
data['classification'].replace('ckd\t', 'ckd', inplace = True)

#imputing missing values
# Function to impute missing values with mean for numerical variables
def impute_mean(variable):
    data[variable].fillna(data[variable].mean(), inplace=True)
# Function to impute missing values with mode for categorical variables
def impute_mode(variable):
    mode_val = data[variable].mode().iloc[0]
    data[variable].fillna(mode_val, inplace=True)

# Impute missing values for numerical variables using mean imputation
for var in num_var:
    impute_mean(var)
# Impute missing values for categorical variables using mode imputation
for var in cat_var:
    impute_mode(var)
data.isna().sum()

# Visualizations
if nav_choice == 'Data Visualizations':
    st.header('Data Visualizations')

    # Random Blood Glucose Test Distribution by CKD Classification - Histogram
    st.subheader('Random Blood Glucose Test Distribution by CKD Classification')
    st.markdown("Examining the distribution of random blood glucose test results for each CKD classification.")

    hist_chart_glucose = alt.Chart(data).mark_bar().encode(
        x=alt.X('blood_glucose_random:Q', bin=alt.Bin(maxbins=30)),
        y='count()',
        color='classification:N',
        tooltip=['classification:N', 'blood_glucose_random:Q']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(hist_chart_glucose)

    # Serum Creatinine Distribution by CKD Classification - Histogram
    st.subheader('Serum Creatinine Distribution by CKD Classification')
    st.markdown("Histogram of serum creatinine levels for each CKD classification.")

    hist_chart_creatinine = alt.Chart(data).mark_bar().encode(
        x=alt.X('serum_creatinine:Q', bin=alt.Bin(maxbins=30)),
        y='count()',
        color='classification:N',
        tooltip=['classification:N', 'serum_creatinine:Q']
    ).properties(
        width=600,
        height=400
    )

    st.altair_chart(hist_chart_creatinine)

 # Scatter Plot of Blood Urea vs. Serum Creatinine
    st.subheader('Scatter Plot of Blood Urea vs. Serum Creatinine')
    st.markdown("Scatter plot to explore the relationship between blood urea and serum creatinine.")
    
    scatter_chart = alt.Chart(data).mark_circle().encode(
        x='blood_urea:Q',
        y='serum_creatinine:Q',
        color='classification:N',
        tooltip=['classification:N', 'blood_urea:Q', 'serum_creatinine:Q']
    ).properties(
        width=600,
        height=400
    )

    st.altair_chart(scatter_chart)



# Machine Learning Model
elif nav_choice == 'Machine Learning Model':
    st.header('Machine Learning Model')

     # Transform categorical variables to numerical values
    encoded_data = pd.get_dummies(data, drop_first=True)

    # Split data
    y = encoded_data['classification_notckd']
    x = encoded_data.drop('classification_notckd', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=222)

    # Logistic Regression
    st.subheader('Logistic Regression')
    st.markdown("Logistic Regression is a classification algorithm that predicts the probability of an instance belonging to a particular class.")
    model_lr = LogisticRegression(max_iter=200, random_state=222)
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    st.write(f"Accuracy of Logistic Regression: {accuracy_score(y_test, y_pred_lr):.2f}")

    # Display classification report in a DataFrame
    classification_report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
    df_classification_report_lr = pd.DataFrame(classification_report_lr).transpose()
    st.dataframe(df_classification_report_lr)

    # Decision Tree
    st.subheader('Decision Tree')
    st.markdown("Decision Tree is a tree-structured model where the dataset is split based on feature values. It is useful for both classification and regression tasks.")
    clf_dt = DecisionTreeClassifier()
    pipeline_dt = Pipeline([('scaler', StandardScaler()), ('clf', clf_dt)])
    fit_dt = pipeline_dt.fit(X_train, y_train)
    y_pred_dt = fit_dt.predict(X_test)
    st.write(f"Accuracy of Decision Tree: {accuracy_score(y_test, y_pred_dt):.2f}")

    # Display classification report in a DataFrame
    classification_report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
    df_classification_report_dt = pd.DataFrame(classification_report_dt).transpose()
    st.dataframe(df_classification_report_dt)

    # K-Nearest Neighbors (KNN)
    st.subheader('K-Nearest Neighbors (KNN)')
    st.markdown("K-Nearest Neighbors is a simple, effective, and versatile classification algorithm. It classifies a data point based on the majority class of its k-nearest neighbors.")
    k_values = list(range(1, 100, 2))
    acc_values = []
    for k in k_values:
        clf_knn = KNeighborsClassifier(k)
        pipeline_knn = Pipeline([('scaler', StandardScaler()), ('knn', clf_knn)])
        fit_knn = pipeline_knn.fit(X_train, y_train)
        y_pred_knn = fit_knn.predict(X_test)
        acc_sc_knn = accuracy_score(y_test, y_pred_knn)
        acc_values.append(acc_sc_knn)

    st.line_chart(pd.DataFrame({'k': k_values, 'accuracy': acc_values}).set_index('k'))

    # Display classification report in a DataFrame
    classification_report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
    df_classification_report_knn = pd.DataFrame(classification_report_knn).transpose()
    st.dataframe(df_classification_report_knn)
# Home
else:
    encoded_data = pd.get_dummies(data, drop_first=True)
    # Split data
    y = encoded_data['classification_notckd']
    x = encoded_data.drop('classification_notckd', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=222)

    # Logistic Regression
    model_lr = LogisticRegression(max_iter=1000, random_state=222, class_weight={0: 0.6, 1: 0.4}, C=1.0)
    model_lr.fit(X_train, y_train)
    # Explanation about Chronic Kidney Disease
    st.subheader('What is Chronic Kidney Disease (CKD)?')
    st.write(
        "Chronic Kidney Disease (CKD) is a condition where your kidneys gradually lose their ability "
        "to function properly. It is essential to monitor your kidney health through regular check-ups. "
        "This app provides insights and predictions related to CKD based on various health indicators."
    )
    
    # Input form for user to enter values for each feature
    st.subheader('Enter Health Indicators for Prediction')
    features = [ 'specific_gravity', 'albumin', 'sugar',
                'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
                'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                'potassium', 'haemoglobin', 'packed_cell_volume', 'white_cell_count',
                'red_cell_count', 'hypertension', 'diabetes_mellitus',
                'coronary_artery_disease', 'appetite', 'pedal_edema', 'anaemia']

    user_input = {}
    for feature in features:
        # Check if the feature is numeric or categorical
        if data[feature].dtype == 'float64':
            # For numeric features
            min_value = float(data[feature].min())
            user_input[feature] = st.number_input(f"Enter {feature}", min_value=min_value, max_value=data[feature].max())
        else:
            # For categorical features
            categories = data[feature].unique()
            selected_category = st.selectbox(f"Select {feature}", categories)
            user_input[feature] = selected_category

    # Predict button
    if st.button('Predict'):
        # Create a DataFrame from the user input
        input_data = pd.DataFrame([user_input])

        # Preprocess the input data (one-hot encoding)
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)

        # Ensure the input feature names match the ones used during training
        missing_features = set(X_train.columns) - set(input_data_encoded.columns)
        input_data_encoded = input_data_encoded.reindex(columns=X_train.columns, fill_value=0)

        # Make prediction using Logistic Regression model
        prediction = model_lr.predict(input_data_encoded)[0]

        # Display the prediction
        st.subheader('Prediction')
        if prediction == 1:
            st.write("Based on the provided health indicators, the model predicts that the person has Chronic Kidney Disease (CKD).")
        else:
            st.write("Based on the provided health indicators, the model predicts that the person does not have Chronic Kidney Disease (CKD).")


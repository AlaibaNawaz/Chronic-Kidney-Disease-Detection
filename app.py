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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

# Page title and navigation bar
st.title('Chronic Kidney Disease')
# Sidebar navigation
nav_choice = st.sidebar.radio('Navigation', ['Home', 'Data Visualizations', 'Machine Learning Model'])
# Load the data
data = pd.read_csv("chronic_kidney_disease.csv" , decimal = '.',
header = 0,
names = ['id', 'age' , 'blood_pressure','specific_gravity', 'albumin', 'sugar',
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
for var in data.columns:
    if data[var].dtype == 'object':
        cat_var.append(var)
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

#dealing with outliers
# Function to identify and handle outliers in all columns
def handle_outliers(df):
    # Create a copy of the dataframe to avoid modifying the original data
    df_outliers = df.copy()

    # Loop through each column
    for column in df_outliers.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df_outliers[column]):
            # Identify and handle outliers using IQR method
            Q1 = df_outliers[column].quantile(0.25)
            Q3 = df_outliers[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Clip values outside the lower and upper bounds
            df_outliers[column] = df_outliers[column].clip(lower=lower_bound, upper=upper_bound)

    return df_outliers

# Identify and handle outliers in all columns
data = handle_outliers(data)

#imputing missing values
# Function to impute missing values with mode for categorical variables
def impute_mode(variable):
    mode_val = data[variable].mode().iloc[0]
    data[variable].fillna(mode_val, inplace=True)
# Impute missing values for categorical variables using mode imputation
for var in cat_var:
    impute_mode(var)

#hist for age shows skewed data
plt.hist(data['age'],ec='black')
plt.show()

#hist for blood_pressure shows skewed data
plt.hist(data['blood_pressure'],ec='black')
plt.show()

#hist for specific_gravity shows ridged platue data
plt.hist(data['specific_gravity'],ec='black')
plt.show()

#hist for albumin shows ridged platue data
plt.hist(data['albumin'],ec='black')
plt.show()

#hist for albumin shows ridged platue data
plt.hist(data['sugar'],ec='black')
plt.show()

#hist for blood_glucose_random shows skewed data
plt.hist(data['blood_glucose_random'],ec='black')
plt.show()

#hist for blood_urea shows skewed data
plt.hist(data['blood_urea'],ec='black')
plt.show()

#hist for serum_creatinine shows skewed data
plt.hist(data['serum_creatinine'],ec='black')
plt.show()

#hist for blood_urea shows skewed data
plt.hist(data['blood_urea'],ec='black')
plt.show()

#hist for sodium shows skewed data
plt.hist(data['sodium'],ec='black')
plt.show()

#hist for potassium shows skewed data
plt.hist(data['potassium'],ec='black')
plt.show()

#hist for haemoglobin shows skewed data
plt.hist(data['haemoglobin'],ec='black')
plt.show()

#hist for packed_cell_volume shows skewed data
plt.hist(data['packed_cell_volume'],ec='black')
plt.show()

#hist for white_cell_count shows normal data
plt.hist(data['white_cell_count'],ec='black')
plt.show()

#hist for white_cell_count shows normal data
plt.hist(data['red_cell_count'],ec='black')
plt.show()

# Impute missing values in the "Age" column with the median bec skewed data
ageMedian = data["age"].median()
data['age'].fillna(ageMedian, inplace=True)

# Impute missing values in the "blood_pressure" column with the median bec skewed data
bpMedian = data["blood_pressure"].median()
data['blood_pressure'].fillna(bpMedian, inplace=True)

# Impute missing values in the "specific_gravity" column with the mode bec ridged platue histogram
sgMode = data['specific_gravity'].mode().iloc[0]
data['specific_gravity'].fillna(sgMode, inplace=True)

# Impute missing values in the "albumin" column with the mode bec ridged platue histogram
albuminMode = data['albumin'].mode().iloc[0]
data['albumin'].fillna(albuminMode, inplace=True)

# Impute missing values in the "sugar" column with the mode bec ridged platue histogram
sugarMode = data['sugar'].mode().iloc[0]
data['sugar'].fillna(sugarMode, inplace=True)

# Impute missing values in the "blood_glucose_random" column with the median bec skewed data
bgrMedian = data["blood_glucose_random"].median()
data['blood_glucose_random'].fillna(bgrMedian, inplace=True)

# Impute missing values in the "blood_urea" column with the median bec skewed data
buMedian = data["blood_urea"].median()
data['blood_urea'].fillna(buMedian, inplace=True)

# Impute missing values in the "serum_creatinine" column with the median bec skewed data
scMedian = data["serum_creatinine"].median()
data['serum_creatinine'].fillna(scMedian, inplace=True)

# Impute missing values in the "potassium" column with the median bec skewed data
pMedian = data["potassium"].median()
data['potassium'].fillna(pMedian, inplace=True)

# Impute missing values in the "haemoglobin" column with the median bec skewed data
hMedian = data["haemoglobin"].median()
data['haemoglobin'].fillna(hMedian, inplace=True)

# Impute missing values in the "sodium" column with the median bec skewed data
sodMedian = data["sodium"].median()
data['sodium'].fillna(sodMedian, inplace=True)

# Impute missing values in the "packed_cell_volume" column with the median bec skewed data
pcvMedian = data["packed_cell_volume"].median()
data['packed_cell_volume'].fillna(pcvMedian, inplace=True)

# Impute missing values in the "white_cell_count" column with the mean bec normal data
wccdMean = data["white_cell_count"].mean()
data['white_cell_count'].fillna(wccdMean, inplace=True)

# Impute missing values in the "red_cell_count" column with the mean bec normal data
rccdMean = data["red_cell_count"].mean()
data['red_cell_count'].fillna(rccdMean, inplace=True)

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
    # Line Chart of Average Blood Pressure by Age
    st.subheader('Line Chart of Average Blood Pressure by Age')
    st.markdown("Line chart showing the average blood pressure for each age.")

    line_chart_avg_blood_pressure = alt.Chart(data).mark_line().encode(
        x='age:O',  # Using ordinal scale for age
        y='average(blood_pressure):Q',  # Calculating average blood pressure
        color='classification:N',
        tooltip=['classification:N', 'age:O', 'average(blood_pressure):Q']
    ).properties(
        width=600,
        height=400
    )

    st.altair_chart(line_chart_avg_blood_pressure)




# Machine Learning Model
elif nav_choice == 'Machine Learning Model':
    st.header('Machine Learning Model')

    encoded_data = pd.get_dummies(data, drop_first=True)

    # Balance the classes
    class_counts = encoded_data['classification_notckd'].value_counts()
    class_1_indices = encoded_data[encoded_data['classification_notckd'] == 1].index
    num_samples_to_add = class_counts[0] - class_counts[1]
    random_indices = np.random.choice(class_1_indices, size=num_samples_to_add, replace=True)
    duplicated_samples = encoded_data.loc[random_indices]
    balanced_encoded_data = pd.concat([encoded_data, duplicated_samples], axis=0)
    balanced_encoded_data = balanced_encoded_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Verify the class distribution
    balanced_class_counts = balanced_encoded_data['classification_notckd'].value_counts()

    X = balanced_encoded_data.drop('classification_notckd', axis=1)
    y = balanced_encoded_data['classification_notckd']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

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

    # Support Vector Machine (SVM)
    st.subheader('Support Vector Machine (SVM)')
    st.markdown("Support Vector Machine is a powerful classification algorithm that finds the hyperplane that best separates the classes.")
    clf_svm = SVC(random_state=42)
    pipeline_svm = Pipeline([('scaler', StandardScaler()), ('svm', clf_svm)])
    fit_svm = pipeline_svm.fit(X_train, y_train)
    y_pred_svm = fit_svm.predict(X_test)
    st.write(f"Accuracy of Support Vector Machine: {accuracy_score(y_test, y_pred_svm):.2f}")

    # Display classification report in a DataFrame
    classification_report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
    df_classification_report_svm = pd.DataFrame(classification_report_svm).transpose()
    st.dataframe(df_classification_report_svm)

# Home
else:
    encoded_data = pd.get_dummies(data, drop_first=True)

    # Balance the classes
    class_counts = encoded_data['classification_notckd'].value_counts()
    class_1_indices = encoded_data[encoded_data['classification_notckd'] == 1].index
    num_samples_to_add = class_counts[0] - class_counts[1]
    random_indices = np.random.choice(class_1_indices, size=num_samples_to_add, replace=True)
    duplicated_samples = encoded_data.loc[random_indices]
    balanced_encoded_data = pd.concat([encoded_data, duplicated_samples], axis=0)
    balanced_encoded_data = balanced_encoded_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Verify the class distribution
    balanced_class_counts = balanced_encoded_data['classification_notckd'].value_counts()

    X = balanced_encoded_data.drop('classification_notckd', axis=1)
    y = balanced_encoded_data['classification_notckd']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train Support Vector Machine (SVM) model
    model_svm = SVC(random_state=42)
    model_svm.fit(X_train, y_train)

    # Explanation about Chronic Kidney Disease
    st.subheader('What is Chronic Kidney Disease (CKD)?')
    st.write(
        "Chronic Kidney Disease (CKD) is a condition where your kidneys gradually lose their ability "
        "to function properly. It is essential to monitor your kidney health through regular check-ups. "
        "This app provides insights and predictions related to CKD based on various health indicators."
    )

    # Input form for the user to enter values for each feature
    st.subheader('Enter Health Indicators for Prediction')
    features = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
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

        # Make prediction using Support Vector Machine (SVM) model
        prediction = model_svm.predict(input_data_encoded)[0]

        # Display the prediction
        st.subheader('Prediction')
        if prediction == 1:
            st.write("Based on the provided health indicators, the model predicts that the person has Chronic Kidney Disease (CKD).")
        else:
            st.write("Based on the provided health indicators, the model predicts that the person does not have Chronic Kidney Disease (CKD).")

# Import key libraries and packages
import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier



# Load the pipeline using pickle
#with open(r'C:\Users\GilB\OneDrive\Documents\Git Repo\ML-App\Embedding-a-Machine-Learning-Model-Into-A-GUI\pipeline.pkl', 'rb') as f:
#    pipeline = pickle.load(f)

       
# Load the toolkits
encoder = joblib.load(r'C:\Users\lenovo\Desktop\Azubi\Project_P4\Churn_gradio\Gradio_interface_of_TelcoChurn\exports\encoder.pkl')
imputer = joblib.load(r'C:\Users\lenovo\Desktop\Azubi\Project_P4\Churn_gradio\Gradio_interface_of_TelcoChurn\exports\imputer.pkl')
model = joblib.load(r'C:\Users\lenovo\Desktop\Azubi\Project_P4\Churn_gradio\Gradio_interface_of_TelcoChurn\exports\model.pkl')
scaler = joblib.load(r'C:\Users\lenovo\Desktop\Azubi\Project_P4\Churn_gradio\Gradio_interface_of_TelcoChurn\exports\scaler.pkl')

model=RandomForestClassifier()
model=joblib.load(r'C:\Users\lenovo\Desktop\Azubi\Project_P4\Churn_gradio\Gradio_interface_of_TelcoChurn\exports\model.pkl')

# Key lists
expected_inputs = ['SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','MonthlyCharges','TotalCharges','gender','InternetService','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
categoricals = ['gender','InternetService','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
numerics = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'MonthlyCharges', 'TotalCharges']

# Function to process inputs and return prediction
# 
def churn_pred(*args, 
               encoder=encoder, 
               imputer=imputer, 
               scaler=scaler, 
               model=model):
     
     # Convert inputs into a dataframe
     input_data=pd.DataFrame([args], columns=expected_inputs)
        #encode
     cat_encoded_cols = encoder.get_feature_names_out().tolist()
     encoded_category = pd.DataFrame(encoder.transform(input_data), columns=cat_encoded_cols)
        #scale
     scaled_numerics = pd.DataFrame(scaler.transform(input_data), columns=numerics)
         # combine scaled and encoded datasets
     input_list = pd.concat([encoded_category, scaled_numerics], axis=1)
         #Make the prediction
     model_output = model.predict(input_list)
     
     if model_output=='Yes':
          prediction=1
     else:
          prediction=0

     #Return the prediction
     return{'Prediction: Customer is likely to LEAVE': prediction,
            'Prediction: Customer is likely to STAY': 1-prediction}


 
# Setup app interface
## Inputs 
SeniorCitizen = gr.Radio(choices= ["No", "Yes"], value="No",label="SeniorCitizen"),
Partner = gr.Checkbox(['Yes','No'],label="Partner"),
Dependents = gr.Checkbox(['Yes','No'], label="Dependents"),
tenure = gr.Slider(minimum=0, maximum=1500, label="tenure"),
PhoneService = gr.Radio(choices= ["No", "Yes"], value="No", label="PhoneService"),
MultipleLines = gr.Radio(choices= ["No", "Yes"], value="No", label="MultipleLines"),
MonthlyCharges = gr.Slider(minimum=0, maximum=2000, label="MonthlyCharges"),
TotalCharges=gr.Slider(minimum=0, maximum=3000,label= "TotalCharges"),
gender = gr.Radio(choices= ["No", "Yes"], value="No", label="Gender"),
InternetService = gr.Radio(choices= ["No", "Yes"], value="No", label="InternetService"),
StreamingTV = gr.Radio(choices= ["No", "Yes"], value="No", label="StreamingTV"),
StreamingMovies = gr.Radio(choices= ["No", "Yes"], value="No", label="StreamingMovies"),
Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
PaperlessBilling = gr.Checkbox(label="PaperlessBilling"),
PaymentMethod = gr.Radio(choices= ["No", "Yes"], value="No", label="PaymentMethod")

## Outputs
gr.Interface(fn=churn_pred,
             inputs=[SeniorCitizen,Partner,Dependents,tenure,PhoneService,
                     MultipleLines,MonthlyCharges,TotalCharges,gender,
                     InternetService,StreamingTV,StreamingMovies,Contract,
                     PaperlessBilling,PaymentMethod],
             outputs=gr.Label('Awaiting submission ...'), 
             title= "Telco Churn Rate App", 
             description="This app was created for a Telco company",live=True).launch(inbrowser=True, show_error=True)
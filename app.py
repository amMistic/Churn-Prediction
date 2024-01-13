from flask import Flask,render_template,request,redirect,url_for
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/Input")
def Input():
    return render_template('Input.html')

@app.route("/Input", methods=['POST','GET'])
def predict():

    # load intiall data
    init_data = pd.read_csv('first_telc.csv')
    
    SeniorCitizen = request.form['query1']
    MonthlyCharges = request.form['query2']
    TotalCharges = request.form['query3']
    gender = request.form['query4']
    Partner = request.form['query5']
    Dependents = request.form['query6']
    tenure = request.form['query7']
    PhoneService = request.form['query8']
    MultipleLines = request.form['query9']
    InternetService = request.form['query10']
    OnlineSecurity = request.form['query11']
    OnlineBackup = request.form['query12']
    DeviceProtection = request.form['query13']
    TechSupport = request.form['query14']
    StreamingTV = request.form['query15']
    StreamingMovies = request.form['query16']
    Contract = request.form['query17']
    PaperlessBilling = request.form['query18']
    PaymentMethod = request.form['query19']

    # load model
    model  = pickle.load(open("model_RFC.sav",'rb'))

    # data
    data = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, tenure, PhoneService, MultipleLines,InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod ]]

    # convert into dataframe
    new_df = pd.DataFrame(data, columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                           'Partner', 'Dependents',
                                           'tenure',
                                           'PhoneService', 'MultipleLines', 'InternetService',
                                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                           'PaymentMethod' ])
    

    # concatenate this into the given
    update_data = pd.concat([init_data,new_df],ignore_index=True)

    #Group the labels
    labels = [ f"{i} - {i+11}" for i in range(1,80,12)]

    #tenure group 
    update_data['tenure'] = pd.cut(update_data.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

    
    # get the dummies
    update_data_dummies = pd.get_dummies(update_data[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
           'Contract', 'PaperlessBilling', 'PaymentMethod','tenure']])
    
    # Get the prediction, Whether the customer will churn or not
    pred = model.predict(update_data_dummies.tail(1))

    # Get the probability; whether the customer will churn or not
    churn_prob = model.predict_prob(update_data_dummies.tail(1))[:,1]

    # Displaying the output 
    if pred == 1:
        output_1= "The Customer is likely to Churn."
        output2 = f"Confidence level: {churn_prob*100}"
    else:
        output_1= "The Customer is not likely to Churn."
        output_2 = f"Confidence level: {(1- churn_prob)*100}"

    render_template('home.html', output1=output_1, output2=output_2, 
    query1 = request.form['query1'], 
    query2 = request.form['query2'],
    query3 = request.form['query3'],
    query4 = request.form['query4'],
    query5 = request.form['query5'], 
    query6 = request.form['query6'], 
    query7 = request.form['query7'], 
    query8 = request.form['query8'], 
    query9 = request.form['query9'], 
    query10 = request.form['query10'], 
    query11 = request.form['query11'], 
    query12 = request.form['query12'], 
    query13 = request.form['query13'], 
    query14 = request.form['query14'], 
    query15 = request.form['query15'], 
    query16 = request.form['query16'], 
    query17 = request.form['query17'],
    query18 = request.form['query18'], 
    query19 = request.form['query19'])


if __name__ == "__main__":
    app.run(debug=True)


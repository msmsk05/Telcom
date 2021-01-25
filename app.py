import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
resim=Image.open("customer.jpg")
st.image(resim, width=600)

html_temp = """
<div style="background-color:white;padding:10px">
<h1 style="color:black;text-align:center;"><b> Customer Churn Prediction</b></h1>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)



html_temp = """
<div style="background-color:green;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML App </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
my_slot1 = st.empty()
my_slot1.text('')
my_slot2 = st.empty()
my_slot2.text('')

model = pickle.load(open("my_model_2", "rb"))
features=pickle.load(open("features", "rb"))
deploy_df=pickle.load(open("deploy_df", "rb"))
df_final=pickle.load(open("df_final", "rb"))

tenure=st.sidebar.slider("Number of months the customer has stayed with the company (tenure)", 1, 72, step=1)
MonthlyCharges=st.sidebar.slider("The amount charged to the customer monthly", 0,100, step=5)
TotalCharges=st.sidebar.slider("The total amount charged to the customer", 0,5000, step=10)
OnlineSecurity=st.sidebar.selectbox("Whether the customer has online security or not", ('No', 'Yes', 'No internet service'))
Contract=st.sidebar.selectbox("The contract term of the customer", ('Month-to-month', 'One year', 'Two year'))
InternetService=st.sidebar.selectbox("Customer’s internet service provider", ('DSL', 'Fiber optic', 'No'))
TechSupport=st.sidebar.selectbox("Whether the customer has tech support or not", ('No', 'Yes', 'No internet service'))
    


@st.cache
def single_customer(tenure, MonthlyCharges, TotalCharges, OnlineSecurity, Contract, InternetService, TechSupport):
    my_dict={"tenure":tenure, "OnlineSecurity":OnlineSecurity, "Contract":Contract, "TotalCharges":TotalCharges, "InternetService":InternetService, TechSupport:"TechSupport", "MonthlyCharges":MonthlyCharges}
    
    columns = ['tenure', 'MonthlyCharges', 'TotalCharges',
       'OnlineSecurity_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No', 'TechSupport_No',
       'TechSupport_No internet service', 'TechSupport_Yes']
    
    df = pd.DataFrame.from_dict([my_dict], orient="columns")
    X = pd.get_dummies(df).reindex(columns=features, fill_value=0)
    result= "Churn probability of the customer is {} %".format(round(model.predict_proba(X)[:,1][0]*100,2))
    return result

if st.sidebar.button("predict"):
    st.sidebar.success(single_customer(tenure, MonthlyCharges, TotalCharges, OnlineSecurity, Contract, InternetService, TechSupport))

    
def random_customer():
    df_sample=df_final.sample(number1)
    df_5=df_sample.copy()
    df_2=pd.get_dummies(df_sample).reindex(columns=features, fill_value=0)
    prediction=model.predict_proba(df_2)
    res=model.predict(df_2)
    df_sample["Churn Probability"]= prediction[:,1]
    df_sample["Churn Probability"]=df_sample["Churn Probability"].apply(lambda x: round(x,2))
    df_sample["result"]=prediction[:,1]
    df_sample["result"]=df_sample["result"].apply(lambda x: round(x))
    df_sample["result"]=df_sample["result"].apply(lambda x: "Churn" if x==1 else "Retain")
    a=df_sample.result.value_counts().values[0]
    b=df_sample.result.value_counts().values[1]
    fig, ax= plt.subplots()
    ax.bar(df_sample.result.value_counts().index, df_sample.result.value_counts())
    ax.set_title("Distribution of the Randomly Selected {} Customers".format(number1))
    return st.success("Number of the customers that will retain: {}".format(a)), st.warning("Number of the customers that will churn:{}".format(b)), st.pyplot(fig), st.table(df_sample)
    
    
randomly_selected=st.checkbox("Churn Probability of Randomly Selected Customers")


if randomly_selected:
    st.subheader("How many customers to select?")
    st.text("Please select the number of customers")
    number1=st.selectbox(" ", (1,10,50,100,1000))
    if st.button("Analyze"):
        st.info("The analysis of randomly selected {} customers is shown below:".format(number1))
        random_customer()
        

def top_customers():
    df_sample=df_final.copy()
    df_2=pd.get_dummies(df_sample).reindex(columns=features, fill_value=0)
    prediction=model.predict_proba(df_2)
    df_sample["Churn Probability"]= prediction[:,1]
    df_sample["Churn Probability"]=df_sample["Churn Probability"].apply(lambda x: round(x,2))
    df_sample=df_sample.sort_values(by="Churn Probability", ascending=False).head(number2)
    return st.table(df_sample)
    
    
churn=st.checkbox("Top Customers to Churn")    
if churn:
    st.subheader("How many customers to select?")
    st.text("Please select the number of customers")
    number2=st.selectbox(" ", (1,10,50,100,1000))
    if st.button("Show"):
        st.warning("Top {} customers to churn".format(number2))
        top_customers()
    

def loyal_customers():
    df_sample=df_final.copy()
    df_2=pd.get_dummies(df_sample).reindex(columns=features, fill_value=0)
    prediction=model.predict_proba(df_2)
    df_sample["Churn Probability"]= prediction[:,1]
    df_sample["Churn Probability"]=df_sample["Churn Probability"].apply(lambda x: round(x,2))
    df_sample=df_sample.sort_values(by="Churn Probability").head(number)
    return st.table(df_sample)

loyal=st.checkbox("Top N Loyal Customers")   
if loyal:
    st.subheader("How many customers to select?")
    st.text("Please select the number of customers")
    number=st.selectbox(" ", (1,10,50,100,1000))
    if st.button("Display"):
        st.success("Top {} loyal customers".format(number))
        loyal_customers()


  

    
    
    
    

    

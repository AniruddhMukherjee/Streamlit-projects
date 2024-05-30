import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Salary Predictor")
data = pd.read_csv("/workspaces/Streamlit-projects/Data/Salary_Data.csv")

x = np.array(data["YearsExperience"]).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))

nav = st.sidebar.radio("Navigation", ["Home","Prediction","Contribute"])

# HOME PAGE SECTION STARTS ----------------------------

if nav == "Home":
  st.image("/workspaces/Streamlit-projects/Data/salary.jpeg", width=400)

  # TO SHOW THE DATASET TABLE
  # Create a state variable to track the visibility of the content
  show_content = st.session_state.get('show_content', False)

  # Create a button to toggle the visibility of the content
  if st.button('Show table'):
    show_content = not show_content
    st.session_state['show_content'] = show_content

  # Display the content based on the state variable
  if show_content:
    st.table(data)

  #if gph.button("Show Dataset in table format"):
  graph = st.selectbox("What kind of graph ?", ["None","Non-Interactive", "Interactive"])
  
  val = st.slider("Filter Data using Years", 0, 20)
  data = data.loc[data["YearsExperience"]>= val]

  if graph == "Non-Interactive":
    plt.figure(figsize = (10,5))
    plt.scatter(data["YearsExperience"], data["Salary"])
    plt.ylim(0)
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()
    # TO REMOVE THE WARNING IN THE WEBPAGE
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

  if graph == "Interactive":
    layout = go.Layout(
      xaxis = dict(range = [0,16]),
      yaxis = dict(range = [0,210000])
    )
    fig = go.Figure(data = go.Scatter(x = data["YearsExperience"], y = data["Salary"],mode = 'markers'), layout = layout)
    st.plotly_chart(fig) 



# PREDICTION SECTION STARTS --------------------------------
if nav == "Prediction":
  st.header("Know your Salary")
  val = st.number_input("Enter your exp", 0.00, 20.00, step = 0.25)
  val = np.array(val).reshape(1,-1)
  pred = lr.predict(val)[0]

  if st.button("Predict"):
    st.success(f"Your Predicted Salary is {round(pred)}")

# CONTRIBUTE PAGE --------------------------------------------
if nav == "Contribute":
  st.header("Contribute to our Dataset")
  ex = st.number_input("Enter your Experience",0.00,20.00, step=0.25)
  sal = st.number_input("Enter you Salary",0.00, 1000000.00, step = 1000.00)
  if st.button("Submit"):
    to_add = {"YearsExperience":[ex], "Salary":[sal]}
    to_add = pd.DataFrame(to_add)
    to_add.to_csv("/workspaces/Streamlit-projects/Data/Salary_Data.csv", mode = 'a', header = False, index = False)
    st.success("submitted")

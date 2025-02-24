import pandas as pd
import streamlit as st 
import plotly as plt
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
import plotly.express as px
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image
from tensorflow.keras.models import load_model
from pickle import load

df_bank = pd.read_csv('df_bank.csv')
#print(st.session_state['userID'])
image_url = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1647462652019-72bdb84235ea?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8fHx8fHx8MTcxMzI3OTI4Mg&ixlib=rb-4.0.3&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1080');
    background-size: cover;
    background-repeat: no-repeat;
    }
    </style>
    '''
st.markdown(image_url, unsafe_allow_html=True)

#Creating side panel with navigation options for admin
with st.sidebar:
    selected = option_menu( menu_title="Employee navigation options",
    options=['Fiscal Stats Gallery', "Fiscal Collapse Check"],
    icons=['bar-chart','gear'], 
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "20px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    },
    menu_icon="cast",
    default_index=0)
st.markdown("<h1 style='text-align: center; color: black;'>FINANCE RADAR</h1>", unsafe_allow_html=True)
if selected == 'Fiscal Stats Gallery':
    col1,space,col2 = st.columns([5,1,5])
    df_bank['X6'].fillna(0, inplace=True)
    df_bank['X18'].fillna(0, inplace=True)

    filtered_df = df_bank[df_bank['company_name'] == 'C_1']

    # Create scatter plot using Plotly Express
    fig = px.scatter(filtered_df, x='year', y='X6', color='status_label', title="Variation in the Net Income over the years",
                    labels={"year": "Year", "X6": "Net Income"})  # Adjust labels as needed

    # Adding tooltips
    fig.update_traces(mode='markers')  # Set mode to markers for scatter plot
    fig.update_layout(hovermode='x unified')  # Update layout for unified x-axis hover

    # Displaying the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)



    total_x18 = df_bank['X18'].sum()
    total_x14 = df_bank['X14'].sum()

    # Data for the pie chart
    labels = ['X18', 'X14']
    values = [total_x18, total_x14]
    colors = ['#FF9999', '#66B3FF']  # Custom colors for the pie chart

    # Creating the pie chart
    fig = px.pie(
    values=values,
    names=labels,
    color_discrete_sequence=colors,
    title="Distribution of Total Operating Expenses and Current Liabilities",
    width=700,  # Adjust the width as needed
    height=500,  # Adjust the height as needed
    )

    # Adding customizations
    fig.update_traces(textposition='inside', textinfo='percent+label')
    

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

        # Filter data for the selected company
    company_data = df_bank[df_bank['company_name'] == 'C_1']
    company_name = df_bank['company_name']
    # Count the status values
    status_counts = company_data['status_label'].value_counts()
    values = status_counts.values
    labels = status_counts.index
    colors = ['#BFEFFF', '#1E90FE']  # Colors for the pie chart


    # Create the pie chart
    fig = go.Figure(data=go.Pie(values=values, labels=labels, pull=[0.1, 0], hole=0.3, marker_colors=colors))
    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)
    fig.add_annotation(x=0.5, y=0.5, text='Company Status',
                    font=dict(size=18, family='Verdana', color='black'), showarrow=False)
    fig.update_layout(title_text=f'Proportion of Bankruptcy Status for the company')

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    col1,space,col2 = st.columns([5,1,5])



    fig = px.bar(df_bank, x='X1', y='X10',
            barmode='group',
            title="Comparison of Current Assets and Total Assets by Company")

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


        # Filter data by company
    filtered_df = df_bank[df_bank['company_name'] == 'C_1']

    # Plot using Seaborn with better label handling
    plt.figure(figsize=(10, 4))
    ax = sns.lineplot(data=filtered_df, x='year', y='X6', hue='status_label')
    plt.title("The trend for Company C_1")

    # Improve the x-axis label readability
    plt.xticks(rotation=45)  # Rotate labels to prevent overlap
    plt.tight_layout()  # Adjust subplot parameters to give the plot elements more room

    # Show plot in Streamlit using st.pyplot()
    st.pyplot(plt)

if selected == 'Fiscal Collapse Check':
    st.markdown('<p style="text-align:justify;font-weight:bold;">This section allows you to upload a new financial record data. Make sure to upload a data file of the specified file format which follows the dataset allowed factors list</p>',unsafe_allow_html = True)
    uploaded_file = st.file_uploader("Please upload the test data in csv file format", type=['xlsx',"csv"])
    if uploaded_file is not None:
        # Process the uploaded file (you can customize this part based on your needs)
        try:
            if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                df = pd.read_excel(uploaded_file)
            else:

                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully to the database!")
                if st.checkbox("View file"):
                    st.write("Uploaded Data:")
                    st.write(df)
                if st.button("Predict"):
                    test = pd.read_csv('test_f.csv')
                    test = test.drop("X4", axis=1)
                    test = test.drop("X13", axis=1)
                    test = test.drop("X16", axis=1)
                    test["X19"] = test["X18"] - test["X2"]
                    test = test.drop("company_name", axis=1)
                    test = test.drop("year", axis=1)

                    test=test.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,15]]

                    
                    scaler = load(open('scaler.pkl', 'rb'))
                    test.iloc[:,:-1]=scaler.transform(test.iloc[:,:-1])
                    X_new=[]
                    X_new.append(test.values)
                    X_new=np.array(X_new)
                    model_new = load_model('rnn_20240416-101623.keras')
                    output = model_new.predict(X_new, batch_size=64,verbose=1)
                    binary_predictions = (output >= 0.5).astype(int)
                    binary_predictions_list = binary_predictions.flatten().tolist()
                    x_labels = ["Year" f"{i} " for i in range(1, 11)]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_labels, y=binary_predictions_list, mode='lines+markers'))
                    fig.update_layout(title='Binary Predictions Line Chart', xaxis_title='Index', yaxis_title='Company Bankruptcy Status')
                    fig.update_yaxes(tickvals=[0, 1])
                    st.plotly_chart(fig)
                    if 0 in binary_predictions_list:
                        st.error("Bankrupty is likely for your company")
                    else:
                        st.success("Your company is financially sorted for the upcoming years!!")
                    
        except Exception as e:
            st.error(f"Error reading the file: {e}")

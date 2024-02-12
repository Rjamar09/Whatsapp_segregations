import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import json
import io
import requests
import time
import base64
# Load your image
image = Image.open('logo.jpg')

# Center-align the title and change the color using HTML-style formatting
title_html = """
    <style>
        .title {
            color: skyblue; /* Change this to the color you want */
            text-align: center;
        }
    </style>
    <h1 class="title">Whatsapp Data Analysis Dashboard</h1>
"""

st.markdown(title_html, unsafe_allow_html=True)
# Display the image
st.image(image, use_column_width=True)

# Custom function to remove decimal points
def remove_decimal(value):
    if isinstance(value, float):
        return str(int(value))
    return str(value)

# Sidebar
uploaded_file = st.sidebar.file_uploader("Upload your Whatsapp data (CSV or Excel format)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Display basic information in the sidebar
    st.sidebar.subheader("File Information")
    st.sidebar.write(f"Uploaded File Name: {uploaded_file.name}")
    st.sidebar.write(f"Number of Rows: {df.shape[0]}")
    st.sidebar.write(f"Number of Columns: {df.shape[1]}")
   
    # Apply the custom function to remove decimal points
    df['msisdn'] = df['msisdn'].apply(remove_decimal)

    # Display the uploaded DataFrame
    st.subheader("Uploaded Data")
    st.write(df)

    
    # Convert the 'sent_timestamp' column to datetime data type
    df['sent_timestamp'] = pd.to_datetime(df['sent_timestamp'], format="%d-%m-%Y %H:%M")
    # Extract the date and time components into separate columns
    df['sent_date'] = df['sent_timestamp'].dt.date
    df['sent_time'] = df['sent_timestamp'].dt.time

    # Convert the 'delivery_timestamp' column to datetime data type
    df['delivery_timestamp'] = pd.to_datetime(df['delivery_timestamp'], format="%d-%m-%Y %H:%M")
    # Extract the date and time components into separate columns
    df['delivery_date'] = df['delivery_timestamp'].dt.date
    df['delivery_time'] = df['delivery_timestamp'].dt.time

    # Convert the 'read_timestamp' column to datetime data type
    df['read_timestamp'] = pd.to_datetime(df['read_timestamp'], format="%d-%m-%Y %H:%M")
    # Extract the date and time components into separate columns
    df['read_date'] = df['read_timestamp'].dt.date
    df['read_time'] = df['read_timestamp'].dt.time

    # Convert the 'failed_timestamp' column to datetime data type
    df['failed_timestamp'] = pd.to_datetime(df['failed_timestamp'], format="%d-%m-%Y %H:%M")
    # Extract the date and time components into separate columns
    df['failed_date'] = df['failed_timestamp'].dt.date
    df['failed_time'] = df['failed_timestamp'].dt.time

    df.drop(columns=['submit_timestamp','sent_timestamp','delivery_timestamp','failed_timestamp','expiration_timestamp','read_timestamp'],inplace=True)

    # Increase the time by 5 hours and 30 minutes
    df['sent_time'] = pd.to_datetime(df['sent_time'].astype(str)) + pd.Timedelta(hours=5, minutes=30)
    df['sent_time'] = df['sent_time'].dt.time

    # Increase the time by 5 hours and 30 minutes
    df['delivery_time'] = pd.to_datetime(df['delivery_time'].astype(str)) + pd.Timedelta(hours=5, minutes=30)
    df['delivery_time'] = df['delivery_time'].dt.time

    # Increase the time by 5 hours and 30 minutes
    df['read_time'] = pd.to_datetime(df['read_time'].astype(str)) + pd.Timedelta(hours=5, minutes=30)
    df['read_time'] = df['read_time'].dt.time


    # Increase the time by 5 hours and 30 minutes
    df['failed_time'] = pd.to_datetime(df['failed_time'].astype(str)) + pd.Timedelta(hours=5, minutes=30)
    df['failed_time'] = df['failed_time'].dt.time

    # divide the data based on mt and mo
    df1=df[df['mo/mt']=='mt']
    df2=df[df['mo/mt']=='mo']

    # Reset the index of df2
    df2.reset_index(drop=True, inplace=True)

    # Drop unnecessary Columns
    columns_to_drop = []

    for col in df1.columns:
        if df1[col].isnull().all():
            columns_to_drop.append(col)
    columns_to_drop.append('media_url')
    columns_to_drop.append('profile_name')
    # Drop the unnecessary Columns
    data = df1.drop(columns=columns_to_drop, axis=1)


    # Extract the JSON data from the message column
    json_data = data['message_payload'].apply(json.loads)
    

    # Normalize the JSON data into separate columns
    df3 = pd.json_normalize(json_data)

    # Concatenate the normalized columns with the original DataFrame
    df4 = pd.concat([data, df3], axis=1)

    columns_to_drop=['country_code','msisdn','user_name','country_name','campaign_name','message_payload','request_id','timezone','conversation_id','message_id','pricing_category','source','media.lang_code','media.template_name','media.type']
    df4 = df4.drop(columns=columns_to_drop)
    columns_to_drop=['message_type']
    df4 = df4.drop(columns=columns_to_drop)

    # Define a function to extract the values from the 'media.body' column
    def extract_values(row):
        if isinstance(row, list):
            values = []
            for item in row:
                if 'text' in item:
                    values.append(item['text'])
            return values
        else:
            return []

    # Apply the function to the 'media.body' column and create new columns
    df_extracted_values = df4['media.body'].apply(lambda x: pd.Series(extract_values(x)))
    # Define the new column names
    new_column_names = ['name','auction', 'material','company_name','location', 'date', 'number', 'link']
    # Rename the columns
    df_extracted_values.columns = new_column_names
    

    # Concatenate the DataFrames vertically
    df_concatenated = pd.concat([df4, df_extracted_values], axis=1)
    columns_to_drop=['media.body']
    df_concatenated=df_concatenated.drop(columns=columns_to_drop)
    columns_to_drop=['payload_type']
    df_concatenated=df_concatenated.drop(columns=columns_to_drop)

    df_concatenated=df_concatenated[['phone','mo/mt','name','auction','material','company_name','location','date','number','link','special_comments','sent_date','sent_time','delivery_date','delivery_time','read_date','read_time','failed_date','failed_time']]


    # Filter the DataFrame to show rows where 'special_comments' is not empty
    non_empty_special_comments = df_concatenated[df_concatenated['failed_time'].notnull()]
    # Drop duplicates from the 'msisdn' column
    non_empty_special_comments.drop_duplicates(subset='phone', keep='first', inplace=True)
    # Get unique numbers from the filtered DataFrame
    unique_numbers = non_empty_special_comments['phone'].unique()
    # Create a DataFrame with only the unique numbers
    unique_numbers_df = pd.DataFrame({'phone': unique_numbers})
    # Reset the index of the filtered DataFrame
    non_empty_special_comments.reset_index(drop=True, inplace=True)
    # Add a section to select a specific MSISDN number
    st.header('Select MSISDN Number:')
    selected_msisdn = st.selectbox('Select MSISDN:', ['View All'] + unique_numbers.tolist())

   # Display the table with all rows
    st.write("All Rows:")

    # Add a download button for the filtered DataFrame
    if st.button('Download Filtered Data as Excel'):
        # Create a download link for the Excel file
        excel_data = non_empty_special_comments.copy()  # Make a copy to prevent modifying the original DataFrame
        excel_data.to_excel('filtered_data.xlsx', index=False, engine='openpyxl')
        
        with open('filtered_data.xlsx', 'rb') as f:
            excel_file_data = f.read()
        
        # Create a data URL for the Excel file
        b64 = base64.b64encode(excel_file_data).decode('utf-8')
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="filtered_data.xlsx">Download Excel File</a>'
        
        # Display the download link
        st.markdown(href, unsafe_allow_html=True)

    # Filter and display details for the selected MSISDN
    if selected_msisdn == 'View All':
        st.write(non_empty_special_comments)
    else:
        selected_row = non_empty_special_comments[non_empty_special_comments['phone'] == selected_msisdn]
        st.table(selected_row)


    # df2 replied messages
    columns_to_drop = []
    for col in df2.columns:
        if df2[col].isnull().all():
            columns_to_drop.append(col)


    columns_to_drop.append('timezone')
    columns_to_drop.append('message_id')
    columns_to_drop.append('country_code')
    columns_to_drop.append('payload_type')
    columns_to_drop.append('user_name')
    columns_to_drop.append('country_name')

    data_1=df2.drop(columns_to_drop, axis=1)

    # Step 2: Extract the JSON data from the message column
    json_data = data_1['message_payload'].apply(json.loads)

    # Step 3: Normalize the JSON data into separate columns
    df5 = pd.json_normalize(json_data)

    # Step 4: Concatenate the normalized columns with the original DataFrame
    df6 = pd.concat([data_1, df5], axis=1)

    df_concatenated_1=df6[['msisdn','sent_date','sent_time','profile_name','message_type','text.body','media_url']]



    # Streamlit app header
    st.header('Select Data by "mo" or "mt"')
    st.subheader('mo = message received from bidder')
    st.subheader('mt = message sent to bidder')

    # Create a dropdown widget for selecting "mo" or "mt"
    selection = st.selectbox('Select mo or mt:', ['mo', 'mt'])

    if selection == 'mt':
        # Display df_concatenated when 'mt' is chosen
        st.write('Concatenated Data for "mt":')
        st.write(df_concatenated)
    else:
        # Display df_concatenated_1 when 'mo' is chosen
        st.write('Concatenated Data for "mo":')
        st.write(df_concatenated_1.reset_index(drop=True))

    # Add a button to download the selected data as an Excel file
    if st.button('Download Data as Excel'):
        if selection == 'mt':
            # Download df_concatenated as Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_concatenated.to_excel(writer, sheet_name='mt_data', index=False)
            output.seek(0)
            st.download_button(label="Download Excel (mt)", data=output, file_name="mt_data.xlsx", key='download_mt')
        else:
            # Download df_concatenated_1 as Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_concatenated_1.to_excel(writer, sheet_name='mo_data', index=False)
            output.seek(0)
            st.download_button(label="Download Excel (mo)", data=output, file_name="mo_data.xlsx", key='download_mo')

    # Calculate and display the total counts for the selected data
    if selection == 'mt':
        total_count = len(df_concatenated)
    else:
        total_count = len(df_concatenated_1)

    st.subheader(f'Total Count for Selected Data: {total_count}')


   # Add a section to select MSISDN data
    st.header('Select phone Data')
    selected_msisdn = st.selectbox('Select phone:', df_concatenated['phone'].unique(), key='msisdn_selectbox')

    # Filter the data based on the selected MSISDN
    filtered_data = df_concatenated[df_concatenated['phone'] == selected_msisdn]

    # Reset the index to start from 0
    filtered_data = filtered_data.reset_index(drop=True)

    # Convert the column containing numbers to a numeric type (e.g., float)
    filtered_data['phone'] = pd.to_numeric(filtered_data['phone'], errors='coerce')  # Replace 'your_column_name'

    # Sort the data based on the values in 'your_column_name' in ascending order
    filtered_data = filtered_data.sort_values(by='phone')  # Replace 'your_column_name'

    # Convert the 'phone' column values to strings before displaying
    filtered_data['phone'] = filtered_data['phone'].astype(str)

    # Display the filtered and sorted data
    st.write(f'Data for selected MSISDN {selected_msisdn} (Sorted from Small to Big):')
    st.write(filtered_data)

    # Check if phone numbers in failed messages and selected phone data are the same
    common_phone_numbers = non_empty_special_comments[non_empty_special_comments['phone'].isin(filtered_data['phone'])]

    # Include phone numbers with length less than 10 or greater than 12
    less_than_10_or_greater_than_12 = non_empty_special_comments[
        (non_empty_special_comments['phone'].str.len() < 10) | (non_empty_special_comments['phone'].str.len() > 12)
    ]

    # Concatenate common phone numbers and additional phone numbers
    combined_phone_numbers = pd.concat([common_phone_numbers, less_than_10_or_greater_than_12])

    # Display the common entries in a new table
    st.header('Common Entries and Additional Entries:')
    st.write(combined_phone_numbers)



st.write("CREATED BY: DATA ANALYST")

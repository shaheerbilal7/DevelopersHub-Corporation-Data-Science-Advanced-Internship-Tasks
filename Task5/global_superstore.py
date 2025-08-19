# %%
# Streamlit Dashboard for Global Superstore Dataset

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load & Clean Dataset
# Replace with dataset path
superstoredf = pd.read_csv("Global_Superstore2.csv", encoding='latin1')

# Drop duplicates
superstoredf.drop_duplicates(inplace=True)

# Handle missing values
superstoredf.dropna(inplace=True)

# Convert Order Date to datetime
superstoredf['Order Date'] = pd.to_datetime(superstoredf['Order Date'], errors='coerce')

# Step 3: Build Streamlit App
def run_dashboard():
    st.title("📊 Interactive Business Dashboard - Global Superstore")

    # Sidebar Filters
    st.sidebar.header("Filters")
    regions = st.sidebar.multiselect("Select Region", superstoredf['Region'].unique(), default=superstoredf['Region'].unique())
    categories = st.sidebar.multiselect("Select Category", superstoredf['Category'].unique(), default=superstoredf['Category'].unique())
    sub_categories = st.sidebar.multiselect("Select Sub-Category", superstoredf['Sub-Category'].unique(), default=superstoredf['Sub-Category'].unique())

    # Filter Data
    filtered_superstoredf = superstoredf[
        (superstoredf['Region'].isin(regions)) &
        (superstoredf['Category'].isin(categories)) &
        (superstoredf['Sub-Category'].isin(sub_categories))
    ]

    # KPIs
    total_sales = filtered_superstoredf['Sales'].sum()
    total_profit = filtered_superstoredf['Profit'].sum()

    st.subheader("Key Performance Indicators")
    col1, col2 = st.columns(2)
    col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
    col2.metric("📈 Total Profit", f"${total_profit:,.2f}")

    # Charts
    st.subheader("Sales & Profit by Category")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=filtered_superstoredf.groupby("Category")[["Sales", "Profit"]].sum().reset_index().melt(id_vars="Category"), 
                x="Category", y="value", hue="variable", ax=ax)
    st.pyplot(fig)

    # Top 5 Customers by Sales
    st.subheader("🏆 Top 5 Customers by Sales")
    top_customers = filtered_superstoredf.groupby("Customer Name")['Sales'].sum().nlargest(5).reset_index()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_customers, x="Sales", y="Customer Name", ax=ax2, palette="Blues_r")
    st.pyplot(fig2)

# Step 4: Run Streamlit
# Uncomment below line when running with: streamlit run app.py
run_dashboard()

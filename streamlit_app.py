import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import scipy.stats
# from scipy.stats import norm
# import altair as alt

st.set_page_config(
    page_title="Exploratory Data Visualization", page_icon="📊", initial_sidebar_state="expanded"
)

st.write("""
# 📊 Exploratory Data Visualization
Upload to find quick exploratory data visualization.
""")

uploaded_file = st.file_uploader("Upload CSV", type=".csv")
use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example file to demo the app"
)

if use_example_file:
    uploaded_file = "houseprices.csv"

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # data_frame = sns.load_dataset(uploaded_file)
    columns_list = df.columns
    columns_tuple = tuple(columns_list)    



# def load_data():
#     df = pd.read_csv("houseprices.csv")
#     return df

# df = load_data()

def main():
    page = st.sidebar.selectbox(
                          "Select a Page",
                          [
                            "Homepage",
                            "Scatter Plot",
                            "Histogram",
                            "Box Plot",
                            "Heat Map",
                          ],)
    if page == "Homepage":
        # st.header("Data Application")
        st.balloons()
        st.markdown("### Data preview")
        st.dataframe(df.head())
        # df.columns()
        # columns_list = df.columns()
        # columns_tuple = tuple(columns_list)
        # print(columns_list)
        target_variable = st.selectbox('Please select the target variable', columns_tuple)
        st.write('You selected:', target_variable)
    elif page == "Scatter Plot":
        scatter_plot()
    elif page == "Histogram":
        st.header("Histogram")
        histogram()
    elif page == "Box Plot":
        st.header("Box Plot")
        box_plot()
    elif page == "Heat Map":
        st.header("Heat Map")
        heat_map()
        st.header("Top 5 Heat Map")
        heat_map_5()

def scatter_plot():
    fig = plt.figure(figsize=(12, 5))
    data = pd.concat([df['SalePrice'], df['GrLivArea']],axis=1)
    data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
    plt.xlabel("GrLivArea")
    plt.ylabel("SalePrice")
    plt.title("Sale Price & GrLivArea Plot")
    st.pyplot(plt)

def histogram():
    fig = plt.figure(figsize=(12, 5))
    plt.hist(df["SalePrice"], color="y", bins=50)
    st.pyplot(fig)

def box_plot():
    f,ax = plt.subplots(figsize=(12, 10))
    data = pd.concat([df['SalePrice'], df['OverallQual']],axis=1)
    fig = sns.boxplot(x='OverallQual',y='SalePrice', data=data)
    fig.axis(ymin=0, ymax=800000)
    st.pyplot(f)

def heat_map():
    cormat = df.corr()
    f,ax = plt.subplots(figsize=(12,9))
    sns.heatmap(cormat, vmax=.8, square=True)
    st.pyplot(f)

def heat_map_5():
    k = 10
    cormat = df.corr()
    f,ax = plt.subplots(figsize=(12,9))
    cols = cormat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    # sns.heatmap(cormat, vmax=.8, square=True)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    st.pyplot(f)

if __name__ == "__main__":
    main()

#st.line_chart(df)
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
    columns_array = np.asarray(columns_tuple)


# def load_data():
#     df = pd.read_csv("houseprices.csv")
#     return df

# df = load_data()

def main():

    # page = st.sidebar.selectbox(
    #                       "Select a Page",
    #                       [
    #                         "Homepage",
    #                         # "Scatter Plot",
    #                         # "Histogram",
    #                         # "Box Plot",
    #                         # "Heat Map",
    #                       ],)
    # if page == "Homepage":  

    st.balloons()
    st.markdown("### Data preview")
    if uploaded_file:
        st.dataframe(df.head())
        st.selectbox('Please select the target variable', key="target_variable", options=columns_tuple, index=len(columns_tuple)-1)
        st.write(st.session_state)
        st.markdown("### Exploring the Target Variable")
        st.markdown("#### 1. Describe Target Variable")
        st.write('Describing the Targeted Variable')
        st.write(df[st.session_state.target_variable].describe())
        st.write('Important Note: Make sure minimum is not less than zero as it will destroy the model')
        st.markdown("#### 2. Histogram")
        distributionplot(st.session_state.target_variable)
        st.write('Skewness: ', df[st.session_state.target_variable].skew())
        st.write('Kurtosis: ', df[st.session_state.target_variable].kurt())
        st.markdown("#### 3. Heatmap")
        st.write('Select the top Independent Variable with close relationship the Target Variable')
        heat_map_10(df, st.session_state.target_variable)
        st.multiselect('Please select the strong relationship variables', columns_list, default= columns_list[0],key="independent_vars")
        st.write('Independent Variables:', st.session_state.independent_vars)
        st.write(st.session_state)
        st.markdown("#### 4. Scatter Plots (Independent vs Dependent) variables")
        for idvr in st.session_state.independent_vars:
            scatter_plot(idvr, st.session_state.target_variable)


    # elif page == "Scatter Plot":
    #     st.write(st.session_state)
    #     st.write(st.session_state.target_variable)
    #     st.session_state['target_variable'] = st.session_state.target_variable
    #     scatter_plot()
    # elif page == "Histogram":
    #     st.write(st.session_state)
    #     st.write(st.session_state.target_variable)
    #     st.session_state['target_variable'] = st.session_state.target_variable
    #     st.header("Histogram")
    #     histogram()
    # elif page == "Box Plot":
    #     st.header("Box Plot")
    #     st.session_state['target_variable'] = st.session_state.target_variable
    #     box_plot()
    # elif page == "Heat Map":
    #     st.header("Heat Map")
    #     st.session_state['target_variable'] = st.session_state.target_variable
    #     heat_map()
    #     st.header("Top 5 Heat Map")
    #     heat_map_5()

def distributionplot(target_variable):
    f,ax = plt.subplots(figsize=(12,5))
    sns.distplot(df[target_variable])
    st.pyplot(f)

def heat_map_10(df, target_variable):
    k = 10
    cormat = df.corr()
    f,ax = plt.subplots(figsize=(12,9))
    cols = cormat.nlargest(k, target_variable)[target_variable].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    st.pyplot(f)

def scatter_plot(indep_vars, dep_vars):
    fig = plt.figure(figsize=(12, 5))
    data = pd.concat([df[dep_vars], df[indep_vars]],axis=1)
    data.plot.scatter(x=indep_vars, y=dep_vars)
    plt.xlabel(indep_vars)
    plt.ylabel(dep_vars)
    plt.title("%s vs %s" %(indep_vars,dep_vars))
    st.pyplot(plt)

def heat_map():
    cormat = df.corr()
    f,ax = plt.subplots(figsize=(12,9))
    sns.heatmap(cormat, vmax=.8, square=True)
    st.pyplot(f)

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


if __name__ == "__main__":
    main()

#st.line_chart(df)
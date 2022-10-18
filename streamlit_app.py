import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

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
    columns_list = df.columns
    columns_tuple = tuple(columns_list)
    columns_array = np.asarray(columns_tuple)

def main():  

    st.balloons()
    st.markdown("### Data preview")
    if uploaded_file:
        st.dataframe(df.head(50))
        st.write('Shape: ',df.shape)
        st.selectbox('Please select the target variable', key="target_variable", options=columns_tuple, index=len(columns_tuple)-1)
        # st.write(st.session_state)
        st.markdown("### Exploring the Target Variable")

        # Target Variable
        st.markdown("## 1. Describe Target Variable")
        st.write('Describing the Targeted Variable')
        st.write(df[st.session_state.target_variable].describe())
        st.write('Important Note: Make sure minimum is not less than zero as it will destroy the model')

        # Histogram
        st.markdown("## 2. Histogram Target Variable")
        distributionplot(st.session_state.target_variable)
        st.write('Skewness: ', df[st.session_state.target_variable].skew())
        st.write('Kurtosis: ', df[st.session_state.target_variable].kurt())

        # Heat Maps
        st.markdown("## 3. Heatmap")
        st.write('Select the top Independent Variable with close relationship the Target Variable')
        heat_map_10(df, st.session_state.target_variable)
        # st.write('Scatter Plot Indepedent Variables:', st.session_state.scatter_plot_vars)
        # st.write(st.session_state)

        # Scatter Plots
        st.markdown("## 4. Scatter Plots (Independent vs Dependent) variables")
        st.multiselect('Please select the strong relationship variables for scatter plot', columns_list, default=columns_list[1],key="scatter_plot_vars")
        for idvr in st.session_state.scatter_plot_vars:
            scatter_plot(idvr, st.session_state.target_variable)

        # Box Plots
        st.markdown("## 5. Select Box Plots (Independent vs Dependent) variables")
        st.write('Objective: How is the spread and the skewness' )
        st.multiselect('Please select the strong relationship variables for box plot', columns_list, default=columns_list[1],key="box_plot_vars")
        # st.write('Box Plot Indepedent Variables:', st.session_state.box_plot_vars)
        # st.write(st.session_state)
        for idvr in st.session_state.box_plot_vars:
            box_plot(idvr, st.session_state.target_variable)

        # User can plot histogram by selecting independent variables  
        st.markdown('## 6. Select Histogram Variables')
        st.write('Objective: Are the features a normal distribution or is it skewed?')
        st.write('Requirements: Select a numerical value')
        st.multiselect('Please select the independent variable', columns_list, default=columns_list[-1], key="data_transformation_vars")
        for idvr in st.session_state.data_transformation_vars:
            distributionplot(idvr)
            probplot(idvr) 


        # st.markdown("## ---Missing Values & Transformation---")
        
        # Missing Values 
        # st.markdown("## 6. Missing Indepedent Values")
        # missing_data = missing_values(df)
        # st.write('Do you want to remove the values with more than 1 missing values?')
        # if st.button('Yes! Please Log Transform', key="missing_data_btn"):
        #     remove_values(missing_data, df)
        

        # if st.button('Yes', key="log_transform_btn"):
        #     st.write('After Transformation')
        #     for idvr in st.session_state.data_transformation_vars:
        #         log_transformation(idvr)
        #         probplot(idvr)

    # elif page == "Scatter Plot":
    #     st.write(st.session_state)
    #     st.write(st.session_state.target_variable)
    #     st.session_state['target_variable'] = st.session_state.target_variable
    #     scatter_plot()


def probplot(indep_vars):
    f,ax = plt.subplots(figsize=(12,5))
    # if df[indep_vars]:
    result = stats.probplot(df[indep_vars], plot=plt)
    # sns.distplot(df[indep_vars])
    st.pyplot(f)

def log_transformation(indep_vars):
    # if df[indep_vars]
    df[indep_vars] = np.log(df[indep_vars])
    f,ax = plt.subplots(figsize=(12,5))
    sns.distplot(df[indep_vars])
    st.pyplot(f)

def box_plot(indep_vars, dep_vars):
    f, ax = plt.subplots(figsize=(8, 6))
    data =pd.concat([df[dep_vars], df[indep_vars]],axis=1)
    fig = sns.boxplot(x=indep_vars, y=dep_vars, data=data)
    st.pyplot(f)

def scatter_plot(indep_vars, dep_vars):
    fig = plt.figure(figsize=(12, 5))
    data = pd.concat([df[dep_vars], df[indep_vars]],axis=1)
    data.plot.scatter(x=indep_vars, y=dep_vars)
    plt.xlabel(indep_vars)
    plt.ylabel(dep_vars)
    plt.title("%s vs %s" %(indep_vars,dep_vars))
    st.pyplot(plt)


def remove_values(missing_data,df):
    st.write('Removing values')
    st.write('Removing number of columns: ', len((missing_data[missing_data['Total'] > 1])))
    st.write('Column Being Removed: ', (missing_data[missing_data['Total'] > 1]))
    df = df.drop((missing_data[missing_data['Total'] > 1]).index,1)
    df = df.drop(df.loc[df['Electrical'].isnull()].index)
    st.write('Column successfully removed!')
    st.write('Current available columns:')
    st.write(df.head(50))
    st.write('Current Shape: ',df.shape)

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

def heat_map():
    cormat = df.corr()
    f,ax = plt.subplots(figsize=(12,9))
    sns.heatmap(cormat, vmax=.8, square=True)
    st.pyplot(f)

def histogram():
    fig = plt.figure(figsize=(12, 5))
    plt.hist(df["SalePrice"], color="y", bins=50)
    st.pyplot(fig)

def missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() * 100/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
    st.write(missing_data.head(20))
    return missing_data

if __name__ == "__main__":
    main()

#st.line_chart(df)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import base64

# Function to handle missing values
def handle_missing_values(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

# Function to encode categorical data
def encode_categorical_data(df, categorical_cols):
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df

# Function to scale numerical data
def scale_numerical_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

# Main function for Streamlit app
def main():
    st.title("Streamlit App with Imama Kainat")
    
    choice = st.sidebar.selectbox("Menu", ["Home", "Data Preprocessing", "Machine Learning"])

    if choice == "Home":
        st.subheader("Welcome to the Streamlit App")

    elif choice == "Data Preprocessing":
        st.header("Data Preprocessing with Imama Kainat")

        uploaded_file = st.file_uploader("Upload a dataset for preprocessing", type=["csv", "txt"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                st.subheader("Handling Missing Values")
                df = handle_missing_values(df)
                st.write("Missing values handled. New data:")
                st.dataframe(df.head())

                st.subheader("Encoding Categorical Data")
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    df_encoded = encode_categorical_data(df, categorical_cols)
                    st.write("Encoded data:")
                    st.dataframe(df_encoded.head())

                st.subheader("Scaling Numerical Data")
                numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if numerical_cols:
                    df_scaled = scale_numerical_data(df[numerical_cols])
                    st.write("Scaled data:")
                    st.dataframe(df_scaled.head())

                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_data.csv">Download Preprocessed Data</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif choice == "Machine Learning":
        st.header("Machine Learning with Imama Kainat")

        uploaded_file = st.file_uploader("Upload a dataset for clustering", type=["csv", "txt"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                le = LabelEncoder()
                df_encoded = df.apply(le.fit_transform)

                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(df_encoded)
                df_pca = pd.DataFrame(data=principal_components, columns=["Principal Component 1", "Principal Component 2"])

                k = st.slider("Select number of clusters", min_value=2, max_value=10)
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(df_encoded)
                df["Cluster"] = kmeans.labels_

                st.subheader("Cluster Visualization")
                fig, ax = plt.subplots()
                scatter = ax.scatter(df_pca["Principal Component 1"], df_pca["Principal Component 2"], c=df["Cluster"], cmap="viridis")
                legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
                ax.add_artist(legend)
                st.pyplot(fig)

                st.subheader("Cluster Details")
                numeric_df = df.select_dtypes(include=np.number)
                st.write(numeric_df.groupby("Cluster").mean())

                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="clustering_results.csv">Download Clustering Results</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.markdown('<p class="stFooter">Developed by Imama Kainat with Streamlit</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

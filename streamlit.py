import streamlit as st

# https://streamlit.io/components
# https://docs.streamlit.io/library/api-reference

st.title("Mushroom Prediction")

st.sidebar.title("Table of Content")
pages = ["Exploratory Data", "Data Visualization"]
page = st.sidebar.radio("Go to", pages)


if page == pages[0]:
    st.write("Introduction")
    if st.checkbox("Display"):
        st.write("Streamlit continuation")

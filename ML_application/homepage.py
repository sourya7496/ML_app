import streamlit as st


x=1
st.set_page_config(
    page_title="Machine Learning for Beginners"
)

# Sidebar navigation
st.sidebar.title("Choose One!")
page = st.sidebar.radio("Select your task!", ["Home", "Classification", "Regression", "Testing"])
if page=="Home":

    st.write("# Welcome!")

    st.write("Please choose one of your task where you want to use Machine Learning and upload dataset. See how it goes! Let's try!")



if page == "Classification":
    x=0
    import ml_classify
    ml_classify.run_2()
elif page == "Regression":
    x=0
    import ME599
    ME599.run_1()
elif page == "Testing":
    x=0
    import test_app
    test_app.run_3()
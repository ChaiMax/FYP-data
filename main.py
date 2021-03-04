import streamlit as st
import fyp
import forecast

PAGES={
    "Best Classifier Used":fyp,
    "Eczema Prediction Based on Weather Analytics":forecast
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
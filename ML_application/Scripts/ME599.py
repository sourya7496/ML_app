import pandas as pd
import streamlit as st

# Title of the web application
st.title('Square Calculator')

# Input field for the user to enter a number
number = st.number_input('Enter a number')

# Calculate the square of the input number
square = number ** 2

# Display the square of the input number
st.write(f'The square of {number} is: {square}')

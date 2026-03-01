# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np

# Define a function to fetch weather data

def fetch_weather_data(api_key: str, location: str) -> pd.DataFrame:
    """Fetches weather data from the API and returns it as a DataFrame."""
    try:
        # Call the weather API
        response = requests.get(f'http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}')
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        st.error(f'An error occurred: {e}')
        return pd.DataFrame()  # Return an empty DataFrame on error

# Define the main function for the Streamlit app

def main() -> None:
    """Main function to run the Streamlit app."""
    st.title('Weather and Energy Insights')
    st.sidebar.header('Input Parameters')
    location = st.sidebar.text_input('Location', 'City, Country')
    api_key = st.sidebar.text_input('API Key', 'Your_API_Key')

    if st.sidebar.button('Get Weather Data'):
        weather_data = fetch_weather_data(api_key, location)
        if not weather_data.empty:
            st.write(weather_data)

# Run the app
if __name__ == '__main__':
    main()
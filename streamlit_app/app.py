import streamlit as st
from weather import Weather
from energy import Energy

# Weather and Energy Insights

class WeatherEnergyApp:
    def __init__(self):
        self.weather_service = Weather()
        self.energy_service = Energy()

    def run(self):
        st.title("Weather and Energy Insights")

        city = st.text_input("Enter City", "")
        if city:
            self.display_weather(city)
            self.display_energy(city)

    def display_weather(self, city: str) -> None:
        try:
            weather_data = self.weather_service.get_weather(city)
            st.subheader("Weather Information")
            st.write(f"Temperature: {weather_data['temp']} °C")
            st.write(f"Condition: {weather_data['condition']}")
        except Exception as e:
            st.error(f"Error fetching weather data: {str(e)}")

    def display_energy(self, city: str) -> None:
        try:
            energy_data = self.energy_service.get_energy_data(city)
            st.subheader("Energy Consumption")
            st.write(f"Energy Usage: {energy_data['usage']} kWh")
            st.write(f"Cost: ${energy_data['cost']}")
        except Exception as e:
            st.error(f"Error fetching energy data: {str(e)}")

if __name__ == '__main__':
    app = WeatherEnergyApp()
    app.run()
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import plotly.express as px
from dateutil import parser
import re

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar buttons for page navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("Home"):
        st.session_state.page = 'Home'
    if st.button("Drivers"):
        st.session_state.page = 'Drivers'
    if st.button("Seasons"):
        st.session_state.page = 'Seasons'
    if st.button("Races"):
        st.session_state.page = 'Races'

# Main content based on current page
if st.session_state.page == 'Home':
    st.title("🏁 Formula 1 Analytics")
    st.write("Welcome! Use the sidebar to explore F1 data by Driver, Season, or Race.")

elif st.session_state.page == 'Drivers':
    st.title("🏎️ Driver Statistics")

    # Function to fetch active drivers from current season standings
    def fetch_current_drivers(season="2025"):
        url = f"https://api.jolpi.ca/ergast/f1/{season}/driverStandings.json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            standings = data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
            return [
                {
                    "name": f"{d['Driver']['givenName']} {d['Driver']['familyName']}",
                    "driverId": d['Driver']['driverId'],
                    "nationality": d['Driver']['nationality'],
                    "points": d['points'],
                    "wins": d['wins'],
                    "constructors": [c['name'] for c in d['Constructors']],
                    "dateOfBirth": d['Driver']['dateOfBirth'],
                    "url": d['Driver']['url']
                }
                for d in standings
            ]
        else:
            st.error("Failed to fetch current drivers.")
            return []
    def scrape_driver_info(wiki_url):
        def extract_car_number(text):
            match = re.search(r'\b\d+\b', text)
            return match.group(0) if match else "N/A"

        try:
            response = requests.get(wiki_url)
            if response.status_code != 200:
                return {}

            soup = BeautifulSoup(response.text, 'html.parser')
            infobox = soup.find("table", {"class": "infobox"})

            info = {
                "image": None,
                "championships": "N/A",
                "wins": "N/A",
                "points": "N/A",
                "car_number": "N/A",
                "podiums": "N/A"
            }

            # Get image
            if infobox:
                img = infobox.find("img")
                if img:
                    info["image"] = "https:" + img['src']

                rows = infobox.find_all("tr")
                for row in rows:
                    header = row.find("th")
                    data = row.find("td")
                    if header and data:
                        label = header.get_text(strip=True).lower()
                        value = data.get_text(" ", strip=True)

                        if "championships" in label:
                            info["championships"] = value
                        elif "wins" in label and "career" not in label:
                            info["wins"] = value
                        elif "points" in label:
                            info["points"] = value
                        elif "car number" in label:
                            info["car_number"] = extract_car_number(value)
                        elif "podiums" in label:
                            info["podiums"] = value

            return info

        except Exception as e:
            st.warning(f"Error scraping additional info: {e}")
            return {}


    # Main app
    def main(): 
        drivers = fetch_current_drivers()
        if not drivers:
            return

        driver_names = [driver['name'] for driver in drivers]
        selected_driver_name = st.selectbox("Select a Driver", driver_names)
        selected_driver = next(d for d in drivers if d['name'] == selected_driver_name)

        driver_info = scrape_driver_info(selected_driver['url'])

        col1, col2 = st.columns([2.9, 2.8])

        with col1:
            st.subheader(f"{selected_driver['name']}")
            st.markdown(f"**Car Number:** {driver_info.get('car_number', 'N/A')}")
            st.markdown(f"**Nationality:** {selected_driver['nationality']}")
            st.markdown(f"**Date of Birth:** {selected_driver['dateOfBirth']}")
            st.markdown(f"**Team:** {', '.join(selected_driver['constructors'])}")
            st.markdown(f"**Points in this season:** {selected_driver['points']}")
            st.markdown(f"**Wins in this season:** {selected_driver['wins']}")

            st.markdown("---")
            st.markdown("### 🏁 Career Statistics")
            st.markdown(f"**World Championships:** {driver_info.get('championships', 'N/A')}")
            st.markdown(f"**Career Wins:** {driver_info.get('wins', 'N/A')}")
            st.markdown(f"**Career Points:** {driver_info.get('points', 'N/A')}")
            st.markdown(f"**Podiums:** {driver_info.get('podiums', 'N/A')}")

        with col2:
            if driver_info.get("image"):
                st.image(driver_info["image"], width=180, caption=selected_driver['name'])
            else:
                st.info("No image available.")
    if __name__ == "__main__":
        main()
import streamlit as st
import plotly.graph_objects as go
import requests 
from bs4 import BeautifulSoup
from datetime import datetime
import re
from collections import defaultdict

def f_drivers():
    st.title("🏎️ Driver Statistics")
    # Function to fetch active drivers from current season standings
    def fetch_current_drivers(season=None):
        if season is None:
            season = datetime.now().year
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
        
    #Function to scrape driver info from wikipedia
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
                "championship_years": [],
                "wins": "N/A",
                "points": "N/A",
                "car_number": "N/A",
                "podiums": "N/A",
                "first_entry": "N/A",
                "first_win": "N/A",
                "last_entry": "N/A"
            }

            # scrape image
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
                            years = re.findall(r'\d{4}', value)
                            if years:
                                info["championship_years"] = years
                        elif "wins" in label and "career" not in label:
                            info["wins"] = value
                        elif "points" in label:
                            info["points"] = value
                        elif "car number" in label:
                            info["car_number"] = extract_car_number(value)
                        elif "podiums" in label:
                            info["podiums"] = value
                        elif "first entry" in label:
                            info["first_entry"] = value
                        elif "first win" in label:
                            info["first_win"] = value
                        elif "last entry" in label:
                            info["last_entry"] = value

            return info

        except Exception as e:
            st.warning(f"Error scraping additional info: {e}")
            return {}
        
    #Function to generate career timeline plot
    def create_driver_timeline(info, driver_name):  
        def extract_year(text):
            match = re.search(r'\b(19|20)\d{2}\b', text)
            return int(match.group(0)) if match else None

        def extract_championship_count(champ_string):
            if not champ_string or champ_string == "N/A":
                return 0
            match = re.match(r"^\d+", champ_string.strip())
            return int(match.group(0)) if match else 0
            
        events = []
        if info.get("first_entry") != "N/A":
            events.append(("First Entry", info["first_entry"], extract_year(info["first_entry"])))
        if info.get("first_win") != "N/A":
            events.append(("First Win", info["first_win"], extract_year(info["first_win"])))
        for year in info.get("championship_years", []):
            events.append(("Championship", f"Won Championship ({year})", int(year)))
        if info.get("last_entry") != "N/A":
            events.append(("Last Entry", info["last_entry"], extract_year(info["last_entry"])))
            
        events = [e for e in events if e[2] is not None]
        if not events:
            st.info("No timeline data available.")
            return
            
        events.sort(key = lambda x: x[2])

        years = [e[2] for e in events]
        labels = [e[0] for e in events]
        races = [e[1] for e in events]
            
        min_year = min(years) 
        max_year = max(years) 

        dark_blue = "#003366"
        fig = go.Figure()
        fig.add_shape(
            type="line",
            x0=min_year,
            y0=0,
            x1=max_year,
            y1=0,
            line=dict(color=dark_blue, width=3)
        )
        year_counts = defaultdict(int)
        for y in years:
            year_counts[y] += 1

        used_offsets = defaultdict(int)

        for i, year in enumerate(years):
            count = year_counts[year]
            if count > 1:
                offset_index = used_offsets[year]
                x_jitter = 0.12 * (offset_index - (count - 1) / 2)
                x_value = year + x_jitter
                used_offsets[year] += 1
            else:
                x_value = year

            fig.add_shape(
                type="line",
                x0=x_value,
                y0=0,
                x1=x_value,
                y1=0.6,
                line=dict(color=dark_blue, width=2)
            )

            fig.add_trace(go.Scatter(
                x=[x_value],
                y=[0.6],
                mode="markers",
                marker=dict(color=dark_blue, size=20),
                showlegend=False,
                hovertemplate=f"<b>{labels[i]}</b><br>{races[i]}<br>Year: {year}<extra></extra>"
            ))

            bottom_y_base = -0.1
            bottom_y_offset = 0.1 *used_offsets[year] 
            bottom_y_value = bottom_y_base - bottom_y_offset

            fig.add_annotation(
                x=x_value,
                y=bottom_y_value,
                text=f"<b>{labels[i]}</b>",
                showarrow=False,
                yanchor="top",
                textangle=0,
                font=dict(size=10),
                align="center"
            )
            top_y_base = 0.8
            top_y_offset = 0.15  *used_offsets[year] 
            top_y_value = top_y_base - top_y_offset

            race_name_only = re.sub(r'^(\d{4}\s+)|(\s+\(\d{4}\))$', '', races[i])
            fig.add_annotation(
                x=x_value,
                y=top_y_value,
                text=race_name_only,
                textangle=30,
                showarrow=False,
                yanchor="bottom",
                font=dict(size=11, color="gray"),
                align="center",
                xanchor="center", 
                xshift= 5
            )
        fig.update_yaxes(visible=False, range=[-0.5, 1])
        padding = 0.3
        fig.update_xaxes(range=[min_year - padding, max_year + padding], dtick=1, showgrid=False, zeroline=False)

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=40),
            plot_bgcolor="white",
            hovermode="closest",
            xaxis_title="Year"
        )

        st.plotly_chart(fig, use_container_width=True)

        #dynamic summary text
        first_entry = info.get("first_entry", "N/A")
        first_win = info.get("first_win", "N/A")
        championships_raw = info.get("championships", "0")
        championships = extract_championship_count(championships_raw)
        championship_years = info.get("championship_years", [])

        commentary = f"**{driver_name}** first raced in Formula 1 at **{first_entry}**."
            
        if first_win != "N/A" and first_win != first_entry:
            commentary += f" Their first win came at **{first_win}**."
        elif first_win == first_entry:
            commentary += " They won their debut race."

        if championships > 0:
            years_str = ", ".join(str(y) for y in championship_years)
            commentary += f" They have won **{championships}** World Championship{'s' if championships > 1 else ''} in the year{'s' if championships > 1 else ''} {years_str}."
        else:
            commentary += " They have not won a World Championship yet."

        st.markdown(commentary)

    def build_wiki_url(driver_name):
        name_formatted = driver_name.strip().replace(" ", "_")
        return f"https://en.wikipedia.org/wiki/{name_formatted}"

    drivers = fetch_current_drivers()
    if not drivers:
        return

    driver_names = [driver['name'] for driver in drivers]
    selected_driver_name = st.selectbox("Select a Driver from current season", driver_names)
    driver_wiki_url = build_wiki_url(selected_driver_name)
    selected_driver = next(d for d in drivers if d['name'] == selected_driver_name)

    driver_info = scrape_driver_info(driver_wiki_url)

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
        st.markdown(" ")
        if driver_info.get("image"):
            st.image(driver_info["image"], width=300, caption=selected_driver['name'])
        else:
            st.info("No image available.")
    st.subheader("Career Timeline")
    st.caption("Timeline of key milestones in the driver's Formula 1 career")
    create_driver_timeline(driver_info,selected_driver_name)        

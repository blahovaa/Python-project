import streamlit as st
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import requests 
from bs4 import BeautifulSoup
from datetime import datetime
from datetime import timedelta
from dateutil import parser
import re
from fastf1 import utils
import seaborn as sns
from collections import defaultdict


# Design of the app
st.set_page_config(layout="wide")
         
# Defining variables, paths
TEAM_COLORS = {
    "Red Bull":  "#1E5BC6",
    "Ferrari":   "#DC0000",
    "Mercedes":  "#00D2BE",
    "McLaren":   "#FF8700",
    "Aston Martin": "#229971",
    "Alpine":    "#0090FF",
    "Williams":  "#005AFF",
    "RB":        "#6692FF",
    "Sauber":    "#52E252",
    "Haas":      "#B6BABD",
}

# Home page = main page
previous_page = st.session_state.get('page', 'Home')
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

if st.session_state.page != previous_page:
    st.rerun()

#----------- MAIN PAGE ------------
if st.session_state.page == 'Home':
    # Making 2 columns 
    col1, col2 = st.columns([2, 1]) 

    with col1:
        st.title("Formula 1 Analytics")
        st.markdown("""
        Welcome to **Formula 1 Analytics**, a data-driven dashboard built for exploring and visualizing key moments and trends from the world of Formula 1. This project pulls in real F1 race data using the FastF1 package and Ergast API, making it easy to dive into statistics across seasons, drivers, and races.

        Right here on the **Home** page, you’ll find a summary of the most recent Grand Prix podium, along with the current driver standings for the ongoing season. Furthermore, a brief overview of Formula 1, scraped from Wikipedia, is provided below.

        Use the sidebar to explore more:
        - Browse detailed **driver profiles** with extra info pulled from Wikipedia
        - Analyze how **teams have performed** across past seasons
        - Compare **individual races**, including lap times, race pace, and telemetry charts

        Whether you're a fan, a motorsport analyst, or just curious, this tool offers a simple way to understand the numbers behind the racing.
         """)

    # Podium of the most recent race
    with col2:
        API = "https://api.jolpi.ca/ergast/f1"
        try:
            race = requests.get(f"{API}/current/last/results.json", timeout=6).json()["MRData"]["RaceTable"]["Races"][0]
            race_name = race["raceName"]
            top3_unordered = race["Results"][:3]
            top3 = [top3_unordered[i] for i in (1, 0, 2)]                    

            podium_df = pd.DataFrame({
                "Place":  ["🥈 2nd", "🏆 1st", "🥉 3rd"],                   
                "Driver": [f'{r["Driver"]["givenName"]} {r["Driver"]["familyName"]}' for r in top3],
                "Team":   [r["Constructor"]["name"] for r in top3],
                "Step":   [2, 3, 1]                     
            })
            
            st.markdown(f"<div style='text-align: center;'>Latest Grand Prix:<br><strong style='font-size: 20px;'>{race_name}</strong></div>", unsafe_allow_html=True)

            fig = px.bar(
                podium_df, x="Place", y="Step", color="Team", text="Driver", labels={"Step": ""},
                color_discrete_map=TEAM_COLORS, category_orders={"Place": ["🥈 2nd", "🏆 1st", "🥉 3rd"]}
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, width=400, height=320,yaxis=dict(showticklabels=False, range=[0,3.6]))
            st.plotly_chart(fig, use_container_width=True)

            podium_sorted = podium_df.sort_values("Step", ascending=False)
            st.table(podium_sorted[["Place", "Driver", "Team"]].set_index("Place"))

        except Exception as e:
            st.error(f"Could not load latest podium: {e}")

    # Current season results
    try:
        rows = requests.get(f"{API}/current/driverStandings.json", timeout=6).json() \
            ["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]

        standings_df = pd.DataFrame([{
            "Pos":   int(r["position"]),
            "Driver":f'{r["Driver"]["givenName"]} {r["Driver"]["familyName"]}',
            "Team":  r["Constructors"][0]["name"],
            "Pts":   int(r["points"]),
            "Wins":  int(r["wins"])
        } for r in rows]).set_index("Pos")

    except Exception as e:
        st.error(f"Could not load championship standings: {e}")
    
    # Summary of results
    driver_standings = standings_df.reset_index()
    top_drivers = driver_standings.head(5)
    leader = top_drivers.iloc[0]
    second = top_drivers.iloc[1]
    third = top_drivers.iloc[2]
    fourth = top_drivers.iloc[3]
    fifth = top_drivers.iloc[4]
    gap_1_2 = int(leader.Pts) - int(second.Pts)
    gap_1_3 = int(leader.Pts) - int(third.Pts)
    gap_3_4 = int(third.Pts) - int(fourth.Pts)
    teams_in_top5 = top_drivers['Team'].value_counts()
    dominant_team = teams_in_top5.idxmax()
    dominant_team_count = teams_in_top5.max()
    
    summary = f"""
    The current Formula 1 season highlights the essence of the sport: speed, competition, strategy, world-class driving and technical innovation.
    **{leader.Driver}**, driving for **{leader.Team}**, currently leads the Drivers' Championship with **{leader.Pts} points**. He holds a lead of **{gap_1_2} points** over second-placed **{second.Driver}** ({second.Team}), and a margin of **{gap_1_3} points** to third-placed **{third.Driver}** ({third.Team}).

    The fight for podium positions is particularly intense. **{third.Driver}**, **{fourth.Driver}**, and **{fifth.Driver}** are separated by only a few points, with just **{gap_3_4} points** between third and fourth place.  Among the top five, **{dominant_team}** stands out with **{dominant_team_count} drivers** currently occupying top spots in the standings. This highlights the team's consistency and strength across multiple races.

    The season is still underway, thus, the title race remains wide open.
      """

    st.markdown("---")
    st.header("Current Season")
    st.markdown(summary)
    st.dataframe(standings_df, use_container_width=True)

    # About Formula 1
    st.markdown("---")
    st.header("About Formula 1")

    # Scraping Wikipedia
    def get_f1_wiki_paragraphs():
        url = "https://en.wikipedia.org/wiki/Formula_One"
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.content, "html.parser")

        for sup in soup.find_all("sup"):
            sup.decompose()

        # Intro - first long paragraph
        intro = ""
        for p in soup.select("p"):
            if len(p.get_text(strip=True)) > 100:
                intro = p.get_text(" ", strip=True)
                break
        # Drivers - first paragraph
        drivers = soup.find(id="Drivers")
        drivers_paragraph = ""
        if drivers:
            for s in drivers.find_parent().find_next_siblings():
                if s.name == "p":
                    drivers_paragraph = s.get_text(strip=True)
                    break
        # Constructors - first paragraph
        constructors = soup.find(id="Constructors")
        constructors_paragraph = ""
        if constructors:
            for s in constructors.find_parent().find_next_siblings():
                if s.name == "p":
                    constructors_paragraph = s.get_text(strip=True)
                    break
        # Logo 
        logo_url = ""
        logo= soup.find("table", class_="infobox")
        if logo:
            img = logo.find("img")
            if img and img.get("src"):
                logo_url = "https:" + img.get("src") 

        return intro, drivers_paragraph, constructors_paragraph, logo_url


    # Display 
    intro, drivers, constructors, logo_url = get_f1_wiki_paragraphs()
    co1, co2 = st.columns([2, 1]) 

    with co1:
        st.markdown(intro)
    with co2:
        if logo_url:
            st.image(logo_url, caption="Formula One logo", width=250)
    st.subheader("Drivers")
    st.markdown(drivers)

    st.subheader("Constructors")
    st.markdown(constructors)

    st.markdown(
        """
        <div style="font-size: 0.9em; color: gray;">
        Source: Adapted from <a href="https://en.wikipedia.org/wiki/Formula_One" target="_blank">Wikipedia – Formula One</a> """, unsafe_allow_html=True
    )




#----------- DRIVERS ------------
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
                "championship_years": [],
                "wins": "N/A",
                "points": "N/A",
                "car_number": "N/A",
                "podiums": "N/A",
                "first_entry": "N/A",
                "first_win": "N/A",
                "last_entry": "N/A"
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

        for i, year in enumerate(years):
            fig.add_shape(
                type="line",
                x0=year,
                y0=0,
                x1=year,
                y1=0.6,
                line=dict(color=dark_blue, width=2)
            )

            fig.add_trace(go.Scatter(
                x=[year],
                y=[0.6],
                mode="markers",
                marker=dict(color=dark_blue, size=20),
                showlegend=False,
                hovertemplate=f"<b>{labels[i]}</b><br>{races[i]}<br>Year: {year}<extra></extra>"
            ))

            fig.add_annotation(
                x=year,
                y=-0.1,
                text=f"<b>{labels[i]}</b><br>{year}",
                showarrow=False,
                yanchor="top",
                textangle=0,
                font=dict(size=10),
                align="center"
            )

            race_name_only = re.sub(r'^(\d{4}\s+)|(\s+\(\d{4}\))$', '', races[i])
            fig.add_annotation(
                x=year,
                y=0.8,
                text=race_name_only,
                textangle=30,
                showarrow=False,
                yanchor="bottom",
                font=dict(size=11, color="gray"),
                align="center",
                xanchor="center"
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

    # Main app
    def main(): 
        drivers = fetch_current_drivers()
        if not drivers:
            return

        driver_names = [driver['name'] for driver in drivers]
        selected_driver_name = st.selectbox("Select a Driver", driver_names)
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
    if __name__ == "__main__":
        main()


#----------- RACE ------------
elif st.session_state.page == 'Races':
    st.title("🏎️ Race Statistics")

    # Year selection
    current_year = datetime.now().year
    year = st.selectbox("Select a Year", list(range(2018, current_year + 1))[::-1])

    def get_race_events(year):
        schedule = fastf1.get_event_schedule(year)
        # Exclude testing or undefined rounds
        schedule = schedule[
            (schedule['EventFormat'] != 'Testing') &
            (~schedule['EventName'].str.contains("Test", case=False, na=False)) &
            (schedule['RoundNumber'].notna())
        ]
        return schedule

    schedule = get_race_events(year)
    race_names = schedule['EventName'].tolist()
    selected_race_name = st.selectbox("Select a Race", race_names)

    selected_race = schedule[schedule['EventName'] == selected_race_name].iloc[0]
    round_number = int(selected_race['RoundNumber'])

    try:
        with st.spinner("Loading data from FastF1..."):
            session = fastf1.get_session(year, round_number, 'R')
            session.load()

        st.title(f"🏁 {selected_race_name} {year}")
        results = session.results
    except Exception as e:
        st.error(f"Could not load session: {e}")

    def get_wikipedia_circuit_map(event_name, year):
        page_title = f"{year} {event_name}".replace(' ', '_')
        url = f"https://en.wikipedia.org/wiki/{page_title}"

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")

            infobox = soup.find("table", {"class": "infobox"})
            if infobox:
                img_tag = infobox.find("img")
                if img_tag and "src" in img_tag.attrs:
                    image_url = "https:" + img_tag["src"]
                    return image_url
        except Exception as e:
            print(f"Error scraping Wikipedia: {e}")

        return None

    def format_ordinal(n):
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    formatted_date = f"{format_ordinal(session.event['EventDate'].day)} {session.event['EventDate'].strftime('%B %Y')}"

    st.markdown(f"""
                On **{formatted_date}**, the **{selected_race_name}** took place in **{session.event['Location']}, {session.event['Country']}**.""")
    col1, col2 = st.columns([2.9, 2.8])
    with col1:
        #Race info
        weather = session.weather_data.iloc[0]
        air_temp = weather['AirTemp']
        track_temp = weather['TrackTemp']
        rainfall = "with rainfall" if weather['Rainfall'] > 0 else "with dry conditions"

        st.markdown(f"""
        ### Race Info
        - Location: {session.event['Location']}, {session.event['Country']}
        - Circuit: {session.event['OfficialEventName']}
        - Date: {session.event['EventDate'].date()}
        - Weather: Air {air_temp}°C, Track {track_temp}°C, {rainfall}
        """)
        #Race Stats
        laps = session.laps

        fastest_lap = laps.pick_fastest()
        fastest_driver_code = fastest_lap['Driver']
        fastest_driver_info = session.get_driver(fastest_driver_code)
        fastest_driver_full_name = f"{fastest_driver_info['FirstName']} {fastest_driver_info['LastName']}"
        
        valid_laps = laps[laps['LapTime'].notna()]
        slowest_lap = valid_laps.sort_values('LapTime').iloc[-1]
        slowest_driver_code = slowest_lap['Driver']
        slowest_driver_info = session.get_driver(slowest_driver_code)
        slowest_driver_full_name = f"{slowest_driver_info['FirstName']} {slowest_driver_info['LastName']}"

        avg_lap_time = laps['LapTime'].mean()


        def format_lap_time(td):
            if pd.isnull(td):
                return "N/A"
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            parts = []
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
            return " ".join(parts)
        
        st.markdown(f"""
        ### Lap Stats
        - Total Laps: {int(laps['LapNumber'].max())}
        - Fastest Lap: {fastest_driver_full_name} - {format_lap_time(fastest_lap['LapTime'])}
        - Slowest Lap: {slowest_driver_full_name} - {format_lap_time(slowest_lap["LapTime"])}
        - Average Lap Time: {format_lap_time(avg_lap_time)}
        """)
        st.markdown(f"""The session included a total of **{int(laps['LapNumber'].max())}** laps. The fastest lap was achieved by **{fastest_driver_full_name}** with a time of **{format_lap_time(fastest_lap['LapTime'])}**. 
                    On the other hand, the slowest lap was recorded by **{slowest_driver_full_name}**, finishing in **{format_lap_time(slowest_lap['LapTime'])}**. The average lap time recorded during the session was approximately **{format_lap_time(avg_lap_time)}**.""")
    with col2:
        map_url = get_wikipedia_circuit_map(selected_race_name, year)
        if map_url:
            st.caption(f"The map of the {selected_race_name} circuit")
            st.image(map_url, use_container_width=True)
        else:
            st.info("Circuit layout map not found on Wikipedia.")

    #positions by laps
    lap_pos = session.laps.groupby(["LapNumber", "Driver"])["Position"].mean().unstack()

    #positions after final lap
    final_lap = lap_pos.index.max()
    final_positions = lap_pos.loc[final_lap].sort_values()
    top3 = final_positions.head(3)

    top3_info = []
    for driver_code in top3.index:
        driver_info = results[results['Abbreviation'] == driver_code].iloc[0]
        full_name = f"{driver_info['FirstName']} {driver_info['LastName']}"
        team = driver_info['TeamName']
        pos = top3.loc[driver_code]
        top3_info.append((full_name, team, int(pos)))

    st.markdown("### 🏆 Results")
    st.markdown(f""" The race concluded with an exciting finish. The winner of the race was **{top3_info[0][0]}**, representing **{top3_info[0][1]}**. 
                Following closely in second place was **{top3_info[1][0]}** from **{top3_info[1][1]}**. Completing the podium in third place was **{top3_info[2][0]}**, driving for **{top3_info[2][1]}**.
                """ )
    col1, col2 = st.columns([1.3, 1.7]) 
    with col1:
        final_classification = []
        for driver_code, pos in final_positions.dropna().items():
            driver_info = results[results['Abbreviation'] == driver_code].iloc[0]
            full_name = f"{driver_info['FirstName']} {driver_info['LastName']}"
            team = driver_info['TeamName']
            final_classification.append({'Position': int(pos), 'Driver': full_name, 'Team': team})

        final_classification_df = pd.DataFrame(final_classification).sort_values(by='Position').set_index("Position")

        st.markdown("Below is the complete list of drivers with their finishing positions and teams:")

        st.dataframe(final_classification_df.style.format({'Position': '{:.0f}'}))

    with col2:
        palette = sns.color_palette("tab20", n_colors=len(lap_pos.columns))

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8))
        lap_pos.plot(ax=ax, linewidth=2, color=palette)

        ax.set_xlabel("Lap Number", fontsize=14)
        ax.set_ylabel("Position", fontsize=14)
        ax.invert_yaxis() 
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="Driver", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

        plt.tight_layout()
        st.subheader("Race Position Chart")
        st.caption("This figure shows how drivers' positions changed throughout the race laps.")
        st.pyplot(fig)

    #number of drivers that did not finish 
    dnfs = session.results[session.results['Status'] != 'Finished']

    dnf_laps = []
    for driver in dnfs['Abbreviation']:
        driver_laps = session.laps.pick_driver(driver)
        if not driver_laps.empty:
            last_lap = int(driver_laps['LapNumber'].max())
            dnf_laps.append((driver, last_lap))
        else:
            dnf_laps.append((driver, None)) 
    
    dnf_lap_df = pd.DataFrame(dnf_laps, columns=['Abbreviation', 'Retired On Lap'])
    dnf_table = dnfs.merge(dnf_lap_df, on='Abbreviation')
    dnf_table = dnf_table[['Abbreviation', 'FullName', 'Status', 'Retired On Lap']]
    dnf_table = dnf_table.rename(columns={
        'Abbreviation': 'Driver Code',
        'FullName': 'Driver Name',
        'Status': 'Reason'
    })
    dnf_table = dnf_table.sort_values('Retired On Lap')

    st.subheader("DNFs")
    st.markdown(f"{len(dnf_table)} drivers did not finish the race.")
    st.dataframe(dnf_table, use_container_width=True)
    st.markdown("_Note: 'Retired' is a general term and may include mechanical failure, damage, or strategic retirement._")

    #pit stop stats
    pit_data = session.laps.loc[session.laps['PitInTime'].notnull()]
    total_pitstops = pit_data.shape[0]
    st.subheader("Pit Stop Stats")
    
    pit_counts = pit_data.groupby('Driver')['LapNumber'].count().reset_index(name='Pit Stops')
    pit_counts['Driver Name'] = pit_counts['Driver'].apply(
    lambda code: f"{session.get_driver(code)['FirstName']} {session.get_driver(code)['LastName']}")
    pit_counts = pit_counts[['Driver Name', 'Pit Stops']].sort_values('Pit Stops', ascending=False)

    max_pits = pit_counts['Pit Stops'].max()
    min_pits = pit_counts['Pit Stops'].min()
    most_pits_df = pit_counts[pit_counts['Pit Stops'] == max_pits]
    fewest_pits_df = pit_counts[pit_counts['Pit Stops'] == min_pits]
    most_pits_drivers = ', '.join(most_pits_df['Driver Name'].tolist())
    fewest_pits_drivers = ', '.join(fewest_pits_df['Driver Name'].tolist())

    st.markdown(f"""
    There were a total of **{total_pitstops}** pit stops made during the race. The driver(s) with the most pit stops was **{most_pits_drivers}**, making **{max_pits}** stops.
     The fewest pit stops were made by **{fewest_pits_drivers}**, with just **{min_pits}** pits.""" )
    
    col1, col2 = st.columns([1.3, 1.7]) 
    with col1: 
        st.caption("Pit Stops per Driver")
        st.dataframe(pit_counts.sort_values('Pit Stops', ascending=False))
    with col2:
        st.caption("The chart shows the timing of pit stops for all drivers.")
        fig, ax = plt.subplots()
        for driver in pit_data['Driver'].unique():
            driver_pits = pit_data[pit_data['Driver'] == driver]
            ax.scatter(driver_pits['LapNumber'], [driver] * len(driver_pits), label=driver, s=50)

        ax.set_xlabel("Lap")
        ax.set_ylabel("Driver")
        st.pyplot(fig)

    #add a button to compare drivers
    st.markdown("""
    ### Driver Comparison

    Select and compare the performance of any two drivers from the selected race.""")
        #select drivers by names
    driver_map = {
        row['Abbreviation']: f"{row['FirstName']} {row['LastName']}"
        for _, row in results.iterrows()}

    name_to_abbr = {v: k for k, v in driver_map.items()}

    driver1_full = st.selectbox("Select Driver 1", list(driver_map.values()))
    driver2_full = st.selectbox("Select Driver 2", list(driver_map.values()))

    if driver1_full and driver2_full and driver1_full != driver2_full:
        driver1 = name_to_abbr[driver1_full]
        driver2 = name_to_abbr[driver2_full]

        laps = session.laps
        driver1_laps = laps.pick_driver(driver1).pick_quicklaps()
        driver2_laps = laps.pick_driver(driver2).pick_quicklaps()

        #DRIVER STATS
        laps_completed1 = len(driver1_laps)
        laps_completed2 = len(driver2_laps)
            
        avg_time1 = driver1_laps['LapTime'].mean()
        avg_time2 = driver2_laps['LapTime'].mean()
            
        fastest1 = driver1_laps['LapTime'].min()
        fastest2 = driver2_laps['LapTime'].min()
            
        pos1 = results[results['Abbreviation'] == driver1]['Position'].values[0]
        pos2 = results[results['Abbreviation'] == driver2]['Position'].values[0]
            
        grid1 = results[results['Abbreviation'] == driver1]['GridPosition'].values[0]
        grid2 = results[results['Abbreviation'] == driver2]['GridPosition'].values[0]
            
        driver1_pits = laps.pick_driver(driver1).query("PitInTime.notnull()")
        driver2_pits = laps.pick_driver(driver2).query("PitInTime.notnull()")

        num_pits1 = len(driver1_pits)
        num_pits2 = len(driver2_pits)

        def format_lap_time(td):
            if isinstance(td, timedelta):
                total_seconds = td.total_seconds()
            else:
                total_seconds = td
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            milliseconds = int((total_seconds - int(total_seconds)) * 1000)
            return f"{minutes}m {seconds}s {milliseconds}ms"

        st.markdown("### Race Stats Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(driver1_full)
            st.write(f"Fastest Lap: `{format_lap_time(fastest1)}`")
            st.write(f"Avg Lap Time: `{format_lap_time(avg_time1)}`")
            st.write(f"Position: `{int(pos1)}`")
            st.write(f"Pits: `{int(num_pits1)}`")
            st.write(f"Grid → Finish: `{int(grid1)} → {int(pos1)}`")

        with col2:
            st.subheader(driver2_full)
            st.write(f"Fastest Lap: `{format_lap_time(fastest2)}`")
            st.write(f"Avg Lap Time: `{format_lap_time(avg_time2)}`")
            st.write(f"Position: `{int(pos2)}`")
            st.write(f"Pits: `{int(num_pits2)}`")
            st.write(f"Grid → Finish: `{int(grid2)} → {int(pos2)}`")
        #plots
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(
                driver1_laps['LapNumber'], 
                driver1_laps['LapTime'].dt.total_seconds(), 
                label=driver1_full, 
                color="red", 
                linewidth=2, 
                marker='o'
            )

            ax.plot(
                driver2_laps['LapNumber'], 
                driver2_laps['LapTime'].dt.total_seconds(), 
                label=driver2_full, 
                color="orange", 
                linewidth=2, 
                marker='s'
            )

            ax.set_title("Lap Time Comparison", fontsize=15)
            ax.set_xlabel("Lap Number", fontsize=14)
            ax.set_ylabel("Lap Time (s)", fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=12)
            plt.tight_layout()

            st.pyplot(fig)
            st.caption("This chart shows how each driver’s lap times changed throughout the race. Dots mark completed laps — lower values indicate faster laps.")

        #compare speed on their fastest lap
        driver1_fastest = laps.pick_driver(driver1).pick_fastest()
        driver2_fastest = laps.pick_driver(driver2).pick_fastest()

        tel1 = driver1_fastest.get_car_data().add_distance()
        tel2 = driver2_fastest.get_car_data().add_distance()

        with col2: 
            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(
                tel1['Distance'], tel1['Speed'], 
                label=f"{driver1_full} Speed", 
                color="red", 
                linewidth=2, 
                linestyle='-'
            )

            ax.plot(
                tel2['Distance'], tel2['Speed'], 
                label=f"{driver2_full} Speed", 
                color="orange", 
                linewidth=2, 
                linestyle='--'
            )

            ax.set_title("Telemetry Speed Comparison", fontsize=15)
            ax.set_xlabel("Distance (m)", fontsize=14)
            ax.set_ylabel("Speed (km/h)", fontsize=14)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(fontsize=12, loc='best')

            plt.tight_layout()
            st.pyplot(fig)
            st.caption("A detailed look at the speed of each driver over distance during their fastest lap. Useful for comparing top speed, braking points, and cornering behavior.")


#----------- SEASONS ------------
elif st.session_state.page == 'Seasons':
    st.title("📅 Season Analytics")

    API = "https://api.jolpi.ca/ergast/f1"
    year = st.selectbox("Season", list(range(1990, 2025+1)), index=24)

    # Race schedule from FastF1
    try:
        sched = fastf1.get_event_schedule(year)
        races = sched[sched.EventName.str.contains("Grand Prix")]
    except Exception as e:
        st.error(f"Schedule error: {e}")
        st.stop()

    # Standings from Jolpica
    try:
        standings_data = requests.get(f"{API}/{year}/driverStandings.json", timeout=6).json()
        all_standings = standings_data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
        driver_id_to_name = {
            d["Driver"]["driverId"]: f"{d['Driver']['givenName']} {d['Driver']['familyName']}"
            for d in all_standings
        }
    except Exception:
        st.error("Could not fetch driver standings.")
        st.stop()

    # All races results (problems with Jolpica encountered, thus, limit was set)
    season_results = []
    offset = 0
    limit = 100
    try:
        while True:
            url = f"{API}/{year}/results.json?limit={limit}&offset={offset}"
            response = requests.get(url, timeout=10).json()
            races_page = response["MRData"]["RaceTable"]["Races"]
            if not races_page:
                break
            season_results.extend(races_page)
            offset += limit
    except Exception as e:
        st.error(f"Failed to fetch race results: {e}")
        st.stop()

    results_by_round = {
        int(race["round"]): race for race in season_results if race.get("Results")
    }

    # Analysis of results
    team_wins = {}
    winner_rows = []
    lines = []
    skipped_rounds=[]
    dnf_counts = defaultdict(int)
    dnf_reasons = defaultdict(int)
    driver_points = defaultdict(float)

    with st.spinner("Processing race data..."):
        for _, race in races.iterrows():
                rnd = int(race.RoundNumber)
                results_entry = results_by_round.get(rnd)
                if not results_entry:
                    skipped_rounds.append((rnd, race.EventName))
                    continue
                results = results_entry["Results"]

                # Winner data
                winner = results[0]
                winner_name = f"{winner['Driver']['givenName']} {winner['Driver']['familyName']}"
                winner_team = winner["Constructor"]["name"]
                team_wins[winner_team] = team_wins.get(winner_team, 0) + 1
                winner_rows.append({"#": rnd, "Race": race.EventName, "Driver": winner_name, "Team": winner_team})

                # DNF data
                for r in results:
                    driver_id = r["Driver"]["driverId"]
                    status = r["status"].lower()
                    if any(k in status for k in ["ret", "not classified", "dnf", "accident", "collision"]):
                        dnf_counts[driver_id] += 1
                        dnf_reasons[status] += 1

                    # For the purpose of next section
                    driver_points[driver_id] += float(r["points"])

                # Top 5 drivers
                top5 = sorted(driver_points.items(), key=lambda x: x[1], reverse=True)[:5]
                for did, pts in top5:
                    driver_name = driver_id_to_name.get(did, did)
                    lines.append({"Round": rnd, "Driver": driver_name, "Points": int(pts)})
    
    # In case of missing or unavailable data
    if skipped_rounds:
        skipped_text = ", ".join([f"{name} (Round {rnd})" for rnd, name in skipped_rounds])
        st.warning(f"⚠ There were recorded missing data in {len(skipped_rounds)} race(s): {skipped_text}")

    # Total Wins By Team (bar chart and summary)
    if team_wins:
        st.subheader("Total Wins by Team")
        st.caption(f"The bar chart displays the number of Grand Prix wins achieved by each team during the **{year}** season.")
        wins_series = pd.Series(team_wins, name="Wins")
        st.bar_chart(wins_series)

        num_races = len(races)
        num_available_races = len(winner_rows)
        top_team, top_wins = max(team_wins.items(), key=lambda x: x[1])
        total_teams = len(team_wins)
        win_pct = round((top_wins / num_races) * 100, 1)
        teams_with_2plus = sum(1 for w in team_wins.values() if w >= 2)
        teams_with_1 = sum(1 for w in team_wins.values() if w == 1)

        st.markdown(
            f"The {year} Formula 1 season featured **{num_races} Grands Prix** (for analysis, {num_available_races} available races are used). \n\n"
            f"Race victories were spread across **{total_teams} different teams**. **{top_team}** led the season with **{top_wins} wins**, accounting for **{win_pct}%** of all races. "
            f"Out of all winning teams, **{teams_with_2plus}** team(s) won two or more races and **{teams_with_1}** team(s) secured a single victory.\n"
        )
        st.markdown("---")

    # Winner of each race in selected season
    if winner_rows:
        st.subheader("Race Winners")
        st.caption("The table lists the winners of each Grand Prix.")
        winner_df = pd.DataFrame(winner_rows)
        winner_df.index = winner_df.index + 1
        
        # Displaying table
        st.dataframe(winner_df.iloc[:, 1:], use_container_width=True)

    else:
        st.info("No winner data returned.")

    # Cumulative points graph of 5 top drivers
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("The line chart shows how the total points of the top drivers evolved round-by-round during the season (using cumulative championship points).")

    df_lines = pd.DataFrame(lines)
    final_round = df_lines["Round"].max()
    final_standings = df_lines[df_lines["Round"] == final_round]
    top5_drivers = final_standings.sort_values("Points", ascending=False).head(5)["Driver"].tolist()
    df_top5 = df_lines[df_lines["Driver"].isin(top5_drivers)]
    if lines:
        fig = px.line(df_top5, x="Round", y="Points", color="Driver", markers=True)
        fig.update_layout(
            xaxis_title="Round",
            yaxis_title="Points",
            legend_title=None,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        sorted_standings = final_standings.sort_values("Points", ascending=False).head(5) 
        season_leader = sorted_standings.iloc[0]
        season_leader2 = sorted_standings.iloc[1]
        season_leader3 = sorted_standings.iloc[2]
        leader_points = season_leader["Points"]
        leader_points2 = season_leader2["Points"]
        leader_points3 = season_leader3["Points"]
        top_points = sorted_standings["Points"].tolist()
        gaps = [leader_points - p for p in top_points[1:]]
        min_gap = min(gaps) if gaps else 0
        max_gap = max(gaps) if gaps else 0
        close_to_win = sum(1 for p in top_points[1:] if leader_points - p <= 10)
        top_driver = max(set([row["Driver"] for row in winner_rows]), key=[row["Driver"] for row in winner_rows].count)
        top_count = [row["Driver"] for row in winner_rows].count(top_driver)

        st.markdown(
            f"After **{final_round} rounds** of the **{year} Formula 1 season**, **{season_leader['Driver']}** led the championship with **{int(leader_points)} points**. **{season_leader2['Driver']}** and **{season_leader3['Driver']}**  ended up being at the second and third place with **{int(leader_points2)}** and **{int(leader_points3)} points**, respectively. The final standings are visualized below. \n\n"
            f"The points gap between the leader and the rest of the 5 top ranged from **{min_gap}** to **{max_gap} points**. "
            f"There were **{close_to_win} drivers** within 10 points of the leader. "
            f"The driver in **5th place** held **{int(top_points[4])} points**, a total spread of **{leader_points - top_points[4]} points from the leader**.\n\n"
            f"**{top_driver}** won the most races, specifically **{top_count}** of them."
            )
   
    else:
        st.markdown("No points data available.")

    st.markdown("---")
    
    # Visualising final results
    st.subheader("Championships")

    # Best 3 racers of season
    st.caption("Drivers' Championship")
    col2, col1, col3 = st.columns([1, 1.2, 1])
    with col1:
        st.markdown("### 🥇")
        st.markdown(f"**{season_leader['Driver']}**")
        st.markdown(f"{int(season_leader['Points'])} points")
    with col2:
        st.markdown("### 🥈")
        st.markdown(f"**{season_leader2['Driver']}**")
        st.markdown(f"{int(season_leader2['Points'])} points")
    with col3:
        st.markdown("### 🥉")
        st.markdown(f"**{season_leader3['Driver']}**")
        st.markdown(f"{int(season_leader3['Points'])} points")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Best 3 constructors of season
    st.caption("Constructors' Championship")
    constructor_respose = requests.get(f"{API}/{year}/constructorStandings.json", timeout=6).json()
    constructor_standings = constructor_respose["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]

    try:
        col2, col1, col3 = st.columns([1, 1.2, 1])
        with col1:
            st.markdown("### 🥇")
            st.markdown(f"**{constructor_standings[0]["Constructor"]["name"]}**")
            st.markdown(f"{int(constructor_standings[0]["points"])} points")
        with col2:
            st.markdown("### 🥈")
            st.markdown(f"**{constructor_standings[1]["Constructor"]["name"]}**")
            st.markdown(f"{int(constructor_standings[1]["points"])} points")
        with col3:
            st.markdown("### 🥉")
            st.markdown(f"**{constructor_standings[2]["Constructor"]["name"]}**")
            st.markdown(f"{int(constructor_standings[2]["points"])} points")
    except Exception as e:
        st.warning("Could not load constructor standings.")
        st.caption(f"Error: {e}")
    
    st.markdown(f"Grands Prix use a scoring system to determine the winners of two yearly championships: one for individual drivers and one for teams (constructors). \
                The Drivers' Championship is awarded to the driver who accumulates the most points over the course of a season. \
                In {year}, **{season_leader['Driver']}** won the Drivers' Championship with **{int(season_leader['Points'])}** points. Second and third was **{season_leader2['Driver']}** and **{season_leader3['Driver']}**, respectively. \
                The Constructors' Championship points are calculated by adding points scored in each race by any driver for that constructor. \
                In {year}, **{constructor_standings[0]["Constructor"]["name"]}** won the Constructors' Championship with **{int(constructor_standings[0]["points"])}** points. Second and third scored **{constructor_standings[1]["Constructor"]["name"]} and {constructor_standings[2]["Constructor"]["name"]}**, respectively.")
    
    st.markdown("---")

    # DNFs
    st.subheader("DNFs ('Did Not Finish') Across All Drivers")
    st.caption("The graph shows the number of times each driver failed to finish a race during the season (top 5 drivers of season highlighted in red).")

    if dnf_counts:
        dnf_df = pd.DataFrame({
            "Driver": [driver_id_to_name.get(did, did) for did in dnf_counts],
            "DNFs": [dnf_counts[did] for did in dnf_counts]
        }).sort_values("DNFs", ascending=False)

        top5_names = set(sorted_standings["Driver"].tolist())
        dnf_df["Top5"] = dnf_df["Driver"].apply(lambda x: x in top5_names)
        fig_dnf = px.bar(
            dnf_df,
            x="Driver",
            y="DNFs",
            color="Top5",
            color_discrete_map={True: "crimson", False: "gray"}
            )
        fig_dnf.update_layout(showlegend=False, xaxis_title=None, yaxis_title="DNFs")
        
        # Plot DNFs
        st.plotly_chart(fig_dnf, use_container_width=True)

        total_drivers = len(dnf_counts)
        total_dnfs = sum(dnf_counts.values())
        most_dnf_driver = dnf_df.iloc[0]

        #Reasons of DNFs
        if dnf_reasons:
            reason_df = pd.DataFrame({
                "Reason": list(dnf_reasons.keys()),
                "Count": list(dnf_reasons.values())
            }).sort_values("Count", ascending=False)
            reason_df.index = reason_df.index + 1
            top_reason = reason_df.iloc[0]

        # Summary    
        st.markdown(
            f"In the {year} season, a total of **{total_dnfs} retirements (DNFs)** were recorded among **{total_drivers} drivers**. "
            f"**{most_dnf_driver['Driver']}** had the most reliability issues, retiring from **{most_dnf_driver['DNFs']} races**. "
            f"High DNF counts may point to reliability issues or frequent involvement in incidents. These figures thus highlight how mechanical reliability and incident avoidance play a key role in Formula 1; even one or two retirements in a tight season can have a dramatic impact on final standings. \n\n"
        
            f"The most common reasons of DNFs include mechanical failures, crashes, and other incidents. As shown in the table below, in the **{year} Formula 1 season**, the most prevalent cause was **{top_reason['Reason']}**, which occurred **{top_reason['Count']} times**.\n"
            f"The following table offers insights into whether retirements were more often due to mechanical failure, accidents, or racing incidents — essential knowledge for teams aiming to improve reliability and racecraft."         
            )

        st.caption("The table lists the most frequent causes of retirements during the season.")
        
        # Display table with reasons of DNFs
        st.dataframe(reason_df, use_container_width=True)
    else:
        st.info("No DNF data available for this season.")
    
    


#----------- ABOUT THE APP ------------
with st.sidebar:
    st.markdown("---")
    with st.expander("About this app"):
        st.markdown("""
        ### Formula 1 Analytics

        #### Powered by
        - [FastF1](https://theoehrly.github.io/Fast-F1/) – detailed race data
        - [Ergast API](https://api.jolpi.ca/ergast/f1) via [Jolpica proxy](https://api.jolpi.ca/) – historical F1 results
        - [Wikipedia](https://www.wikipedia.org/) – driver info, images and additional statistics 
        - [Streamlit](https://streamlit.io/) – for builidng the app

        #### Developer
        *Adéla Bláhová, Anna Marie Břicháčková*  
        Version: `0.1.0`  
        GitHub: [github.com/blahovaa/Python-project](https://github.com/blahovaa/Python-project)

        ---
        _This project is for educational and demonstration purposes only. All data belongs to its respective providers._
        """)
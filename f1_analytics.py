import streamlit as st
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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

st.set_page_config(layout="wide")
         

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

previous_page = st.session_state.get('page', 'Home')

cache_path = 'fastf1_cache'
os.makedirs(cache_path, exist_ok=True)

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

# Main content based on current page
if st.session_state.page == 'Home':
    col1, col2 = st.columns([2, 1]) 

    with col1:
        st.title("Formula 1 Analytics")
        st.markdown("""
        Welcome to **Formula 1 Analytics**, a data-driven dashboard built for exploring and visualizing key moments and trends from the world of Formula 1. This project pulls in real F1 race data using the FastF1 and Ergast APIs, making it easy to dive into statistics across seasons, drivers, and races.

        Right here on the **Home** page, you’ll find a summary of the most recent Grand Prix podium, along with the current driver standings for the ongoing season.

        Use the sidebar to explore more:
        - Browse detailed **driver profiles** with extra info pulled from Wikipedia
        - Analyze how **teams have performed** across past seasons
        - Compare **individual races**, including lap times, race pace, and telemetry charts

        Whether you're a fan, a motorsport analyst, or just curious, this tool offers a simple way to understand the numbers behind the racing.
         """)

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
            fig.update_layout(showlegend=False, yaxis=dict(showticklabels=False, range=[0,3.6]))
            st.plotly_chart(fig, use_container_width=True)

            podium_sorted = podium_df.sort_values("Step", ascending=False)
            st.table(podium_sorted[["Place", "Driver", "Team"]].set_index("Place"))

        except Exception as e:
            st.error(f"Could not load latest podium: {e}")

    # Full-width section for championship standings
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

        st.markdown("### Current Season – Driver Results")
        st.dataframe(standings_df, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load championship standings: {e}")



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
            st.markdown(" ")
            if driver_info.get("image"):
                st.image(driver_info["image"], width=180, caption=selected_driver['name'])
            else:
                st.info("No image available.")
    if __name__ == "__main__":
        main()


#----------- RACE ------------
elif st.session_state.page == 'Races':
    st.title("🏎️ Race Statistics")

    
    # Enable cache (run once)
    fastf1.Cache.enable_cache('fastf1_cache')

    # Year selection
    current_year = datetime.now().year
    year = st.selectbox("Select a Year", list(range(2018, current_year + 1))[::-1])

    # Get event schedule and filter out testing events
    @st.cache_data
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

    # Load the race session
    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load()

        st.title(f"🏁 {selected_race_name} {year}")
        results = session.results
    except Exception as e:
        st.error(f"Could not load session: {e}")
    
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

    # Display top 3 drivers
    st.markdown("### 🏆 Podium")
    for i, (name, team, pos) in enumerate(top3_info, 1):
        st.markdown(f"**{i}. {name}** ({team})")

    palette = sns.color_palette("tab20", n_colors=len(lap_pos.columns))

    # Plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    lap_pos.plot(ax=ax, linewidth=2, color=palette)

    ax.set_title("Race Position Chart", fontsize=18, weight='bold')
    ax.set_xlabel("Lap Number", fontsize=14)
    ax.set_ylabel("Position", fontsize=14)
    ax.invert_yaxis() 
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(title="Driver", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    #add a button to compare drivers
    if st.button("Compare Drivers"):
        st.session_state.compare_mode = True

    if 'compare_mode' not in st.session_state:
        st.session_state.compare_mode = False

    if st.session_state.compare_mode:
        laps = session.laps

        #select drivers by names
        driver_map = {
            row['Abbreviation']: f"{row['FirstName']} {row['LastName']}"
            for _, row in results.iterrows()
        }

        name_to_abbr = {v: k for k, v in driver_map.items()}

        driver1_full = st.selectbox("Select Driver 1", list(driver_map.values()))
        driver2_full = st.selectbox("Select Driver 2", list(driver_map.values()))

        driver1 = name_to_abbr[driver1_full]
        driver2 = name_to_abbr[driver2_full]

        driver1_laps = laps.pick_driver(driver1).pick_quicklaps()
        driver2_laps = laps.pick_driver(driver2).pick_quicklaps()

        #DRIVER STATS
        #laps completed
        laps_completed1 = len(driver1_laps)
        laps_completed2 = len(driver2_laps)
        #average lap time
        avg_time1 = driver1_laps['LapTime'].mean()
        avg_time2 = driver2_laps['LapTime'].mean()
        #fastest lap
        fastest1 = driver1_laps['LapTime'].min()
        fastest2 = driver2_laps['LapTime'].min()
        #position at finish
        pos1 = results[results['Abbreviation'] == driver1]['Position'].values[0]
        pos2 = results[results['Abbreviation'] == driver2]['Position'].values[0]
        #start vs. finish position
        grid1 = results[results['Abbreviation'] == driver1]['GridPosition'].values[0]
        grid2 = results[results['Abbreviation'] == driver2]['GridPosition'].values[0]
        #pit stops
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

        st.markdown("### 🧮 Race Stats Comparison")
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
        palette = sns.color_palette("Set2", n_colors=2)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            driver1_laps['LapNumber'], 
            driver1_laps['LapTime'].dt.total_seconds(), 
            label=driver1_full, 
            color=palette[0], 
            linewidth=2, 
            marker='o'
        )

        ax.plot(
            driver2_laps['LapNumber'], 
            driver2_laps['LapTime'].dt.total_seconds(), 
            label=driver2_full, 
            color=palette[1], 
            linewidth=2, 
            marker='s'
        )

        ax.set_title(f"Lap Time Comparison: {driver1_full} vs {driver2_full}", fontsize=16, weight='bold')
        ax.set_xlabel("Lap Number", fontsize=14)
        ax.set_ylabel("Lap Time (s)", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)
        plt.tight_layout()

        st.pyplot(fig)

        #compare speed on their fastest lap
        driver1_fastest = laps.pick_driver(driver1).pick_fastest()
        driver2_fastest = laps.pick_driver(driver2).pick_fastest()

        tel1 = driver1_fastest.get_car_data().add_distance()
        tel2 = driver2_fastest.get_car_data().add_distance()

        fig, ax = plt.subplots(figsize=(12, 6))

        palette = sns.color_palette("Set2", n_colors=2)

        ax.plot(
            tel1['Distance'], tel1['Speed'], 
            label=f"{driver1_full} Speed", 
            color=palette[0], 
            linewidth=2, 
            linestyle='-'
        )

        ax.plot(
            tel2['Distance'], tel2['Speed'], 
            label=f"{driver2_full} Speed", 
            color=palette[1], 
            linewidth=2, 
            linestyle='--'
        )

        ax.set_title(f"Telemetry Speed Comparison: {driver1_full} vs {driver2_full}", fontsize=16, weight='bold')
        ax.set_xlabel("Distance (m)", fontsize=14)
        ax.set_ylabel("Speed (km/h)", fontsize=14)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=12, loc='best')

        plt.tight_layout()
        st.pyplot(fig)

        if st.button("❌ Hide Comparison"):
            st.session_state.compare_mode = False



elif st.session_state.page == 'Seasons':
    st.title("📅 Season Analytics")

    API = "https://api.jolpi.ca/ergast/f1"
    year = st.selectbox("Season", list(range(1990, 2025+1)), index=24)

    # Number of Grand Prix
    try:
        sched = fastf1.get_event_schedule(year)
        races = sched[sched.EventName.str.contains("Grand Prix")]
    except Exception as e:
        st.error(f"Schedule error: {e}")
        st.stop()

    st.write(f"There was {len(races)} Grands Prix in {year}")

    # Extracting data from Jolpica
    team_wins, rows = {}, []
    with st.spinner("Loading winners from Jolpica…"):
        for _, race in races.iterrows():
            rnd = int(race.RoundNumber)  # adjusting the number format
            url = f"{API}/{year}/{rnd}/results.json"
            try:
                res   = requests.get(url, timeout=6).json()
                first = res["MRData"]["RaceTable"]["Races"][0]["Results"][0]

                driver = f'{first["Driver"]["givenName"]} {first["Driver"]["familyName"]}'
                team   = first["Constructor"]["name"]

                team_wins[team] = team_wins.get(team, 0) + 1
                rows.append({"Race": race.EventName, "Driver": driver, "Team": team})
            except Exception:
                pass 

    if team_wins:
        st.subheader("🏆 Wins by Team")
        st.bar_chart(pd.Series(team_wins, name="Wins"))

    if rows:
        st.subheader("🏁 Race Winners")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No winner data returned by Jolpica.")

    # Progress of drivers
    st.subheader("📈 Cumulative Points Progress – Top 5 Drivers")

    lines = []
    for rnd in range(1, len(races) + 1):
        try:
            rows = requests.get(f"{API}/{year}/{rnd}/driverStandings.json", timeout=6).json() \
                   ["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
            for r in rows[:5]:                     # only top-5 keep chart clear
                lines.append({
                    "Round": rnd,
                    "Driver": r["Driver"]["familyName"],    # surname only
                    "Points": int(r["points"])
                })
        except Exception:
            pass

    if lines:
        df_lines = pd.DataFrame(lines)
        fig = px.line(df_lines, x="Round", y="Points", color="Driver",
                      markers=True,
                      title=f"{year} – Cumulative Championship Points (Top 5)")
        fig.update_layout(xaxis_title="Round", yaxis_title="Points",
                          legend_title=None, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No points data returned by Jolpica.")

   


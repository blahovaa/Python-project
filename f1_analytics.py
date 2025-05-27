import streamlit as st
import fastf1 as ff1
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os
import requests 
         

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
    st.title("🏁 Formula 1 Analytics")
    st.write("Welcome! Use the sidebar to explore F1 data by Driver, Season, or Race.")
    
    # Extract data from Jolpica
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

        st.markdown(f"## Latest Grand Prix – **{race_name}**")

        fig = px.bar(
            podium_df, x="Place", y="Step", color="Team", text="Driver", labels={"Step": ""},    
            color_discrete_map=TEAM_COLORS, category_orders={"Place": ["🥈 2nd", "🏆 1st", "🥉 3rd"]})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis=dict(showticklabels=False, range=[0,3.6]))
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

        st.markdown("### Current Season – Driver Results")
        st.dataframe(standings_df, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load championship standings: {e}")



elif st.session_state.page == 'Drivers':
    st.title("🏎️ Driver Statistics")
    st.write("Which driver's statistics would you like to see?")
    drivers = ['Lewis Hamilton', 'Max Verstappen', 'Charles Leclerc']
    selected_driver = st.selectbox("Choose a driver", drivers)
    st.write(f"Showing stats for **{selected_driver}**")

elif st.session_state.page == 'Seasons':
    st.title("📅 Season Analytics")

    API = "https://api.jolpi.ca/ergast/f1"
    year = st.selectbox("Season", list(range(1990, 2025+1)), index=24)

    # Number of Grand Prix
    try:
        sched = ff1.get_event_schedule(year)
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
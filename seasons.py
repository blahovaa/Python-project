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

def f_seasons():
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
    with st.spinner("Processing race data..."):
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
    
    
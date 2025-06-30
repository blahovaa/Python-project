import streamlit as st
import fastf1
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

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

def f_home():
    # Making 2 columns 
    col1, col2 = st.columns([2, 1]) 

    with col1:
        st.title("Formula 1 Analytics")
        st.markdown("""
        Welcome to **Formula 1 Analytics**, a data-driven dashboard built for exploring and visualizing key moments and trends from the world of Formula 1. This project pulls in real F1 race data using the FastF1 package and Ergast API, making it easy to dive into statistics across seasons, drivers, and races.

        Right here on the **Home** page, you’ll find a summary of the most recent Grand Prix podium, along with the current driver standings for the ongoing season. Furthermore, a brief overview of Formula 1, scraped from Wikipedia, is provided below.

        Use the sidebar to explore more:
        - Browse detailed **driver profiles** of active drivers in the current season with extra info pulled from Wikipedia
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

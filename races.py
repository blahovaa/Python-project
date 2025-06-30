import streamlit as st
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import requests 
from bs4 import BeautifulSoup
from datetime import datetime
from datetime import timedelta
import re
import seaborn as sns

def f_races():
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

    race_date = pd.to_datetime(selected_race['EventDate'])
    today = pd.Timestamp.today()

    if race_date > today:
        st.warning(f"The {selected_race_name} race did not take place yet. It is scheduled for {race_date.strftime('%d %B %Y')}.")
    else:
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
                st.image(map_url, width = 600)
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


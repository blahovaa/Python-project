# Python-project

# F1 Analytics
Welcome to **Formula 1 Analytics**, an interactive Python dashboard that provides insights into the world of Formula 1. This tool visualizes driver standings, constructor performance, race results, lap times and much more. It is built using real-world data from FastF1 and the Ergast API, enriched with information scraped from Wikipedia.

Created by: **Adéla Bláhová** & **Anna Marie Břicháčková**  
Version: `0.1.0`  
GitHub: [https://github.com/blahovaa/Python-project](https://github.com/blahovaa/Python-project)

---

## Features

- Overview of the latest Grand Prix podium
- Live driver standings and race summaries for the current season
- Brief introduction of Formula 1 (via scraping of Wikipedia)
- Driver profiles (via scraping of Wikipedia)
- Constructor statistics, championship comparison
- Season analysis with race-by-race breakdowns and cumulative point charts
- Analysis of selected race
- Race circuit maps
- Lap-by-lap race position visualizations
- Driver telemetry and lap comparison tools
- DNF (Did Not Finish) statistics and retirement reasons

---

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/blahovaa/Python-project.git
cd Python-project
```

### 2. Install all required packages

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run main.py
```

---
## How it works?

The dashboard consists of a single Streamlit application split into 4 main pages:
- `Home`: Introduction, current standings, brief info about Formula 1
- `Drivers`: Select driver - driver stats, career data
- `Races`: Select race in a specific year - interactive race session viewer & Select drivers - comparison
- `Seasons`: Select season - historical analysis


Navigation is handled via the sidebar.

---

## Project Structure

```
f1_analytics.py
│
├── main.py             # Main entry point for the app
├── home.py             
├── drivers.py          
├── seasons.py          
├── races.py            
└── requirements.txt  
```

Originally, all logic was contained in a single script: f1_analytics.py (now working file, where commits of each creator can be found). For better maintainability, the app has been recretated so that main.py handles navigation and page routing, while each individual page’s functionality is moved into its own dedicated file (home.py, drivers.py, etc.).

---



## Data Sources

This application integrates multiple data sources:

- **FastF1**: 
    > Schaefer, P. (n.,d.). *FastF1* (Version 3.5.3). https://theoehrly.github.io/Fast-F1

- **Ergast API (via Jolpica Proxy)**: 
    > Jolpica. (n.d.). Jolpica F1 API. Retrieved June 24, 2025, from https://api.jolpi.ca/ergast/f1/
    > Github: jolpica-f1. https://github.com/jolpica/jolpica-f1

- **Wikipedia**: 
    > Formula One. (n.d.). *Formula One*. In Wikipedia. https://en.wikipedia.org/wiki/Formula_One 
    > Driver profiles


---

## Final Words

We hope this project helps you explore and understand Formula 1 from a data perspective. Whether you're a fan, student, or analyst, this dashboard is designed to provide insights in an understandable and interactive way.

Thank you and enjoy the app!

*Adéla & Anna*

---

## Disclaimer

This project is built purely for educational and demonstration purposes. All logos, images, and trademarks belong to their respective owners. All data is sourced from public APIs.


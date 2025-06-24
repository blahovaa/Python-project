import streamlit as st
from home import f_home
from drivers import f_drivers
from seasons import f_seasons
from races import f_races

st.set_page_config(layout="wide")

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

if st.session_state.page == 'Home':
    f_home()
elif st.session_state.page == 'Drivers':
    f_drivers()
elif st.session_state.page == 'Seasons':
    f_seasons()
elif st.session_state.page == 'Races':
    f_races()


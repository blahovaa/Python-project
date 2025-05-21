import streamlit as st


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
    st.write("Which driver's statistics would you like to see?")
    drivers = ['Lewis Hamilton', 'Max Verstappen', 'Charles Leclerc']
    selected_driver = st.selectbox("Choose a driver", drivers)
    st.write(f"Showing stats for **{selected_driver}**")
    
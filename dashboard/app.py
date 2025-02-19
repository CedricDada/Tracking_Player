import streamlit as st
import pandas as pd
import plotly.express as px

def show_dashboard(tracking_data):
    st.title("Football Tracking Analytics")
    
    df = pd.DataFrame(tracking_data)
    
    # Heatmap des positions
    fig = px.density_heatmap(df, x='x', y='y', nbinsx=20, nbinsy=14)
    st.plotly_chart(fig)
    
    # Statistiques des joueurs
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vitesse moyenne")
        st.bar_chart(df.groupby('id')['speed'].mean())
    
    with col2:
        st.subheader("Distance parcourue")
        st.line_chart(df.groupby('id')['distance'].sum())
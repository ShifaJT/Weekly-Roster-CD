import streamlit as st
import pandas as pd
import numpy as np
import pulp
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Call Center Roster Optimizer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .shift-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRosterOptimizer:
    def __init__(self):
        self.required_calling_minutes = 7 * 60 + 50
        self.total_break_minutes = 70
        self.break_components = {'morning': 15, 'lunch': 30, 'evening': 15, 'leisure': 10}
        
        self.languages = ['hi', 'en', 'te', 'ta', 'ka']
        
        # Default configuration
        self.config = {
            'total_champions': 23,
            'straight_shifts': st.session_state.get('straight_shifts', 15),
            'split_shifts': st.session_state.get('split_shifts', 8),
            'language_distribution': {'hi': 0.25, 'en': 0.30, 'te': 0.15, 'ta': 0.15, 'ka': 0.15}
        }
        
        self.champions = self.load_champions()
    
    def load_champions(self):
        return [
            # Hindi (6 champions)
            {"name": "Revathi", "language": "hi", "calls_per_hour": 12, "can_split": True},
            {"name": "Pasang", "language": "hi", "calls_per_hour": 11, "can_split": False},
            {"name": "Anjali", "language": "hi", "calls_per_hour": 13, "can_split": True},
            {"name": "Rakesh", "language": "hi", "calls_per_hour": 10, "can_split": False},
            {"name": "Mohammed Altaf", "language": "hi", "calls_per_hour": 12, "can_split": True},
            {"name": "Sameer Pasha", "language": "hi", "calls_per_hour": 11, "can_split": True},
            
            # English (5 champions)  
            {"name": "Kavya S", "language": "en", "calls_per_hour": 15, "can_split": False},
            {"name": "Alwin", "language": "en", "calls_per_hour": 14, "can_split": True},
            {"name": "Pooja N", "language": "en", "calls_per_hour": 16, "can_split": True},
            {"name": "Navya", "language": "en", "calls_per_hour": 13, "can_split": False},
            {"name": "Divya", "language": "en", "calls_per_hour": 15, "can_split": True},
            
            # Telugu (4 champions)
            {"name": "Marcelina J", "language": "te", "calls_per_hour": 8, "can_split": False},
            {"name": "Sadanad", "language": "te", "calls_per_hour": 9, "can_split": True},
            {"name": "Jyothika", "language": "te", "calls_per_hour": 10, "can_split": True},
            {"name": "Vishal", "language": "te", "calls_per_hour": 8, "can_split": True},
            
            # Tamil (4 champions)
            {"name": "Binita Kongadi", "language": "ta", "calls_per_hour": 9, "can_split": True},
            {"name": "Dundesh", "language": "ta", "calls_per_hour": 8, "can_split": False},
            {"name": "Muthahir", "language": "ta", "calls_per_hour": 10, "can_split": True},
            {"name": "Shashindra", "language": "ta", "calls_per_hour": 9, "can_split": True},
            
            # Kannada (4 champions)
            {"name": "Malikarjun Patil", "language": "ka", "calls_per_hour": 8, "can_split": False},
            {"name": "Rakshith", "language": "ka", "calls_per_hour": 9, "can_split": True},
            {"name": "M Showkath Nawaz", "language": "ka", "calls_per_hour": 10, "can_split": True},
            {"name": "Soubhikotl", "language": "ka", "calls_per_hour": 8, "can_split": True}
        ]
    
    def analyze_uploaded_data(self, uploaded_file):
        """Analyze the uploaded Excel file"""
        try:
            df = pd.read_excel(uploaded_file, sheet_name='Sheet2')
            
            hourly_volume = {}
            for hour in range(7, 22):
                hour_key = f"{hour}.0"
                if hour_key in df.index:
                    try:
                        hourly_volume[hour] = df.loc[hour_key, 'Average Calls']
                    except:
                        hourly_volume[hour] = 0
            
            # Calculate required agents
            required_agents = {}
            for hour, calls in hourly_volume.items():
                required_agents[hour] = max(1, round(calls / 12))
            
            # Identify peak hours
            peak_hours = sorted(hourly_volume.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'hourly_volume': hourly_volume,
                'required_agents': required_agents,
                'peak_hours': [h[0] for h in peak_hours],
                'total_daily_calls': sum(hourly_volume.values()),
                'peak_volume': max(hourly_volume.values()) if hourly_volume else 0
            }
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
    
    def generate_roster(self, analysis):
        """Generate optimized roster"""
        try:
            # Generate shifts based on configuration
            shifts = self.generate_shifts()
            
            # Simple assignment for demo
            roster = []
            for i, champ in enumerate(self.champions[:self.config['total_champions']]):
                if i < self.config['straight_shifts']:
                    shift_type = "straight"
                    shift_time = f"{7 + (i % 5)}:00 - {16 + (i % 5)}:00"
                else:
                    shift_type = "split"
                    shift_time = f"{7 + (i % 4)}:00-11:00 & 16:30-21:00"
                
                roster.append({
                    'Champion': champ['name'],
                    'Language': champ['language'],
                    'Shift Type': shift_type,
                    'Shift Time': shift_time,
                    'Calls/Hour': champ['calls_per_hour']
                })
            
            return pd.DataFrame(roster)
            
        except Exception as e:
            st.error(f"Error generating roster: {str(e)}")
            return None
    
    def generate_shifts(self):
        """Generate shift patterns"""
        shifts = {'straight': [], 'split': []}
        
        straight_patterns = [(7,16), (8,17), (9,18), (10,19), (11,20)]
        split_patterns = [(7,11,16.5,21), (8,12,16.5,21), (9,13,16.5,21), (10,14,16.5,21)]
        
        for i in range(self.config['straight_shifts']):
            pattern = straight_patterns[i % len(straight_patterns)]
            shifts['straight'].append({
                'start': pattern[0], 'end': pattern[1], 'type': 'straight'
            })
        
        for i in range(self.config['split_shifts']):
            pattern = split_patterns[i % len(split_patterns)]
            shifts['split'].append({
                'morning_start': pattern[0], 'morning_end': pattern[1],
                'evening_start': pattern[2], 'evening_end': pattern[3],
                'type': 'split'
            })
        
        return shifts

    def plot_call_volume(self, analysis):
        """Create visualization of call volume"""
        if not analysis or 'hourly_volume' not in analysis:
            return None
        
        hours = list(analysis['hourly_volume'].keys())
        calls = list(analysis['hourly_volume'].values())
        
        fig = px.bar(
            x=hours, y=calls,
            labels={'x': 'Hour of Day', 'y': 'Number of Calls'},
            title='Hourly Call Volume Distribution'
        )
        
        # Add peak hour annotations
        for peak_hour in analysis['peak_hours']:
            fig.add_vline(x=peak_hour, line_dash="dash", line_color="red")
        
        return fig

# Main application
def main():
    st.markdown('<h1 class="main-header">📞 Call Center Roster Optimizer</h1>', unsafe_allow_html=True)
    
    # Initialize optimizer
    optimizer = StreamlitRosterOptimizer()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Shift Settings")
        straight_shifts = st.slider("Straight Shifts", 1, 23, 15, key='straight_shifts')
        split_shifts = st.slider("Split Shifts", 1, 23, 8, key='split_shifts')
        
        if straight_shifts + split_shifts != 23:
            st.warning("Total shifts must equal 23! Adjusting split shifts...")
            split_shifts = 23 - straight_shifts
            st.session_state.split_shifts = split_shifts
        
        st.subheader("Language Distribution")
        for lang in optimizer.languages:
            st.write(f"{lang.upper()}: {optimizer.config['language_distribution'][lang] * 100}%")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📊 Upload Data")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
        
        if uploaded_file:
            with st.spinner("Analyzing call data..."):
                analysis = optimizer.analyze_uploaded_data(uploaded_file)
                
                if analysis:
                    st.success("✅ Data analyzed successfully!")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Daily Calls", f"{analysis['total_daily_calls']:,.0f}")
                    with col2:
                        st.metric("Peak Hour Volume", f"{analysis['peak_volume']:,.0f}")
                    with col3:
                        st.metric("Peak Hours", ", ".join(map(str, analysis['peak_hours'])))
                    
                    # Store analysis in session state
                    st.session_state.analysis = analysis
    
    with col2:
        st.header("📈 Call Volume Analysis")
        
        if 'analysis' in st.session_state:
            fig = optimizer.plot_call_volume(st.session_state.analysis)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload an Excel file to see call volume analysis")
    
    # Generate Roster Button
    if st.button("🚀 Generate Optimized Roster", type="primary", use_container_width=True):
        if 'analysis' in st.session_state:
            with st.spinner("Generating optimized roster..."):
                roster_df = optimizer.generate_roster(st.session_state.analysis)
                
                if roster_df is not None:
                    st.header("📋 Generated Roster")
                    
                    # Display roster
                    st.dataframe(roster_df, use_container_width=True)
                    
                    # Show statistics
                    st.subheader("📊 Roster Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Champions", len(roster_df))
                    with col2:
                        st.metric("Straight Shifts", len(roster_df[roster_df['Shift Type'] == 'straight']))
                    with col3:
                        st.metric("Split Shifts", len(roster_df[roster_df['Shift Type'] == 'split']))
                    
                    # Language distribution
                    st.subheader("🌐 Language Distribution")
                    lang_dist = roster_df['Language'].value_counts()
                    st.plotly_chart(px.pie(
                        values=lang_dist.values,
                        names=lang_dist.index,
                        title="Champion Language Distribution"
                    ), use_container_width=True)
                    
                    # Download button
                    csv = roster_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Roster as CSV",
                        data=csv,
                        file_name="call_center_roster.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("Please upload and analyze data first!")

if __name__ == "__main__":
    main()

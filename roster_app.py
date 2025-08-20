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
        
        # Updated configuration for Hindi/Telugu/Kannada focus
        self.config = {
            'total_champions': 23,
            'straight_shifts': st.session_state.get('straight_shifts', 15),
            'split_shifts': st.session_state.get('split_shifts', 8),
            'language_distribution': {'hi': 0.35, 'en': 0.20, 'te': 0.20, 'ta': 0.10, 'ka': 0.15}
        }
        
        self.champions = self.load_champions()
    
    def load_champions(self):
        return [
            # Hindi (8 champions) - Increased for high volume
            {"name": "Revathi", "language": "hi", "calls_per_hour": 12, "can_split": True},
            {"name": "Pasang", "language": "hi", "calls_per_hour": 11, "can_split": False},
            {"name": "Anjali", "language": "hi", "calls_per_hour": 13, "can_split": True},
            {"name": "Rakesh", "language": "hi", "calls_per_hour": 10, "can_split": False},
            {"name": "Mohammed Altaf", "language": "hi", "calls_per_hour": 12, "can_split": True},
            {"name": "Sameer Pasha", "language": "hi", "calls_per_hour": 11, "can_split": True},
            {"name": "Hindi_Champ_7", "language": "hi", "calls_per_hour": 12, "can_split": True},
            {"name": "Hindi_Champ_8", "language": "hi", "calls_per_hour": 11, "can_split": False},
            
            # English (4 champions) - Reduced
            {"name": "Kavya S", "language": "en", "calls_per_hour": 15, "can_split": False},
            {"name": "Alwin", "language": "en", "calls_per_hour": 14, "can_split": True},
            {"name": "Pooja N", "language": "en", "calls_per_hour": 16, "can_split": True},
            {"name": "Navya", "language": "en", "calls_per_hour": 13, "can_split": False},
            
            # Telugu (5 champions) - Increased
            {"name": "Marcelina J", "language": "te", "calls_per_hour": 8, "can_split": False},
            {"name": "Sadanad", "language": "te", "calls_per_hour": 9, "can_split": True},
            {"name": "Jyothika", "language": "te", "calls_per_hour": 10, "can_split": True},
            {"name": "Vishal", "language": "te", "calls_per_hour": 8, "can_split": True},
            {"name": "Telugu_Champ_5", "language": "te", "calls_per_hour": 9, "can_split": True},
            
            # Tamil (2 champions) - Reduced
            {"name": "Binita Kongadi", "language": "ta", "calls_per_hour": 9, "can_split": True},
            {"name": "Dundesh", "language": "ta", "calls_per_hour": 8, "can_split": False},
            
            # Kannada (4 champions) - Maintained
            {"name": "Malikarjun Patil", "language": "ka", "calls_per_hour": 8, "can_split": False},
            {"name": "Rakshith", "language": "ka", "calls_per_hour": 9, "can_split": True},
            {"name": "M Showkath Nawaz", "language": "ka", "calls_per_hour": 10, "can_split": True},
            {"name": "Soubhikotl", "language": "ka", "calls_per_hour": 8, "can_split": True}
        ]
    
    def analyze_uploaded_data(self, uploaded_file):
        """Analyze the uploaded Excel file - FIXED for your format"""
        try:
            df = pd.read_excel(uploaded_file, sheet_name='Sheet2')
            
            hourly_volume = {}
            
            # Your Excel structure: Hours are in first column, Grand Total is last column
            for index, row in df.iterrows():
                first_cell = row.iloc[0]
                
                # Check if this row contains hour data (7, 8, 9, etc.)
                if pd.notna(first_cell):
                    try:
                        hour = float(first_cell)
                        if hour.is_integer() and 7 <= hour <= 21:
                            # Get the Grand Total value (usually last column)
                            grand_total = row.iloc[-1]
                            if pd.notna(grand_total):
                                hourly_volume[int(hour)] = float(grand_total)
                    except (ValueError, AttributeError):
                        continue
            
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
            import traceback
            st.write("Debug info:", traceback.format_exc())
            return None
    
    def plot_call_volume(self, analysis):
        """Create visualization of call volume"""
        if not analysis or 'hourly_volume' not in analysis or not analysis['hourly_volume']:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(title="No data available", height=400)
            return fig
        
        # Create a proper DataFrame for Plotly
        plot_data = []
        for hour, calls in analysis['hourly_volume'].items():
            plot_data.append({'Hour': hour, 'Calls': calls})
        
        df_plot = pd.DataFrame(plot_data)
        
        fig = px.bar(
            df_plot, 
            x='Hour', 
            y='Calls',
            labels={'Hour': 'Hour of Day', 'Calls': 'Number of Calls'},
            title='📊 Hourly Call Volume Distribution',
            color='Calls',
            color_continuous_scale='blues'
        )
        
        # Add peak hour annotations
        for peak_hour in analysis['peak_hours']:
            fig.add_vline(
                x=peak_hour, 
                line_dash="dash", 
                line_color="red", 
                annotation_text=f"Peak {peak_hour}:00", 
                annotation_position="top"
            )
        
        fig.update_layout(height=400)
        return fig

    def generate_roster(self, analysis):
        """Generate optimized roster with language focus"""
        try:
            # Generate shifts based on configuration
            shifts = self.generate_shifts()
            
            # Assign champions based on language priority
            roster = []
            language_count = {'hi': 0, 'en': 0, 'te': 0, 'ta': 0, 'ka': 0}
            target_counts = {'hi': 8, 'en': 4, 'te': 5, 'ta': 2, 'ka': 4}
            
            # Sort champions by language priority
            sorted_champs = sorted(self.champions, key=lambda x: (
                -target_counts[x['language']],  # Hindi/Telugu/Kannada first
                x['calls_per_hour']  # Higher capacity first
            ))
            
            for i, champ in enumerate(sorted_champs[:self.config['total_champions']]):
                lang = champ['language']
                if language_count[lang] < target_counts[lang]:
                    language_count[lang] += 1
                    
                    if i < self.config['straight_shifts']:
                        shift_type = "straight"
                        shift_time = f"{7 + (i % 5)}:00 - {16 + (i % 5)}:00"
                    else:
                        shift_type = "split"
                        shift_time = f"{7 + (i % 4)}:00-11:00 & 16:30-21:00"
                    
                    roster.append({
                        'Champion': champ['name'],
                        'Language': champ['language'].upper(),
                        'Shift Type': shift_type,
                        'Shift Time': shift_time,
                        'Calls/Hour Capacity': champ['calls_per_hour'],
                        'Can Work Split': 'Yes' if champ['can_split'] else 'No'
                    })
            
            return pd.DataFrame(roster)
            
        except Exception as e:
            st.error(f"Error generating roster: {str(e)}")
            import traceback
            st.write("Error details:", traceback.format_exc())
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
        
        st.subheader("🌐 Language Distribution")
        st.info("Focused on Hindi, Telugu & Kannada (High Volume)")
        for lang, percent in optimizer.config['language_distribution'].items():
            st.write(f"{lang.upper()}: {percent * 100}%")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📊 Upload Data")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'],
                                       help="Upload your 'Hourly Avg last 30 days.xlsx' file")
        
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
                        st.metric("Peak Hours", ", ".join([f"{h}:00" for h in analysis['peak_hours']]))
                    
                    # Store analysis in session state
                    st.session_state.analysis = analysis
                    
                    # Show data preview
                    with st.expander("📊 View Extracted Data"):
                        st.write("Hourly Volume:", analysis['hourly_volume'])
    
    with col2:
        st.header("📈 Call Volume Analysis")
        
        if 'analysis' in st.session_state:
            fig = optimizer.plot_call_volume(st.session_state.analysis)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📝 Upload your Excel file to see call volume analysis")
    
    # Generate Roster Button
    if st.button("🚀 Generate Optimized Roster", type="primary", use_container_width=True):
        if 'analysis' in st.session_state:
            with st.spinner("Generating optimized roster..."):
                roster_df = optimizer.generate_roster(st.session_state.analysis)
                
                if roster_df is not None:
                    st.header("📋 Generated Roster")
                    
                    # Display roster
                    st.dataframe(roster_df, use_container_width=True, height=400)
                    
                    # Show statistics
                    st.subheader("📊 Roster Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Champions", len(roster_df))
                    with col2:
                        st.metric("Straight Shifts", len(roster_df[roster_df['Shift Type'] == 'straight']))
                    with col3:
                        st.metric("Split Shifts", len(roster_df[roster_df['Shift Type'] == 'split']))
                    with col4:
                        avg_capacity = roster_df['Calls/Hour Capacity'].mean()
                        st.metric("Avg Capacity/Hour", f"{avg_capacity:.1f}")
                    
                    # Language distribution
                    st.subheader("🌐 Language Distribution")
                    lang_dist = roster_df['Language'].value_counts()
                    fig_lang = px.pie(
                        values=lang_dist.values,
                        names=lang_dist.index,
                        title="Champion Language Distribution",
                        hole=0.3
                    )
                    st.plotly_chart(fig_lang, use_container_width=True)
                    
                    # Download button
                    csv = roster_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Roster as CSV",
                        data=csv,
                        file_name="call_center_roster.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.warning("⚠️ Please upload and analyze data first!")

if __name__ == "__main__":
    main()

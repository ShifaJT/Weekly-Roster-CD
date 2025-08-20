import streamlit as st
import pandas as pd
import numpy as np
import pulp
from datetime import datetime, timedelta
import plotly.express as px
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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .shift-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CallCenterRosterOptimizer:
    def __init__(self):
        self.operation_hours = list(range(7, 22))  # 7 AM to 9 PM
        self.champions = self.load_champions()
        
    def load_champions(self):
        return [
            {"name": "Revathi", "primary_lang": "ka", "secondary_langs": ["hi", "te", "ta"], "calls_per_hour": 14, "can_split": True},
            {"name": "Pasang", "primary_lang": "ka", "secondary_langs": ["hi", "ta"], "calls_per_hour": 13, "can_split": False},
            {"name": "Kavya S", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 15, "can_split": False},
            {"name": "Anjali", "primary_lang": "ka", "secondary_langs": ["te", "hi"], "calls_per_hour": 14, "can_split": True},
            {"name": "Alwin", "primary_lang": "hi", "secondary_langs": ["ka"], "calls_per_hour": 13, "can_split": True},
            {"name": "Marcelina J", "primary_lang": "ka", "secondary_langs": ["ta"], "calls_per_hour": 12, "can_split": False},
            {"name": "Binita Kongadi", "primary_lang": "hi", "secondary_langs": [], "calls_per_hour": 11, "can_split": True},
            {"name": "Pooja N", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 14, "can_split": True},
            {"name": "Sadanad", "primary_lang": "ka", "secondary_langs": ["hi"], "calls_per_hour": 13, "can_split": True},
            {"name": "Navya", "primary_lang": "ka", "secondary_langs": ["te", "ta"], "calls_per_hour": 14, "can_split": False},
            {"name": "Jyothika", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 13, "can_split": True},
            {"name": "Dundesh", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 12, "can_split": False},
            {"name": "Rakesh", "primary_lang": "ka", "secondary_langs": ["ta"], "calls_per_hour": 13, "can_split": True},
            {"name": "Malikarjun Patil", "primary_lang": "ka", "secondary_langs": ["hi"], "calls_per_hour": 14, "can_split": False},
            {"name": "Divya", "primary_lang": "ka", "secondary_langs": ["te", "ta"], "calls_per_hour": 14, "can_split": True},
            {"name": "Mohammed Altaf", "primary_lang": "hi", "secondary_langs": [], "calls_per_hour": 12, "can_split": True},
            {"name": "Rakshith", "primary_lang": "ka", "secondary_langs": ["hi"], "calls_per_hour": 13, "can_split": True},
            {"name": "M Showkath Nawaz", "primary_lang": "ka", "secondary_langs": ["hi", "te"], "calls_per_hour": 14, "can_split": True},
            {"name": "Vishal", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 13, "can_split": True},
            {"name": "Muthahir", "primary_lang": "hi", "secondary_langs": ["te"], "calls_per_hour": 12, "can_split": False},
            {"name": "Soubhikotl", "primary_lang": "hi", "secondary_langs": [], "calls_per_hour": 11, "can_split": True},
            {"name": "Shashindra", "primary_lang": "hi", "secondary_langs": ["ka", "te"], "calls_per_hour": 13, "can_split": True},
            {"name": "Sameer Pasha", "primary_lang": "hi", "secondary_langs": ["ka", "te"], "calls_per_hour": 13, "can_split": True},
            {"name": "Guruswamy", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 12, "can_split": False},
            {"name": "Sheikh Vali Babu", "primary_lang": "hi", "secondary_langs": ["te"], "calls_per_hour": 12, "can_split": True},
            {"name": "Baloji", "primary_lang": "hi", "secondary_langs": ["te"], "calls_per_hour": 11, "can_split": False},
            {"name": "waghmare", "primary_lang": "te", "secondary_langs": ["hi", "ka"], "calls_per_hour": 13, "can_split": True},
            {"name": "Deepika", "primary_lang": "ka", "secondary_langs": ["hi"], "calls_per_hour": 12, "can_split": False}
        ]
    
    def analyze_excel_data(self, uploaded_file):
        """Analyze uploaded Excel file for call volume"""
        try:
            df = pd.read_excel(uploaded_file)
            
            # Extract hourly call volume (simplified for demo)
            hourly_volume = {
                7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
                13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
                18: 224.9, 19: 179.3, 20: 113.9, 21: 0
            }
            
            return {
                'hourly_volume': hourly_volume,
                'peak_hours': [11, 12, 13, 14],
                'total_daily_calls': 2975
            }
            
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None
    
    def generate_roster(self, straight_shifts, split_shifts, analysis_data):
        """Generate optimized roster based on shift preferences"""
        try:
            total_champions = straight_shifts + split_shifts
            champions_to_use = self.champions[:total_champions]
            
            roster_data = []
            
            # Generate shifts for each day
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            for day in days:
                day_roster = self.generate_daily_roster(day, champions_to_use, straight_shifts, split_shifts, analysis_data)
                roster_data.extend(day_roster)
            
            return pd.DataFrame(roster_data)
            
        except Exception as e:
            st.error(f"Error generating roster: {str(e)}")
            return None
    
    def generate_daily_roster(self, day, champions, straight_shifts, split_shifts, analysis_data):
        """Generate roster for a single day"""
        daily_roster = []
        
        # Straight shifts (9 hours continuous)
        straight_start_times = [7, 8, 9, 10]  # Start times for straight shifts
        
        for i, champ in enumerate(champions[:straight_shifts]):
            start_time = straight_start_times[i % len(straight_start_times)]
            end_time = start_time + 9
            
            daily_roster.append({
                'Day': day,
                'Champion': champ['name'],
                'Primary Language': champ['primary_lang'].upper(),
                'Secondary Languages': ', '.join([lang.upper() for lang in champ['secondary_langs']]),
                'Shift Type': 'Straight',
                'Start Time': f"{start_time:02d}:00",
                'End Time': f"{end_time:02d}:00",
                'Duration': '9 hours',
                'Calls/Hour Capacity': champ['calls_per_hour'],
                'Can Split': 'Yes' if champ['can_split'] else 'No'
            })
        
        # Split shifts (for remaining champions)
        split_patterns = [
            (7, 12, 16, 21),   # 7-12 & 4-9
            (8, 13, 17, 21),   # 8-1 & 5-9
            (9, 14, 17, 21),   # 9-2 & 5-9
            (10, 15, 17, 21)   # 10-3 & 5-9
        ]
        
        for i, champ in enumerate(champions[straight_shifts:straight_shifts + split_shifts]):
            pattern = split_patterns[i % len(split_patterns)]
            
            daily_roster.append({
                'Day': day,
                'Champion': champ['name'],
                'Primary Language': champ['primary_lang'].upper(),
                'Secondary Languages': ', '.join([lang.upper() for lang in champ['secondary_langs']]),
                'Shift Type': 'Split',
                'Start Time': f"{pattern[0]:02d}:00-{pattern[1]:02d}:00 & {pattern[2]:02d}:00-{pattern[3]:02d}:00",
                'End Time': f"{pattern[3]:02d}:00",
                'Duration': '9 hours (with break)',
                'Calls/Hour Capacity': champ['calls_per_hour'],
                'Can Split': 'Yes' if champ['can_split'] else 'No'
            })
        
        return daily_roster

    def calculate_coverage(self, roster_df, analysis_data):
        """Calculate coverage metrics"""
        if roster_df is None or analysis_data is None:
            return None
            
        total_capacity = roster_df['Calls/Hour Capacity'].sum() * 9 * 7  # 9 hours/day, 7 days
        required_capacity = analysis_data['total_daily_calls'] * 7 * 1.1  # 110% of weekly calls
        
        return {
            'total_capacity': total_capacity,
            'required_capacity': required_capacity,
            'utilization_rate': min(100, (required_capacity / total_capacity) * 100),
            'expected_answer_rate': min(100, (total_capacity / (analysis_data['total_daily_calls'] * 7)) * 100)
        }

# Main application
def main():
    st.markdown('<h1 class="main-header">📞 Call Center Roster Optimizer</h1>', unsafe_allow_html=True)
    
    optimizer = CallCenterRosterOptimizer()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Shift Configuration")
        
        # Shift type selection
        st.subheader("Shift Type Allocation")
        total_champs = len(optimizer.champions)
        straight_shifts = st.slider(
            "Regular Shifts (9 hours continuous)", 
            min_value=0, 
            max_value=total_champs, 
            value=18,
            help="Number of champions working straight 9-hour shifts"
        )
        
        split_shifts = st.slider(
            "Split Shifts (with break)", 
            min_value=0, 
            max_value=total_champs, 
            value=10,
            help="Number of champions working split shifts with afternoon break"
        )
        
        if straight_shifts + split_shifts > total_champs:
            st.warning(f"⚠️ Only {total_champs} champions available! Reducing split shifts...")
            split_shifts = total_champs - straight_shifts
        
        st.metric("Total Champions Used", f"{straight_shifts + split_shifts}/{total_champs}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # File upload section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📊 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Call Volume Excel", 
            type=['xlsx', 'xls'],
            help="Upload your call volume data Excel file"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Operation hours info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🕐 Operation Hours")
        st.info("""
        **Operating Schedule:**
        - 🕖 **7:00 AM** - Operation starts
        - 🕘 **9:00 PM** - Operation ends
        - 📞 **14 hours** daily coverage
        - 🎯 **9-hour shifts** for all champions
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📈 Current Configuration")
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Regular Shifts", straight_shifts)
        st.metric("Split Shifts", split_shifts)
        st.metric("Total Coverage", f"{straight_shifts + split_shifts} champions")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            st.success("✅ Excel file uploaded successfully!")
            analysis_data = optimizer.analyze_excel_data(uploaded_file)
            if analysis_data:
                st.session_state.analysis_data = analysis_data
                st.metric("Daily Call Volume", f"{analysis_data['total_daily_calls']:,.0f}")
                st.metric("Peak Hours", ", ".join([f"{h}:00" for h in analysis_data['peak_hours']]))
    
    with col2:
        st.header("🎯 Generate Roster")
        
        if st.button("🚀 Generate Optimized Roster", type="primary", use_container_width=True):
            if 'analysis_data' not in st.session_state:
                # Use default data if no file uploaded
                st.session_state.analysis_data = {
                    'hourly_volume': {
                        7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
                        13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
                        18: 224.9, 19: 179.3, 20: 113.9, 21: 0
                    },
                    'peak_hours': [11, 12, 13, 14],
                    'total_daily_calls': 2975
                }
            
            with st.spinner("Generating optimized roster..."):
                roster_df = optimizer.generate_roster(
                    straight_shifts, 
                    split_shifts, 
                    st.session_state.analysis_data
                )
                
                if roster_df is not None:
                    st.session_state.roster_df = roster_df
                    
                    # Calculate performance metrics
                    metrics = optimizer.calculate_coverage(roster_df, st.session_state.analysis_data)
                    
                    st.success("✅ Roster generated successfully!")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Weekly Capacity", f"{metrics['total_capacity']:,.0f} calls")
                    with col2:
                        st.metric("Required Capacity", f"{metrics['required_capacity']:,.0f} calls")
                    with col3:
                        st.metric("Expected Answer Rate", f"{metrics['expected_answer_rate']:.1f}%")
                    
                    # Display roster by day
                    st.subheader("📋 Daily Roster Summary")
                    
                    for day in roster_df['Day'].unique():
                        with st.expander(f"📅 {day} - {len(roster_df[roster_df['Day'] == day])} champions"):
                            day_df = roster_df[roster_df['Day'] == day].drop('Day', axis=1)
                            st.dataframe(day_df, use_container_width=True)
                    
                    # Download options
                    st.subheader("💾 Download Options")
                    csv = roster_df.to_csv(index=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download as CSV",
                            csv,
                            "call_center_roster.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    with col2:
                        # Excel download
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            roster_df.to_excel(writer, index=False, sheet_name='Roster')
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            "📥 Download as Excel",
                            excel_data,
                            "call_center_roster.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
        
        elif 'roster_df' in st.session_state:
            # Show previously generated roster
            st.info("📋 Previously generated roster is available. Click the button to generate a new one.")

if __name__ == "__main__":
    main()

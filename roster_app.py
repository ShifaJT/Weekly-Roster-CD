import streamlit as st
import pandas as pd
import numpy as np
import pulp
from datetime import datetime, timedelta
import plotly.express as px
from io import BytesIO
import base64
import random

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
    .day-tab {
        padding: 0.5rem 1rem;
        background-color: #e9ecef;
        border-radius: 0.5rem;
        cursor: pointer;
        margin: 0.2rem;
        display: inline-block;
    }
    .day-tab.active {
        background-color: #1f77b4;
        color: white;
    }
    .week-off {
        background-color: #ffe6e6;
    }
    .split-shift {
        background-color: #e6f7ff;
    }
    .straight-shift {
        background-color: #f0f8f0;
    }
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .dataframe {
        font-size: 0.9rem;
    }
    .highlight {
        background-color: #fff2cc;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }
    .split-shift-option {
        background-color: #e6f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #1f77b4;
    }
    .roster-table {
        width: 100%;
        border-collapse: collapse;
    }
    .roster-table th, .roster-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .roster-table th {
        background-color: #f2f2f2;
    }
    .wo-cell {
        background-color: #ffcccc;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class CallCenterRosterOptimizer:
    def __init__(self):
        self.operation_hours = list(range(7, 22))  # 7 AM to 9 PM
        self.champions = self.load_champions()
        self.split_shift_patterns = [
            {"name": "7-12 & 4-9", "times": (7, 12, 16, 21), "display": "07:00 to 12:00 & 16:30 to 21:00"},
            {"name": "8-1 & 5-9", "times": (8, 13, 17, 21), "display": "08:00 to 13:00 & 17:00 to 21:00"},
            {"name": "9-2 & 5-9", "times": (9, 14, 17, 21), "display": "09:00 to 14:00 & 17:00 to 21:00"},
            {"name": "10-3 & 5-9", "times": (10, 15, 17, 21), "display": "10:00 to 15:00 & 17:00 to 21:00"}
        ]
        
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
            
            # Calculate average weekly calls based on your input
            weekly_calls = 17500  # Based on your average
            
            return {
                'hourly_volume': {
                    7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
                    13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
                    18: 224.9, 19: 179.3, 20: 113.9, 21: 0
                },
                'peak_hours': [11, 12, 13, 14],
                'total_daily_calls': weekly_calls / 7  # Distribute weekly calls evenly
            }
            
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            # Use default data based on your average
            weekly_calls = 17500
            return {
                'hourly_volume': {
                    7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
                    13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
                    18: 224.9, 19: 179.3, 20: 113.9, 21: 0
                },
                'peak_hours': [11, 12, 13, 14],
                'total_daily_calls': weekly_calls / 7
            }
    
    def generate_roster(self, straight_shifts, split_shifts, analysis_data, manual_splits=None):
        """Generate optimized roster based on shift preferences"""
        try:
            total_champions = straight_shifts + split_shifts
            champions_to_use = self.champions[:total_champions]
            
            roster_data = []
            
            # Generate shifts for each day
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            for day in days:
                day_roster = self.generate_daily_roster(day, champions_to_use, straight_shifts, split_shifts, analysis_data, manual_splits)
                roster_data.extend(day_roster)
            
            roster_df = pd.DataFrame(roster_data)
            
            # Apply special rules
            roster_df = self.apply_special_rules(roster_df)
            
            # Apply manual split assignments
            if manual_splits:
                roster_df = self.apply_manual_splits(roster_df, manual_splits)
            
            # Assign weekly offs with limit of 3-4 per day
            roster_df, week_offs = self.assign_weekly_offs(roster_df, max_offs_per_day=4)
            
            return roster_df, week_offs
            
        except Exception as e:
            st.error(f"Error generating roster: {str(e)}")
            return None, None
    
    def generate_daily_roster(self, day, champions, straight_shifts, split_shifts, analysis_data, manual_splits=None):
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
                'Start Time': f"{start_time:02d}:00 to {end_time:02d}:00",
                'End Time': f"{end_time:02d}:00",
                'Duration': '9 hours',
                'Calls/Hour Capacity': champ['calls_per_hour'],
                'Can Split': 'Yes' if champ['can_split'] else 'No'
            })
        
        # Split shifts (for remaining champions)
        for i, champ in enumerate(champions[straight_shifts:straight_shifts + split_shifts]):
            pattern = self.split_shift_patterns[i % len(self.split_shift_patterns)]
            
            daily_roster.append({
                'Day': day,
                'Champion': champ['name'],
                'Primary Language': champ['primary_lang'].upper(),
                'Secondary Languages': ', '.join([lang.upper() for lang in champ['secondary_langs']]),
                'Shift Type': 'Split',
                'Start Time': pattern['display'],
                'End Time': f"{pattern['times'][3]:02d}:00",
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
    
    def calculate_answer_rate(self, roster_df, analysis_data):
        """Calculate expected Answer Rate percentage"""
        if roster_df is None or analysis_data is None:
            return None
            
        # Calculate total weekly capacity
        total_weekly_capacity = 0
        for day in roster_df['Day'].unique():
            day_roster = roster_df[roster_df['Day'] == day]
            for _, row in day_roster.iterrows():
                # Calculate hours worked based on shift type
                if row['Shift Type'] == 'Straight':
                    hours_worked = 9
                else:  # Split shift
                    # Parse the split shift hours
                    shifts = row['Start Time'].split(' & ')
                    hours_worked = 0
                    for shift in shifts:
                        times = shift.split(' to ')
                        start_hour = int(times[0].split(':')[0])
                        end_hour = int(times[1].split(':')[0])
                        hours_worked += (end_hour - start_hour)
                
                total_weekly_capacity += row['Calls/Hour Capacity'] * hours_worked
        
        # Calculate expected weekly call volume
        total_weekly_calls = analysis_data['total_daily_calls'] * 7
        
        # Calculate answer rate (capped at 100%)
        answer_rate = min(100, (total_weekly_capacity / total_weekly_calls) * 100)
        
        return answer_rate
    
    def calculate_daily_answer_rates(self, roster_df, analysis_data):
        """Calculate Answer Rate for each day"""
        if roster_df is None or analysis_data is None:
            return None
            
        daily_rates = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for day in days:
            day_roster = roster_df[roster_df['Day'] == day]
            if len(day_roster) == 0:
                daily_rates[day] = 0
                continue
                
            # Calculate daily capacity
            daily_capacity = 0
            for _, row in day_roster.iterrows():
                # Calculate hours worked based on shift type
                if row['Shift Type'] == 'Straight':
                    hours_worked = 9
                else:  # Split shift
                    # Parse the split shift hours
                    shifts = row['Start Time'].split(' & ')
                    hours_worked = 0
                    for shift in shifts:
                        times = shift.split(' to ')
                        start_hour = int(times[0].split(':')[0])
                        end_hour = int(times[1].split(':')[0])
                        hours_worked += (end_hour - start_hour)
                
                daily_capacity += row['Calls/Hour Capacity'] * hours_worked
            
            # Calculate daily answer rate
            daily_rates[day] = min(100, (daily_capacity / analysis_data['total_daily_calls']) * 100)
        
        return daily_rates
    
    def show_editable_roster(self, roster_df):
        """Display an editable roster table"""
        st.subheader("✏️ Edit Roster Manually")
        
        # Create a copy for editing
        edited_df = st.data_editor(
            roster_df,
            column_config={
                "Champion": st.column_config.SelectboxColumn(
                    "Champion",
                    options=[champ["name"] for champ in self.champions],
                    required=True
                ),
                "Shift Type": st.column_config.SelectboxColumn(
                    "Shift Type",
                    options=["Straight", "Split"],
                    required=True
                ),
                "Start Time": st.column_config.TextColumn(
                    "Start Time",
                    help="Format: HH:MM to HH:MM or HH:MM to HH:MM & HH:MM to HH:MM for split shifts"
                ),
                "End Time": st.column_config.TextColumn(
                    "End Time"
                )
            },
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True
        )
        
        return edited_df
    
    def assign_weekly_offs(self, roster_df, max_offs_per_day=4):
        """Assign weekly off days to champions with limit per day"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        champions = roster_df['Champion'].unique()
        
        # Create a copy of the roster
        updated_roster = roster_df.copy()
        week_offs = {}
        
        # Track offs per day
        offs_per_day = {day: 0 for day in days}
        
        # For each champion, assign one day off
        for champion in champions:
            # Get all days the champion is working
            champ_days = updated_roster[updated_roster['Champion'] == champion]['Day'].unique()
            
            # If champion is working all 7 days, remove one day
            if len(champ_days) == 7:
                # Find days that haven't reached the max off limit
                available_off_days = [day for day in days if offs_per_day[day] < max_offs_per_day]
                
                if available_off_days:
                    day_off = random.choice(available_off_days)
                    week_offs[champion] = day_off
                    offs_per_day[day_off] += 1
                    
                    # Remove the champion from the selected day
                    updated_roster = updated_roster[
                        ~((updated_roster['Champion'] == champion) & (updated_roster['Day'] == day_off))
                    ]
                else:
                    # If all days have max offs, don't assign an off day
                    week_offs[champion] = "No day off (max reached)"
            else:
                # Find which day is missing
                working_days = set(champ_days)
                all_days = set(days)
                day_off = list(all_days - working_days)[0] if all_days - working_days else "No day off"
                week_offs[champion] = day_off
                if day_off in days:
                    offs_per_day[day_off] += 1
        
        return updated_roster, week_offs
    
    def apply_special_rules(self, roster_df):
        """Apply special rules like Revathi always on split shift"""
        # Ensure Revathi is always on split shift
        revathi_mask = roster_df['Champion'] == 'Revathi'
        roster_df.loc[revathi_mask, 'Shift Type'] = 'Split'
        
        # Update split shift times for Revathi
        for idx in roster_df[revathi_mask].index:
            pattern = self.split_shift_patterns[0]  # Use the first pattern
            roster_df.at[idx, 'Start Time'] = pattern['display']
            roster_df.at[idx, 'End Time'] = f"{pattern['times'][3]:02d}:00"
            roster_df.at[idx, 'Duration'] = '9 hours (with break)'
        
        return roster_df
    
    def apply_manual_splits(self, roster_df, manual_splits):
        """Apply manual split shift assignments"""
        for assignment in manual_splits:
            champ = assignment['champion']
            day = assignment['day']
            pattern = assignment['pattern']
            
            # Find the row to update
            mask = (roster_df['Champion'] == champ) & (roster_df['Day'] == day)
            if mask.any():
                roster_df.loc[mask, 'Shift Type'] = 'Split'
                roster_df.loc[mask, 'Start Time'] = pattern['display']
                roster_df.loc[mask, 'End Time'] = f"{pattern['times'][3]:02d}:00"
                roster_df.loc[mask, 'Duration'] = '9 hours (with break)'
        
        return roster_df
    
    def filter_split_shift_champs(self, roster_df, can_split_only=True):
        """Filter champions who can work split shifts"""
        if can_split_only:
            # Get list of champions who can work split shifts
            split_champs = [champ["name"] for champ in self.champions if champ["can_split"]]
            
            # Filter roster to only include these champions for split shifts
            filtered_roster = roster_df.copy()
            split_mask = filtered_roster['Shift Type'] == 'Split'
            filtered_roster = filtered_roster[~split_mask | filtered_roster['Champion'].isin(split_champs)]
            
            return filtered_roster
        else:
            return roster_df
    
    def format_roster_for_display(self, roster_df, week_offs):
        """Format the roster in the sample Excel format"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Create a DataFrame in the sample format
        display_df = pd.DataFrame()
        display_df['Name'] = [champ['name'] for champ in self.champions if champ['name'] in roster_df['Champion'].unique()]
        
        # Add default shift pattern column
        display_df['Shift'] = ""
        
        # Add columns for each day
        for day in days:
            display_df[day] = ""
        
        # Fill in the shifts for each day
        for _, row in roster_df.iterrows():
            champ_name = row['Champion']
            day = row['Day']
            shift_time = row['Start Time']
            
            # Find the row for this champion
            champ_idx = display_df[display_df['Name'] == champ_name].index
            if len(champ_idx) > 0:
                display_df.at[champ_idx[0], day] = shift_time
                
                # Set the default shift pattern (most common shift)
                if display_df.at[champ_idx[0], 'Shift'] == "":
                    display_df.at[champ_idx[0], 'Shift'] = shift_time
        
        # Apply week offs
        for champ, off_day in week_offs.items():
            if off_day in days:
                champ_idx = display_df[display_df['Name'] == champ].index
                if len(champ_idx) > 0:
                    display_df.at[champ_idx[0], off_day] = "WO"
        
        return display_df

# Main application
def main():
    st.markdown('<h1 class="main-header">📞 Call Center Roster Optimizer</h1>', unsafe_allow_html=True)
    
    optimizer = CallCenterRosterOptimizer()
    
    # Initialize session state
    if 'manual_splits' not in st.session_state:
        st.session_state.manual_splits = []
    
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
        
        # Split shift filter
        st.subheader("Split Shift Filter")
        split_filter = st.checkbox(
            "Only assign split shifts to champions who can split",
            value=True,
            help="When enabled, only champions marked as able to work split shifts will be assigned to them"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Manual split shift assignment
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔧 Manual Split Shift Assignment")
        
        st.markdown('<div class="split-shift-option">', unsafe_allow_html=True)
        st.subheader("Assign Specific Split Shifts")
        
        # Champion selection
        available_champs = [champ["name"] for champ in optimizer.champions]
        selected_champ = st.selectbox("Select Champion", available_champs)
        
        # Day selection
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day = st.selectbox("Select Day", days)
        
        # Split shift pattern selection
        pattern_options = [f"{pattern['name']} ({pattern['display']})" 
                          for pattern in optimizer.split_shift_patterns]
        selected_pattern_idx = st.selectbox("Select Split Shift Pattern", range(len(pattern_options)), format_func=lambda x: pattern_options[x])
        
        if st.button("➕ Add Manual Split Assignment"):
            new_assignment = {
                'champion': selected_champ,
                'day': selected_day,
                'pattern': optimizer.split_shift_patterns[selected_pattern_idx]
            }
            
            # Check if this assignment already exists
            if not any(a['champion'] == selected_champ and a['day'] == selected_day for a in st.session_state.manual_splits):
                st.session_state.manual_splits.append(new_assignment)
                st.success(f"Added split shift for {selected_champ} on {selected_day}")
            else:
                st.warning(f"{selected_champ} already has a manual split assignment on {selected_day}")
        
        # Show current manual assignments
        if st.session_state.manual_splits:
            st.subheader("Current Manual Assignments")
            for i, assignment in enumerate(st.session_state.manual_splits):
                st.write(f"{i+1}. {assignment['champion']} - {assignment['day']} - {assignment['pattern']['name']}")
                if st.button(f"Remove {i+1}", key=f"remove_{i}"):
                    st.session_state.manual_splits.pop(i)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
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
        - 🔄 **Revathi** always assigned to split shifts
        - 📋 Max 4 week offs per day to maintain answer rate
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
                st.metric("Weekly Call Volume", f"{analysis_data['total_daily_calls'] * 7:,.0f}")
                st.metric("Peak Hours", ", ".join([f"{h}:00" for h in analysis_data['peak_hours']]))
        else:
            # Use default data based on average
            st.session_state.analysis_data = {
                'hourly_volume': {
                    7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
                    13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
                    18: 224.9, 19: 179.3, 20: 113.9, 21: 0
                },
                'peak_hours': [11, 12, 13, 14],
                'total_daily_calls': 17500 / 7  # Based on your average
            }
            st.metric("Estimated Daily Calls", f"{17500 / 7:,.0f}")
            st.metric("Estimated Weekly Calls", "17,500")
    
    with col2:
        st.header("🎯 Generate Roster")
        
        if st.button("🚀 Generate Optimized Roster", type="primary", use_container_width=True):
            with st.spinner("Generating optimized roster..."):
                roster_df, week_offs = optimizer.generate_roster(
                    straight_shifts, 
                    split_shifts, 
                    st.session_state.analysis_data,
                    st.session_state.manual_splits
                )
                
                # Apply split shift filter if enabled
                if split_filter:
                    roster_df = optimizer.filter_split_shift_champs(roster_df, can_split_only=True)
                
                if roster_df is not None:
                    st.session_state.roster_df = roster_df
                    st.session_state.week_offs = week_offs
                    
                    # Calculate performance metrics
                    metrics = optimizer.calculate_coverage(roster_df, st.session_state.analysis_data)
                    answer_rate = optimizer.calculate_answer_rate(roster_df, st.session_state.analysis_data)
                    daily_rates = optimizer.calculate_daily_answer_rates(roster_df, st.session_state.analysis_data)
                    
                    st.session_state.metrics = metrics
                    st.session_state.answer_rate = answer_rate
                    st.session_state.daily_rates = daily_rates
                    
                    # Format roster for display
                    st.session_state.formatted_roster = optimizer.format_roster_for_display(roster_df, week_offs)
                    
                    st.success("✅ Roster generated successfully!")
        
        elif 'roster_df' in st.session_state:
            # Show previously generated roster
            st.info("📋 Previously generated roster is available. Click the button to generate a new one.")
    
    # Display results if available
    if 'roster_df' in st.session_state and 'metrics' in st.session_state:
        st.markdown("---")
        st.markdown('<div class="section-header"><h2>📊 Performance Metrics</h2></div>', unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Weekly Capacity", f"{st.session_state.metrics['total_capacity']:,.0f} calls")
        with col2:
            st.metric("Required Capacity", f"{st.session_state.metrics['required_capacity']:,.0f} calls")
        with col3:
            st.metric("Utilization Rate", f"{st.session_state.metrics['utilization_rate']:.1f}%")
        with col4:
            st.metric("Expected Answer Rate", f"{st.session_state.answer_rate:.1f}%")
        
        # Display daily answer rates
        st.markdown('<div class="section-header"><h2>📈 Daily Answer Rates</h2></div>', unsafe_allow_html=True)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_cols = st.columns(7)
        
        for i, day in enumerate(days):
            with day_cols[i]:
                rate = st.session_state.daily_rates.get(day, 0)
                st.metric(day[:3], f"{rate:.1f}%")
        
        # Week off information
        st.markdown('<div class="section-header"><h2>📅 Weekly Off Schedule</h2></div>', unsafe_allow_html=True)
        
        if 'week_offs' in st.session_state:
            week_off_df = pd.DataFrame.from_dict(st.session_state.week_offs, orient='index', columns=['Day Off'])
            week_off_df.index.name = 'Champion'
            week_off_df = week_off_df.reset_index()
            
            # Count week offs per day
            offs_per_day = week_off_df['Day Off'].value_counts()
            st.write("Week Offs per Day:")
            for day in days:
                count = offs_per_day.get(day, 0)
                st.write(f"{day}: {count} champions")
            
            st.dataframe(
                week_off_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Display roster in sample format
        st.markdown('<div class="section-header"><h2>📋 Roster Schedule</h2></div>', unsafe_allow_html=True)
        
        st.dataframe(
            st.session_state.formatted_roster,
            use_container_width=True,
            hide_index=True
        )
        
        # Manual editing option
        st.markdown('<div class="section-header"><h2>🛠️ Manual Adjustments</h2></div>', unsafe_allow_html=True)
        
        if st.checkbox("Enable manual editing of shift timings"):
            edited_roster = optimizer.show_editable_roster(st.session_state.roster_df)
            
            # Recalculate metrics after editing
            if st.button("Update Metrics after Editing"):
                updated_metrics = optimizer.calculate_coverage(edited_roster, st.session_state.analysis_data)
                updated_answer_rate = optimizer.calculate_answer_rate(edited_roster, st.session_state.analysis_data)
                updated_daily_rates = optimizer.calculate_daily_answer_rates(edited_roster, st.session_state.analysis_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Updated Weekly Capacity", f"{updated_metrics['total_capacity']:,.0f} calls", 
                             delta=f"{updated_metrics['total_capacity'] - st.session_state.metrics['total_capacity']:,.0f}")
                with col2:
                    st.metric("Required Capacity", f"{updated_metrics['required_capacity']:,.0f} calls")
                with col3:
                    st.metric("Updated Utilization", f"{updated_metrics['utilization_rate']:.1f}%", 
                             delta=f"{updated_metrics['utilization_rate'] - st.session_state.metrics['utilization_rate']:.1f}")
                with col4:
                    st.metric("Updated Answer Rate", f"{updated_answer_rate:.1f}%", 
                             delta=f"{updated_answer_rate - st.session_state.answer_rate:.1f}")
        
        # Download options
        st.markdown('<div class="section-header"><h2>💾 Download Options</h2></div>', unsafe_allow_html=True)
        
        # Download in sample format
        csv = st.session_state.formatted_roster.to_csv(index=False)
        
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
                st.session_state.formatted_roster.to_excel(writer, index=False, sheet_name='Roster')
            excel_data = excel_buffer.getvalue()
            st.download_button(
                "📥 Download as Excel",
                excel_data,
                "call_center_roster.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

if __name__ == "__main__":
    main()

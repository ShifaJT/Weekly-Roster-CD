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
    .template-download {
        background-color: #e6f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 2px dashed #1f77b4;
    }
    .al-critical {
        background-color: #ffcccc;
        color: #d93025;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .al-warning {
        background-color: #fff2cc;
        color: #f9ab00;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .al-good {
        background-color: #e6f4ea;
        color: #137333;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
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
        self.AVERAGE_HANDLING_TIME_SECONDS = 202  # 3 minutes 22 seconds
        self.TARGET_AL = 90  # Your Goal Answer Level
        
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
    
    # ================================
    # AL PREDICTION FUNCTIONS
    # ================================
    def calculate_hourly_capacity(self, num_agents, aht_seconds):
        """Calculates how many calls a team can handle in one hour."""
        hourly_capacity = (num_agents * 3600) / aht_seconds
        return round(hourly_capacity)

    def predict_al(self, forecasted_calls, num_agents, aht_seconds):
        """Predicts the Answer Level (AL) for a given hour."""
        capacity = self.calculate_hourly_capacity(num_agents, aht_seconds)
        # In reality, you can't answer more calls than are offered
        answered_calls = min(capacity, forecasted_calls)
        predicted_al = (answered_calls / forecasted_calls) * 100
        return round(predicted_al, 1), int(capacity)

    def agents_needed_for_target(self, forecasted_calls, target_al, aht_seconds):
        """Calculates the minimum number of agents required to hit a target AL."""
        # Calculate the required capacity to meet the target
        required_capacity = (target_al / 100) * forecasted_calls
        # Calculate the agents needed to achieve that capacity
        agents_required = (required_capacity * aht_seconds) / 3600
        # Always round UP because you can't have a fraction of an agent
        return np.ceil(agents_required)

    def analyze_roster_sufficiency(self, forecasted_calls, scheduled_agents, aht_seconds, target_al):
        """Analyzes if the scheduled roster is sufficient and recommends changes."""
        predicted_al, capacity = self.predict_al(forecasted_calls, scheduled_agents, aht_seconds)
        min_agents_required = self.agents_needed_for_target(forecasted_calls, target_al, aht_seconds)
        
        recommendation = {
            'forecasted_calls': forecasted_calls,
            'scheduled_agents': scheduled_agents,
            'current_capacity': capacity,
            'predicted_al': predicted_al,
            'min_agents_required': min_agents_required,
            'agents_deficit': max(0, min_agents_required - scheduled_agents),
            'status': '',
            'recommendation': ''
        }
        
        # Determine status and recommendation
        if predicted_al >= target_al:
            recommendation['status'] = '✅ GREEN (Goal Met)'
            recommendation['recommendation'] = 'Roster is sufficient to target.'
            # Check for overstaffing
            if scheduled_agents > min_agents_required + 1: # Buffer to avoid flip-flopping
                recommendation['recommendation'] = f'Roster is strong. Potential to reduce by {int(scheduled_agents - min_agents_required)} agent(s).'
        elif predicted_al >= 80:
            recommendation['status'] = '🟡 YELLOW (Close to Goal)'
            recommendation['recommendation'] = f'Close to target. Adding 1 agent could help secure {target_al}%.'
        elif predicted_al >= 70:
            recommendation['status'] = '🟠 ORANGE (At Risk)'
            recommendation['recommendation'] = f'Add {int(recommendation["agents_deficit"])} agent(s) to reach {target_al}%.'
        else:
            recommendation['status'] = '🔴 RED (Critical)'
            recommendation['recommendation'] = f'Immediately add {int(recommendation["agents_deficit"])} agent(s) to reach {target_al}%.'
        
        return recommendation

    def generate_what_if_analysis(self, forecasted_calls, base_agents, aht_seconds, target_al):
        """Generates a table showing the impact of adding/removing agents."""
        scenarios = []
        # Test from -3 to +5 agents around the current schedule
        for agent_change in range(-3, 6):
            agent_count = base_agents + agent_change
            if agent_count < 1: continue  # Skip impossible scenarios
            
            al, cap = self.predict_al(forecasted_calls, agent_count, aht_seconds)
            scenarios.append({
                'Agent Change': f'{agent_change:+d}',
                'Agents Scheduled': agent_count,
                'Capacity': cap,
                'Predicted AL': f'{al}%',
                'Status': '✅ Meet Goal' if al >= target_al else '⚠️ Below Goal'
            })
        
        return pd.DataFrame(scenarios)
    
    def calculate_hourly_al_analysis(self, roster_df, analysis_data):
        """Calculate AL predictions for each hour of each day"""
        if roster_df is None or analysis_data is None:
            return None

        # We'll create a structure for each day and each hour
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hourly_al_results = {}

        for day in days:
            # Filter roster for just this day
            daily_roster = roster_df[roster_df['Day'] == day]
            
            for hour in self.operation_hours:
                if hour not in analysis_data['hourly_volume']:
                    continue
                    
                # Count agents working at this hour ON THIS DAY
                agents_at_hour = 0
                for _, row in daily_roster.iterrows():
                    if self.is_agent_working_at_hour(row, hour):
                        agents_at_hour += 1
                
                # Predict AL for this hour
                forecasted_calls = analysis_data['hourly_volume'].get(hour, 0)
                if forecasted_calls > 0:
                    predicted_al, capacity = self.predict_al(forecasted_calls, agents_at_hour, self.AVERAGE_HANDLING_TIME_SECONDS)
                    
                    # Store results with day and hour as key
                    key = f"{day}_{hour}"
                    hourly_al_results[key] = {
                        'day': day,
                        'hour': hour,
                        'agents': agents_at_hour,
                        'forecast': forecasted_calls,
                        'capacity': capacity,
                        'predicted_al': predicted_al,
                        'status': self.get_al_status(predicted_al)
                    }
        
        return hourly_al_results
    
    def is_agent_working_at_hour(self, row, hour):
        """Check if an agent is working at a specific hour based on their shift"""
        try:
            if row['Shift Type'] == 'Straight':
                # Parse straight shift times
                times = row['Start Time'].split(' to ')
                start_hour = int(times[0].split(':')[0])
                end_hour = int(times[1].split(':')[0])
                return start_hour <= hour < end_hour
            else:
                # Parse split shift times
                shifts = row['Start Time'].split(' & ')
                for shift in shifts:
                    times = shift.split(' to ')
                    start_hour = int(times[0].split(':')[0])
                    end_hour = int(times[1].split(':')[0])
                    if start_hour <= hour < end_hour:
                        return True
                return False
        except:
            return False
    
    def get_al_status(self, al_value):
        """Get status classification for AL value"""
        if al_value >= self.TARGET_AL:
            return "✅ GOOD"
        elif al_value >= 80:
            return "🟡 WARNING"
        elif al_value >= 70:
            return "🟠 AT RISK"
        else:
            return "🔴 CRITICAL"
    
    def create_template_file(self):
        """Create a template Excel file for call volume data"""
        # Create sample data for the last 30 days
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
        
        # Create hourly data template
        hourly_data = pd.DataFrame({
            'Hour': list(range(7, 22)),
            'Calls': [0] * 15
        })
        
        # Create daily data template
        daily_data = pd.DataFrame({
            'Date': dates,
            'Total_Calls': [0] * 30,
            'Peak_Hour': [0] * 30,
            'Peak_Volume': [0] * 30
        })
        
        # Create instructions sheet
        instructions = pd.DataFrame({
            'Instruction': [
                'INSTRUCTIONS:',
                '1. Fill in your call volume data in the appropriate sheets',
                '2. For best results, use the Hourly_Data sheet with calls per hour',
                '3. If you only have daily totals, use the Daily_Data sheet',
                '4. Save the file and upload it back to the app',
                '5. The app will analyze your data and generate an optimized roster',
                '',
                'HOURLY_DATA SHEET:',
                '- Hour: Operation hour (7 to 21)',
                '- Calls: Number of calls received in that hour',
                '',
                'DAILY_DATA SHEET:',
                '- Date: Date of the data (YYYY-MM-DD)',
                '- Total_Calls: Total calls received that day',
                '- Peak_Hour: Hour with highest call volume (7-21)',
                '- Peak_Volume: Number of calls during peak hour'
            ]
        })
        
        # Create Excel file with multiple sheets
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            instructions.to_excel(writer, sheet_name='Instructions', index=False)
            hourly_data.to_excel(writer, sheet_name='Hourly_Data', index=False)
            daily_data.to_excel(writer, sheet_name='Daily_Data', index=False)
        
        return excel_buffer.getvalue()
    
    def analyze_excel_data(self, uploaded_file):
        """Analyze uploaded Excel file for call volume"""
        try:
            # Read the Excel file
            xls = pd.ExcelFile(uploaded_file)
            
            # Check if hourly data is available
            if 'Hourly_Data' in xls.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name='Hourly_Data')
                
                if 'Hour' in df.columns and 'Calls' in df.columns:
                    # Calculate average hourly volume
                    hourly_volume = dict(zip(df['Hour'], df['Calls']))
                    
                    # Find peak hours (top 4 hours with highest call volume)
                    peak_hours = df.nlargest(4, 'Calls')['Hour'].tolist()
                    
                    # Calculate total daily calls
                    total_daily_calls = df['Calls'].sum()
                    
                    return {
                        'hourly_volume': hourly_volume,
                        'peak_hours': peak_hours,
                        'total_daily_calls': total_daily_calls
                    }
            
            # Check if daily data is available
            if 'Daily_Data' in xls.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name='Daily_Data')
                
                if 'Total_Calls' in df.columns:
                    # Calculate average daily calls
                    avg_daily_calls = df['Total_Calls'].mean()
                    
                    # Use default hourly distribution based on typical patterns
                    hourly_volume = {
                        7: 0.02 * avg_daily_calls, 8: 0.05 * avg_daily_calls, 9: 0.08 * avg_daily_calls,
                        10: 0.10 * avg_daily_calls, 11: 0.12 * avg_daily_calls, 12: 0.11 * avg_daily_calls,
                        13: 0.10 * avg_daily_calls, 14: 0.09 * avg_daily_calls, 15: 0.08 * avg_daily_calls,
                        16: 0.08 * avg_daily_calls, 17: 0.07 * avg_daily_calls, 18: 0.06 * avg_daily_calls,
                        19: 0.05 * avg_daily_calls, 20: 0.03 * avg_daily_calls, 21: 0.01 * avg_daily_calls
                    }
                    
                    # Find peak hours from data if available, otherwise use defaults
                    if 'Peak_Hour' in df.columns and 'Peak_Volume' in df.columns:
                        avg_peak_hour = df['Peak_Hour'].mode()[0] if not df['Peak_Hour'].mode().empty else 11
                        peak_hours = [avg_peak_hour - 1, avg_peak_hour, avg_peak_hour + 1, avg_peak_hour + 2]
                    else:
                        peak_hours = [11, 12, 13, 14]
                    
                    return {
                        'hourly_volume': hourly_volume,
                        'peak_hours': peak_hours,
                        'total_daily_calls': avg_daily_calls
                    }
            
            # If no recognized format, use default data
            st.warning("No recognized data format. Using sample data.")
            return self.get_sample_data()
            
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return self.get_sample_data()
    
    def get_sample_data(self):
        """Return sample data structure"""
        return {
            'hourly_volume': {
                7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
                13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
                18: 224.9, 19: 179.3, 20: 113.9, 21: 0
            },
            'peak_hours': [11, 12, 13, 14],
            'total_daily_calls': 2975
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
        
        # Template download section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📊 Data Template")
        
        st.markdown('<div class="template-download">', unsafe_allow_html=True)
        st.subheader("Download Data Template")
        st.write("Download this template, add your call volume data, and upload it back.")
        
        # Create template file
        template_data = optimizer.create_template_file()
        
        st.download_button(
            "📥 Download Data Template",
            template_data,
            "call_volume_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # File upload section
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload Your Call Volume Data", 
            type=['xlsx', 'xls'],
            help="Upload your filled-in call volume data Excel file"
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
        - ⏱️ **AHT:** 3 minutes 22 seconds (202 seconds)
        - 🎯 **AL Target:** 90% Answer Level
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
            st.success("✅ Data file uploaded successfully!")
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
                    
                    # Calculate AL predictions
                    hourly_al_results = optimizer.calculate_hourly_al_analysis(roster_df, st.session_state.analysis_data)
                    
                    st.session_state.metrics = metrics
                    st.session_state.answer_rate = answer_rate
                    st.session_state.daily_rates = daily_rates
                    st.session_state.hourly_al_results = hourly_al_results
                    
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
        
        # Display AL prediction metrics
        st.markdown('<div class="section-header"><h2>🎯 AL Prediction Analysis</h2></div>', unsafe_allow_html=True)
        
        if 'hourly_al_results' in st.session_state and st.session_state.hourly_al_results:
            # Create a summary of AL predictions
            al_values = [hour_data['predicted_al'] for hour_data in st.session_state.hourly_al_results.values()]
            avg_al = sum(al_values) / len(al_values) if al_values else 0
            
            # Count hours by status
            status_counts = {
                "✅ GOOD": 0,
                "🟡 WARNING": 0,
                "🟠 AT RISK": 0,
                "🔴 CRITICAL": 0
            }
            
            for hour_data in st.session_state.hourly_al_results.values():
                status = hour_data['status']
                if status in status_counts:
                    status_counts[status] += 1
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Predicted AL", f"{avg_al:.1f}%")
            with col2:
                st.metric("Hours at Target", f"{status_counts['✅ GOOD']}")
            with col3:
                st.metric("Hours Needing Attention", f"{status_counts['🟡 WARNING'] + status_counts['🟠 AT RISK']}")
            with col4:
                st.metric("Critical Hours", f"{status_counts['🔴 CRITICAL']}")
            
            # Show detailed hourly AL predictions by day
            st.subheader("Hourly AL Predictions by Day")
            
            # Let user select a day to view
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            selected_day = st.selectbox("Select day to view hourly AL predictions", days)
            
            # Filter results for the selected day
            al_data = []
            for key, data in st.session_state.hourly_al_results.items():
                if data['day'] == selected_day:
                    al_data.append({
                        'Hour': f"{data['hour']:02d}:00",
                        'Agents': data['agents'],
                        'Forecasted Calls': int(data['forecast']),
                        'Capacity': int(data['capacity']),
                        'Predicted AL': f"{data['predicted_al']}%",
                        'Status': data['status']
                    })
            
            # Sort by hour
            al_data.sort(key=lambda x: x['Hour'])
            
            if al_data:
                al_df = pd.DataFrame(al_data)
                st.dataframe(al_df, use_container_width=True, hide_index=True)
                
                # Identify critical hours that need extra champions
                critical_hours = []
                for data in al_data:
                    if float(data['Predicted AL'].replace('%', '')) < optimizer.TARGET_AL:
                        forecast = next((item['forecast'] for item in al_data if item['Hour'] == data['Hour']), 0)
                        agents_needed = optimizer.agents_needed_for_target(
                            forecast, optimizer.TARGET_AL, optimizer.AVERAGE_HANDLING_TIME_SECONDS
                        )
                        extra_agents = max(0, agents_needed - data['Agents'])
                        if extra_agents > 0:
                            critical_hours.append({
                                'Hour': data['Hour'],
                                'Current Agents': data['Agents'],
                                'Agents Needed': int(agents_needed),
                                'Extra Agents Needed': int(extra_agents),
                                'Current AL': data['Predicted AL']
                            })
                
                if critical_hours:
                    st.subheader("⚠️ Hours Needing Extra Champions")
                    critical_df = pd.DataFrame(critical_hours)
                    st.dataframe(critical_df, use_container_width=True, hide_index=True)
            else:
                st.info("No AL prediction data available for the selected day.")
        
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
                updated_hourly_al = optimizer.calculate_hourly_al_analysis(edited_roster, st.session_state.analysis_data)
                
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

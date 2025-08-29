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
        
        # Custom shift patterns based on your requirements
        self.split_shift_patterns = [
            {"name": "7-12 & 4-9", "times": (7, 12, 16, 21), "display": "07:00 to 12:00 & 16:30 to 21:00"},
            {"name": "8-1 & 5-9", "times": (8, 13, 17, 21), "display": "08:00 to 13:00 & 17:00 to 21:00"},
            {"name": "10-3 & 5-9", "times": (10, 15, 17, 21), "display": "10:00 to 15:00 & 17:00 to 21:00"},
            {"name": "12-9", "times": (12, 21), "display": "12:00 to 21:00"},
            {"name": "11-8", "times": (11, 20), "display": "11:00 to 20:00"},
        ]
        
        self.AVERAGE_HANDLING_TIME_SECONDS = 202  # 3 minutes 22 seconds
        self.TARGET_AL = 80  # Target Answer Level
        
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
            {"name": "Baloji", "primary_lang": "", "secondary_langs": ["te"], "calls_per_hour": 11, "can_split": False},
            {"name": "waghmare", "primary_lang": "te", "secondary_langs": ["hi", "ka"], "calls_per_hour": 13, "can_split": True},
            {"name": "Deepika", "primary_lang": "ka", "secondary_langs": ["hi"], "calls_per_hour": 12, "can_split": False}
        ]
    
    def calculate_hourly_capacity(self, num_agents, aht_seconds):
        """Calculates how many calls a team can handle in one hour."""
        hourly_capacity = (num_agents * 3600) / aht_seconds
        return round(hourly_capacity)

    def predict_al(self, forecasted_calls, num_agents, aht_seconds):
        """Predicts the Answer Level (AL) for a given hour."""
        capacity = self.calculate_hourly_capacity(num_agents, aht_seconds)
        answered_calls = min(capacity, forecasted_calls)
        if forecasted_calls > 0:
            predicted_al = (answered_calls / forecasted_calls) * 100
        else:
            predicted_al = 100
        return round(predicted_al, 1), int(capacity)

    def agents_needed_for_target(self, forecasted_calls, target_al, aht_seconds):
        """Calculates the minimum number of agents required to hit a target AL."""
        required_capacity = (target_al / 100) * forecasted_calls
        agents_required = (required_capacity * aht_seconds) / 3600
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
        
        if predicted_al >= target_al:
            recommendation['status'] = '✅ GREEN (Goal Met)'
            recommendation['recommendation'] = 'Roster is sufficient to target.'
            if scheduled_agents > min_agents_required + 1:
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
        for agent_change in range(-3, 6):
            agent_count = base_agents + agent_change
            if agent_count < 1: continue
            
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

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hourly_al_results = {}

        for day in days:
            daily_roster = roster_df[roster_df['Day'] == day]
            
            for hour in self.operation_hours:
                if hour not in analysis_data['hourly_volume']:
                    continue
                    
                agents_at_hour = 0
                for _, row in daily_roster.iterrows():
                    if self.is_agent_working_at_hour(row, hour):
                        agents_at_hour += 1
                
                forecasted_calls = analysis_data['hourly_volume'].get(hour, 0)
                if forecasted_calls > 0:
                    predicted_al, capacity = self.predict_al(forecasted_calls, agents_at_hour, self.AVERAGE_HANDLING_TIME_SECONDS)
                    
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
                times = row['Start Time'].split(' to ')
                start_hour = int(times[0].split(':')[0])
                end_hour = int(times[1].split(':')[0])
                return start_hour <= hour < end_hour
            else:
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
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
        
        hourly_data = pd.DataFrame({
            'Hour': list(range(7, 22)),
            'Calls': [0] * 15
        })
        
        daily_data = pd.DataFrame({
            'Date': dates,
            'Total_Calls': [0] * 30,
            'Peak_Hour': [0] * 30,
            'Peak_Volume': [0] * 30
        })
        
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
        
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            instructions.to_excel(writer, sheet_name='Instructions', index=False)
            hourly_data.to_excel(writer, sheet_name='Hourly_Data', index=False)
            daily_data.to_excel(writer, sheet_name='Daily_Data', index=False)
        
        return excel_buffer.getvalue()
    
    def analyze_excel_data(self, uploaded_file):
        """Analyze uploaded Excel file for call volume"""
        try:
            xls = pd.ExcelFile(uploaded_file)
            
            if 'Hourly_Data' in xls.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name='Hourly_Data')
                
                if 'Hour' in df.columns and 'Calls' in df.columns:
                    hourly_volume = dict(zip(df['Hour'], df['Calls']))
                    peak_hours = df.nlargest(4, 'Calls')['Hour'].tolist()
                    total_daily_calls = df['Calls'].sum()
                    
                    return {
                        'hourly_volume': hourly_volume,
                        'peak_hours': peak_hours,
                        'total_daily_calls': total_daily_calls
                    }
            
            if 'Daily_Data' in xls.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name='Daily_Data')
                
                if 'Total_Calls' in df.columns:
                    avg_daily_calls = df['Total_Calls'].mean()
                    
                    hourly_volume = {
                        7: 0.02 * avg_daily_calls, 8: 0.05 * avg_daily_calls, 9: 0.08 * avg_daily_calls,
                        10: 0.10 * avg_daily_calls, 11: 0.12 * avg_daily_calls, 12: 0.11 * avg_daily_calls,
                        13: 0.10 * avg_daily_calls, 14: 0.09 * avg_daily_calls, 15: 0.08 * avg_daily_calls,
                        16: 0.08 * avg_daily_calls, 17: 0.07 * avg_daily_calls, 18: 0.06 * avg_daily_calls,
                        19: 0.05 * avg_daily_calls, 20: 0.03 * avg_daily_calls, 21: 0.01 * avg_daily_calls
                    }
                    
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
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            for day in days:
                day_roster = self.generate_daily_roster(day, champions_to_use, straight_shifts, split_shifts, analysis_data, manual_splits)
                roster_data.extend(day_roster)
            
            roster_df = pd.DataFrame(roster_data)
            roster_df = self.apply_special_rules(roster_df)
            
            if manual_splits:
                roster_df = self.apply_manual_splits(roster_df, manual_splits)
            
            active_split_champs = getattr(st.session_state, 'active_split_champs', 4)
            roster_df, week_offs = self.assign_weekly_offs(roster_df, max_offs_per_day=4, min_split_champs=active_split_champs)
            
            return roster_df, week_offs
            
        except Exception as e:
            st.error(f"Error generating roster: {str(e)}")
            return None, None
    
    def generate_daily_roster(self, day, champions, straight_shifts, split_shifts, analysis_data, manual_splits=None):
        """Generate roster for a single day with specific shift requirements"""
        daily_roster = []
        
        # Straight shifts (9 hours continuous)
        straight_start_times = [7, 8, 9, 10]
        
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
        
        # Split shifts with specific requirements
        split_champs = champions[straight_shifts:straight_shifts + split_shifts]
        
        # Apply specific shift requirements
        for i, champ in enumerate(split_champs):
            if i == 0:  # First split shift: 7-12 & 4-9
                pattern = self.split_shift_patterns[0]
            elif i == 1:  # Second split shift: 8-1 & 5-9
                pattern = self.split_shift_patterns[1]
            elif i == 2:  # Third split shift: 10-3 & 5-9
                pattern = self.split_shift_patterns[2]
            elif i == 3:  # Fourth split shift: 12-9
                pattern = self.split_shift_patterns[3]
            elif i == 4:  # Fifth split shift: 11-8
                pattern = self.split_shift_patterns[4]
            else:  # Default to first pattern if more than 5 split shifts
                pattern = self.split_shift_patterns[0]
            
            daily_roster.append({
                'Day': day,
                'Champion': champ['name'],
                'Primary Language': champ['primary_lang'].upper(),
                'Secondary Languages': ', '.join([lang.upper() for lang in champ['secondary_langs']]),
                'Shift Type': 'Split',
                'Start Time': pattern['display'],
                'End Time': f"{pattern['times'][-1]:02d}:00",
                'Duration': '9 hours (with break)',
                'Calls/Hour Capacity': champ['calls_per_hour'],
                'Can Split': 'Yes' if champ['can_split'] else 'No'
            })
        
        return daily_roster

    def calculate_coverage(self, roster_df, analysis_data):
        """Calculate coverage metrics"""
        if roster_df is None or analysis_data is None:
            return None
            
        total_capacity = 0
        for _, row in roster_df.iterrows():
            if row['Shift Type'] == 'Straight':
                hours_worked = 9
            else:
                shifts = row['Start Time'].split(' & ')
                hours_worked = 0
                for shift in shifts:
                    times = shift.split(' to ')
                    start_hour = int(times[0].split(':')[0])
                    end_hour = int(times[1].split(':')[0])
                    hours_worked += (end_hour - start_hour)
            
            total_capacity += row['Calls/Hour Capacity'] * hours_worked
        
        required_capacity = analysis_data['total_daily_calls'] * 7 * 1.1
        
        utilization_rate = min(100, (required_capacity / total_capacity) * 100) if total_capacity > 0 else 0
        expected_answer_rate = min(100, (total_capacity / (analysis_data['total_daily_calls'] * 7)) * 100) if analysis_data['total_daily_calls'] > 0 else 0
        
        return {
            'total_capacity': total_capacity,
            'required_capacity': required_capacity,
            'utilization_rate': utilization_rate,
            'expected_answer_rate': expected_answer_rate
        }
    
    def calculate_answer_rate(self, roster_df, analysis_data):
        """Calculate expected Answer Rate percentage - FIXED VERSION"""
        if roster_df is None or analysis_data is None:
            return None
            
        total_weekly_capacity = 0
        for day in roster_df['Day'].unique():
            day_roster = roster_df[roster_df['Day'] == day]
            for _, row in day_roster.iterrows():
                if row['Shift Type'] == 'Straight':
                    hours_worked = 9
                else:
                    shifts = row['Start Time'].split(' & ')
                    hours_worked = 0
                    for shift in shifts:
                        times = shift.split(' to ')
                        start_hour = int(times[0].split(':')[0])
                        end_hour = int(times[1].split(':')[0])
                        hours_worked += (end_hour - start_hour)
                
                total_weekly_capacity += row['Calls/Hour Capacity'] * hours_worked
        
        total_weekly_calls = analysis_data['total_daily_calls'] * 7
        
        if total_weekly_calls > 0:
            answer_rate = min(100, (total_weekly_capacity / total_weekly_calls) * 100)
        else:
            answer_rate = 0
            
        return answer_rate
    
    def calculate_daily_answer_rates(self, roster_df, analysis_data):
        """Calculate Answer Rate for each day - FIXED VERSION"""
        if roster_df is None or analysis_data is None:
            return None
            
        daily_rates = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for day in days:
            day_roster = roster_df[roster_df['Day'] == day]
            if len(day_roster) == 0:
                daily_rates[day] = 0
                continue
                
            daily_capacity = 0
            for _, row in day_roster.iterrows():
                if row['Shift Type'] == 'Straight':
                    hours_worked = 9
                else:
                    shifts = row['Start Time'].split(' & ')
                    hours_worked = 0
                    for shift in shifts:
                        times = shift.split(' to ')
                        start_hour = int(times[0].split(':')[0])
                        end_hour = int(times[1].split(':')[0])
                        hours_worked += (end_hour - start_hour)
                
                daily_capacity += row['Calls/Hour Capacity'] * hours_worked
            
            if analysis_data['total_daily_calls'] > 0:
                daily_rates[day] = min(100, (daily_capacity / analysis_data['total_daily_calls']) * 100)
            else:
                daily_rates[day] = 0
        
        return daily_rates
    
    def show_editable_roster(self, roster_df, week_offs):
        """Display an editable roster table with week offs"""
        st.subheader("✏️ Edit Roster Manually")
        
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
        
        st.subheader("📅 Edit Weekly Offs")
        week_off_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        week_off_editor_data = []
        for champ, off_day in week_offs.items():
            week_off_editor_data.append({
                "Champion": champ,
                "Current Day Off": off_day if off_day in week_off_days else "No day off"
            })
        
        week_off_df = pd.DataFrame(week_off_editor_data)
        
        edited_week_offs = st.data_editor(
            week_off_df,
            column_config={
                "Champion": st.column_config.SelectboxColumn(
                    "Champion",
                    options=[champ["name"] for champ in self.champions],
                    required=True
                ),
                "Current Day Off": st.column_config.SelectboxColumn(
                    "Day Off",
                    options=week_off_days + ["No day off"],
                    required=True
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        new_week_offs = {}
        for _, row in edited_week_offs.iterrows():
            if row["Current Day Off"] != "No day off":
                new_week_offs[row["Champion"]] = row["Current Day Off"]
        
        return edited_df, new_week_offs
    
    def assign_weekly_offs(self, roster_df, max_offs_per_day=4, min_split_champs=4):
        """Assign weekly off days to champions with limit per day, ensuring split shift coverage"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        champions = roster_df['Champion'].unique()
        
        updated_roster = roster_df.copy()
        week_offs = {}
        offs_per_day = {day: 0 for day in days}
        
        split_champs = []
        for champ in champions:
            champ_shifts = updated_roster[updated_roster['Champion'] == champ]
            if any('Split' in shift_type for shift_type in champ_shifts['Shift Type']):
                split_champs.append(champ)
        
        for champion in champions:
            champ_days = updated_roster[updated_roster['Champion'] == champion]['Day'].unique()
            
            if len(champ_days) == 7:
                available_off_days = []
                for day in days:
                    if offs_per_day[day] < max_offs_per_day:
                        if champion in split_champs:
                            split_champs_working = 0
                            for split_champ in split_champs:
                                if split_champ != champion and day in updated_roster[updated_roster['Champion'] == split_champ]['Day'].values:
                                    split_champs_working += 1
                            if split_champs_working >= min_split_champs:
                                available_off_days.append(day)
                        else:
                            available_off_days.append(day)
                
                if available_off_days:
                    day_off = random.choice(available_off_days)
                    week_offs[champion] = day_off
                    offs_per_day[day_off] += 1
                    
                    updated_roster = updated_roster[
                        ~((updated_roster['Champion'] == champion) & (updated_roster['Day'] == day_off))
                    ]
                else:
                    week_offs[champion] = "No day off (max reached)"
            else:
                working_days = set(champ_days)
                all_days = set(days)
                day_off = list(all_days - working_days)[0] if all_days - working_days else "No day off"
                week_offs[champion] = day_off
                if day_off in days:
                    offs_per_day[day_off] += 1
        
        return updated_roster, week_offs
    
    def apply_special_rules(self, roster_df):
        """Apply special rules like Revathi always on split shift"""
        revathi_mask = roster_df['Champion'] == 'Revathi'
        roster_df.loc[revathi_mask, 'Shift Type'] = 'Split'
        
        for idx in roster_df[revathi_mask].index:
            pattern = self.split_shift_patterns[0]
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
            
            mask = (roster_df['Champion'] == champ) & (roster_df['Day'] == day)
            if mask.any():
                roster_df.loc[mask, 'Shift Type'] = 'Split'
                roster_df.loc[mask, 'Start Time'] = pattern['display']
                roster_df.loc[mask, 'End Time'] = f"{pattern['times'][-1]:02d}:00"
                roster_df.loc[mask, 'Duration'] = '9 hours (with break)'
        
        return roster_df
    
    def filter_split_shift_champs(self, roster_df, can_split_only=True):
        """Filter champions who can work split shifts"""
        if can_split_only:
            split_champs = [champ["name"] for champ in self.champions if champ["can_split"]]
            
            filtered_roster = roster_df.copy()
            split_mask = filtered_roster['Shift Type'] == 'Split'
            filtered_roster = filtered_roster[~split_mask | filtered_roster['Champion'].isin(split_champs)]
            
            return filtered_roster
        else:
            return roster_df
    
    def format_roster_for_display(self, roster_df, week_offs):
        """Format the roster in the sample Excel format"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        display_df = pd.DataFrame()
        display_df['Name'] = [champ['name'] for champ in self.champions if champ['name'] in roster_df['Champion'].unique()]
        
        display_df['Shift'] = ""
        
        for day in days:
            display_df[day] = ""
        
        for _, row in roster_df.iterrows():
            champ_name = row['Champion']
            day = row['Day']
            shift_time = row['Start Time']
            
            champ_idx = display_df[display_df['Name'] == champ_name].index
            if len(champ_idx) > 0:
                display_df.at[champ_idx[0], day] = shift_time
                
                if display_df.at[champ_idx[0], 'Shift'] == "":
                    display_df.at[champ_idx[0], 'Shift'] = shift_time
        
        for champ, off_day in week_offs.items():
            if off_day in days:
                champ_idx = display_df[display_df['Name'] == champ].index
                if len(champ_idx) > 0:
                    display_df.at[champ_idx[0], off_day] = "WO"
        
        return display_df

    def calculate_late_hour_coverage(self, roster_df):
        """Calculate how many agents are available during late hours (5 PM - 9 PM)"""
        if roster_df is None:
            return None
            
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        late_hour_coverage = {day: {"mid_shift": 0, "split_shift": 0, "total": 0} for day in days}
        
        for day in days:
            day_roster = roster_df[roster_df['Day'] == day]
            for _, row in day_roster.iterrows():
                if self.is_agent_working_at_late_hours(row):
                    if "&" in row['Start Time']:
                        late_hour_coverage[day]["split_shift"] += 1
                    else:
                        late_hour_coverage[day]["mid_shift"] += 1
                    late_hour_coverage[day]["total"] += 1
        
        return late_hour_coverage
    
    def is_agent_working_at_late_hours(self, row):
        """Check if an agent is working during late hours (5 PM - 9 PM)"""
        try:
            if row['Shift Type'] == 'Straight':
                times = row['Start Time'].split(' to ')
                start_hour = int(times[0].split(':')[0])
                end_hour = int(times[1].split(':')[0])
                return start_hour <= 17 < end_hour or start_hour < 21 <= end_hour
            else:
                shifts = row['Start Time'].split(' & ')
                for shift in shifts:
                    times = shift.split(' to ')
                    start_hour = int(times[0].split(':')[0])
                    end_hour = int(times[1].split(':')[0])
                    if start_hour <= 17 < end_hour or start_hour < 21 <= end_hour:
                        return True
                return False
        except:
            return False

    def validate_split_shift_coverage(self, roster_df):
        """Validate that we have enough split shift champions each day"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        validation_results = {}
        
        active_split_champs = getattr(st.session_state, 'active_split_champs', 4)
        
        for day in days:
            day_roster = roster_df[roster_df['Day'] == day]
            split_champs_count = len(day_roster[day_roster['Shift Type'] == 'Split'])
            validation_results[day] = {
                'split_champs': split_champs_count,
                'status': '✅ Sufficient' if split_champs_count >= active_split_champs else '❌ Insufficient'
            }
        
        return validation_results

    def validate_al_target(self, roster_df, analysis_data):
        """Validate that we achieve at least 80% AL target"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        validation_results = {}
        
        for day in days:
            day_roster = roster_df[roster_df['Day'] == day]
            
            daily_capacity = 0
            for _, row in day_roster.iterrows():
                if row['Shift Type'] == 'Straight':
                    hours_worked = 9
                else:
                    shifts = row['Start Time'].split(' & ')
                    hours_worked = 0
                    for shift in shifts:
                        times = shift.split(' to ')
                        start_hour = int(times[0].split(':')[0])
                        end_hour = int(times[1].split(':')[0])
                        hours_worked += (end_hour - start_hour)
                
                daily_capacity += row['Calls/Hour Capacity'] * hours_worked
            
            if analysis_data['total_daily_calls'] > 0:
                expected_al = min(100, (daily_capacity / analysis_data['total_daily_calls']) * 100)
            else:
                expected_al = 0
                
            validation_results[day] = {
                'expected_al': expected_al,
                'status': '✅ Target Met' if expected_al >= 80 else '❌ Below Target'
            }
        
        return validation_results

# Main application
def main():
    st.markdown('<h1 class="main-header">📞 Call Center Roster Optimizer</h1>', unsafe_allow_html=True)
    
    optimizer = CallCenterRosterOptimizer()
    
    if 'manual_splits' not in st.session_state:
        st.session_state.manual_splits = []
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Shift Configuration")
        
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
            value=5,
            help="Number of champions working split shifts with afternoon break"
        )
        
        if straight_shifts + split_shifts > total_champs:
            st.warning(f"⚠️ Only {total_champs} champions available! Reducing split shifts...")
            split_shifts = total_champs - straight_shifts
        
        st.metric("Total Champions Used", f"{straight_shifts + split_shifts}/{total_champs}")
        
        # NEW: Split shift active champions selection
        st.subheader("Active Split Shift Champions")
        min_split_champs = max(4, split_shifts)
        active_split_champs = st.slider(
            "Minimum Active Split Shift Champions per Day",
            min_value=4,
            max_value=split_shifts,
            value=min(5, split_shifts),
            help="Set the minimum number of split shift champions that must be active each day"
        )
        
        st.session_state.active_split_champs = active_split_champs
        
        st.subheader("Split Shift Filter")
        split_filter = st.checkbox(
            "Only assign split shifts to champions who can split",
            value=True,
            help="When enabled, only champions marked as able to work split shifts will be assigned to them"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📊 Data Template")
        
        st.markdown('<div class="template-download">', unsafe_allow_html=True)
        st.subheader("Download Data Template")
        st.write("Download this template, add your call volume data, and upload it back.")
        
        template_data = optimizer.create_template_file()
        
        st.download_button(
            "📥 Download Data Template",
            template_data,
            "call_volume_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload Your Call Volume Data", 
            type=['xlsx', 'xls'],
            help="Upload your filled-in call volume data Excel file"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔧 Manual Split Shift Assignment")
        
        st.markdown('<div class="split-shift-option">', unsafe_allow_html=True)
        st.subheader("Assign Specific Split Shifts")
        
        available_champs = [champ["name"] for champ in optimizer.champions]
        selected_champ = st.selectbox("Select Champion", available_champs)
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day = st.selectbox("Select Day", days)
        
        pattern_options = [f"{pattern['name']} ({pattern['display']})" 
                          for pattern in optimizer.split_shift_patterns]
        selected_pattern_idx = st.selectbox("Select Split Shift Pattern", range(len(pattern_options)), format_func=lambda x: pattern_options[x])
        
        if st.button("➕ Add Manual Split Assignment"):
            new_assignment = {
                'champion': selected_champ,
                'day': selected_day,
                'pattern': optimizer.split_shift_patterns[selected_pattern_idx]
            }
            
            if not any(a['champion'] == selected_champ and a['day'] == selected_day for a in st.session_state.manual_splits):
                st.session_state.manual_splits.append(new_assignment)
                st.success(f"Added split shift for {selected_champ} on {selected_day}")
            else:
                st.warning(f"{selected_champ} already has a manual split assignment on {selected_day}")
        
        if st.session_state.manual_splits:
            st.subheader("Current Manual Assignments")
            for i, assignment in enumerate(st.session_state.manual_splits):
                st.write(f"{i+1}. {assignment['champion']} - {assignment['day']} - {assignment['pattern']['name']}")
                if st.button(f"Remove {i+1}", key=f"remove_{i}"):
                    st.session_state.manual_splits.pop(i)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
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
        - 🎯 **AL Target:** 80% Answer Level
        - 👥 **Late Hour Coverage:** Minimum 3 mid-shift + 3 split-shift agents during 5 PM - 9 PM
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
            st.session_state.analysis_data = {
                'hourly_volume': {
                    7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
                    13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
                    18: 224.9, 19: 179.3, 20: 113.9, 21: 0
                },
                'peak_hours': [11, 12, 13, 14],
                'total_daily_calls': 17500 / 7
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
                
                if split_filter:
                    roster_df = optimizer.filter_split_shift_champs(roster_df, can_split_only=True)
                
                if roster_df is not None:
                    st.session_state.roster_df = roster_df
                    st.session_state.week_offs = week_offs
                    
                    metrics = optimizer.calculate_coverage(roster_df, st.session_state.analysis_data)
                    answer_rate = optimizer.calculate_answer_rate(roster_df, st.session_state.analysis_data)
                    daily_rates = optimizer.calculate_daily_answer_rates(roster_df, st.session_state.analysis_data)
                    
                    hourly_al_results = optimizer.calculate_hourly_al_analysis(roster_df, st.session_state.analysis_data)
                    late_hour_coverage = optimizer.calculate_late_hour_coverage(roster_df)
                    
                    st.session_state.metrics = metrics
                    st.session_state.answer_rate = answer_rate
                    st.session_state.daily_rates = daily_rates
                    st.session_state.hourly_al_results = hourly_al_results
                    st.session_state.late_hour_coverage = late_hour_coverage
                    
                    st.session_state.formatted_roster = optimizer.format_roster_for_display(roster_df, week_offs)
                    
                    st.success("✅ Roster generated successfully!")
        
        elif 'roster_df' in st.session_state:
            st.info("📋 Previously generated roster is available. Click the button to generate a new one.")
    
    # Display results if available
    if 'roster_df' in st.session_state and 'metrics' in st.session_state:
        st.markdown("---")
        st.markdown('<div class="section-header"><h2>📊 Performance Metrics</h2></div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Weekly Capacity", f"{st.session_state.metrics['total_capacity']:,.0f} calls")
        with col2:
            st.metric("Required Capacity", f"{st.session_state.metrics['required_capacity']:,.0f} calls")
        with col3:
            st.metric("Utilization Rate", f"{st.session_state.metrics['utilization_rate']:.1f}%")
        with col4:
            st.metric("Expected Answer Rate", f"{st.session_state.answer_rate:.1f}%")
        
        # Display split shift coverage validation
        st.markdown('<div class="section-header"><h2>✅ Split Shift Coverage Validation</h2></div>', unsafe_allow_html=True)
        
        validation_results = optimizer.validate_split_shift_coverage(st.session_state.roster_df)
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        cols = st.columns(7)
        
        for i, day in enumerate(days):
            with cols[i]:
                result = validation_results[day]
                st.metric(
                    day[:3], 
                    f"{result['split_champs']} split champs", 
                    result['status'],
                    delta_color="normal" if result['status'] == '✅ Sufficient' else "inverse"
                )
        
        insufficient_days = [day for day, result in validation_results.items() if result['status'] == '❌ Insufficient']
        if insufficient_days:
            st.error(f"⚠️ {len(insufficient_days)} days have insufficient split shift coverage!")
            st.info(f"**Recommendation:** Increase the 'Minimum Active Split Shift Champions' value or add more split shifts.")
        
        # Display AL target validation
        st.markdown('<div class="section-header"><h2>🎯 AL Target Validation (Minimum 80%)</h2></div>', unsafe_allow_html=True)
        
        al_validation = optimizer.validate_al_target(st.session_state.roster_df, st.session_state.analysis_data)
        
        cols = st.columns(7)
        
        for i, day in enumerate(days):
            with cols[i]:
                result = al_validation[day]
                st.metric(
                    day[:3], 
                    f"{result['expected_al']:.1f}%", 
                    result['status'],
                    delta_color="normal" if result['status'] == '✅ Target Met' else "inverse"
                )
        
        below_target_days = [day for day, result in al_validation.items() if result['status'] == '❌ Below Target']
        if below_target_days:
            st.error(f"⚠️ {len(below_target_days)} days are below the 80% AL target!")
            st.info("**Recommendation:** Increase the number of agents or adjust shift patterns.")
        
        # Display daily answer rates
        st.markdown('<div class="section-header"><h2>📈 Daily Answer Rates</h2></div>', unsafe_allow_html=True)
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
        
        if st.checkbox("Enable manual editing of shift timings and week offs"):
            edited_roster, edited_week_offs = optimizer.show_editable_roster(
                st.session_state.roster_df, st.session_state.week_offs
            )
            
            st.session_state.roster_df = edited_roster
            st.session_state.week_offs = edited_week_offs
            
            if st.button("Update Metrics after Editing"):
                updated_metrics = optimizer.calculate_coverage(edited_roster, st.session_state.analysis_data)
                updated_answer_rate = optimizer.calculate_answer_rate(edited_roster, st.session_state.analysis_data)
                updated_daily_rates = optimizer.calculate_daily_answer_rates(edited_roster, st.session_state.analysis_data)
                updated_hourly_al = optimizer.calculate_hourly_al_analysis(edited_roster, st.session_state.analysis_data)
                updated_late_hour_coverage = optimizer.calculate_late_hour_coverage(edited_roster)
                
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

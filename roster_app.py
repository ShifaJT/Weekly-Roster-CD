import streamlit as st
import pandas as pd
import numpy as np
import pulp
from datetime import datetime, timedelta
import plotly.express as px
from io import BytesIO
import base64
import random
import openpyxl

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
.sl-cell {
    background-color: #ffcccc;
    font-weight: bold;
}
.cl-cell {
    background-color: #ffddcc;
    font-weight: bold;
}
.pl-cell {
    background-color: #ffeedd;
    font-weight: bold;
}
.al-cell {
    background-color: #ffffcc;
    font-weight: bold;
}
.co-cell {
    background-color: #ccffcc;
    font-weight: bold;
}
.maternity-cell {
    background-color: #e6ccff;
    font-weight: bold;
}
.leave-badge {
    padding: 0.2rem 0.5rem;
    border-radius: 0.3rem;
    font-size: 0.8rem;
    font-weight: bold;
    margin: 0.1rem;
}
.badge-sl {
    background-color: #ffcccc;
    color: #d93025;
}
.badge-cl {
    background-color: #ffddcc;
    color: #e65100;
}
.badge-pl {
    background-color: #ffeedd;
    color: #f57c00;
}
.badge-al {
    background-color: #ffffcc;
    color: #f9ab00;
}
.badge-co {
    background-color: #ccffcc;
    color: #137333;
}
.badge-maternity {
    background-color: #e6ccff;
    color: #6a0dad;
}
.badge-fullday {
    background-color: #ff6666;
    color: white;
}
.badge-firsthalf {
    background-color: #ff9966;
    color: white;
}
.badge-secondhalf {
    background-color: #66aaff;
    color: white;
}
.badge-emergency {
    background-color: #ff3366;
    color: white;
}
</style>
""", unsafe_allow_html=True)

class CallCenterRosterOptimizer:
    def __init__(self):
        self.operation_hours = list(range(7, 22))
        self.champions = self.load_champions()
        
        self.TARGET_AL = 95
        self.MIN_AL = 95

        self.shift_patterns = [
            {"name": "7-4", "times": (7, 16), "display": "07:00 to 16:00", "hours": 9, "type": "straight"},
            {"name": "8-5", "times": (8, 17), "display": "08:00 to 17:00", "hours": 9, "type": "straight"},
            {"name": "9-6", "times": (9, 18), "display": "09:00 to 18:00", "hours": 9, "type": "straight"},
            {"name": "10-7", "times": (10, 19), "display": "10:00 to 19:00", "hours": 9, "type": "straight"},
            {"name": "11-8", "times": (11, 20), "display": "11:00 to 20:00", "hours": 9, "type": "straight"},
            {"name": "12-9", "times": (12, 21), "display": "12:00 to 21:00", "hours": 9, "type": "straight"},
            {"name": "7-12 & 4-9", "times": (7, 12, 16, 21), "display": "07:00 to 12:00 & 16:30 to 21:00", "hours": 9.5, "type": "split"},
            {"name": "8-1 & 5-9", "times": (8, 13, 17, 21), "display": "08:00 to 13:00 & 17:00 to 21:00", "hours": 9, "type": "split"},
            {"name": "9-2 & 6-9", "times": (10, 15, 17, 21), "display": "09:00 to 14:00 & 18:00 to 21:00", "hours": 8, "type": "split"},
        ]

        self.AVERAGE_HANDLING_TIME_SECONDS = 202

    def load_champions(self):
        return [
            {"name": "Revathi", "primary_lang": "ka", "secondary_langs": ["hi", "te", "ta"], "calls_per_hour": 14, "can_split": True, "gender": "F", "status": "Active"},
            {"name": "Pasang", "primary_lang": "ka", "secondary_langs": ["hi", "ta"], "calls_per_hour": 13, "can_split": False, "gender": "F", "status": "Active"},
            {"name": "Kavya S", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 15, "can_split": False, "gender": "F", "status": "Active"},
            {"name": "Anjali", "primary_lang": "ka", "secondary_langs": ["te", "hi"], "calls_per_hour": 14, "can_split": True, "gender": "F", "status": "Active"},
            {"name": "Alwin", "primary_lang": "hi", "secondary_langs": ["ka"], "calls_per_hour": 13, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Marcelina J", "primary_lang": "ka", "secondary_langs": ["ta"], "calls_per_hour": 12, "can_split": False, "gender": "F", "status": "Active"},
            {"name": "Binita Kongadi", "primary_lang": "hi", "secondary_langs": [], "calls_per_hour": 11, "can_split": True, "gender": "F", "status": "Active"},
            {"name": "Pooja N", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 14, "can_split": True, "gender": "F", "status": "Active"},
            {"name": "Sadanad", "primary_lang": "ka", "secondary_langs": ["hi"], "calls_per_hour": 13, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Navya", "primary_lang": "ka", "secondary_langs": ["te", "ta"], "calls_per_hour": 14, "can_split": False, "gender": "F", "status": "Active"},
            {"name": "Jyothika", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 13, "can_split": True, "gender": "F", "status": "Active"},
            {"name": "Dundesh", "primary_lang": "ka", "secondary_langs": ["te", "hi"], "calls_per_hour": 12, "can_split": False, "gender": "M", "status": "Active"},
            {"name": "Rakesh", "primary_lang": "ka", "secondary_langs": ["ta"], "calls_per_hour": 13, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Malikarjun Patil", "primary_lang": "ka", "secondary_langs": ["hi"], "calls_per_hour": 14, "can_split": False, "gender": "M", "status": "Active"},
            {"name": "Divya", "primary_lang": "ka", "secondary_langs": ["te", "ta"], "calls_per_hour": 14, "can_split": True, "gender": "F", "status": "Active"},
            {"name": "Mohammed Altaf", "primary_lang": "hi", "secondary_langs": [], "calls_per_hour": 12, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Rakshith", "primary_lang": "ka", "secondary_langs": ["hi"], "calls_per_hour": 13, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "M Showkath Nawaz", "primary_lang": "ka", "secondary_langs": ["hi", "te"], "calls_per_hour": 14, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Vishal", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 13, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Muthahir", "primary_lang": "hi", "secondary_langs": ["te"], "calls_per_hour": 12, "can_split": False, "gender": "M", "status": "Active"},
            {"name": "Soubhikotl", "primary_lang": "hi", "secondary_langs": [], "calls_per_hour": 11, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Shashindra", "primary_lang": "hi", "secondary_langs": ["ka", "te"], "calls_per_hour": 13, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Sameer Pasha", "primary_lang": "hi", "secondary_langs": ["ka", "te"], "calls_per_hour": 13, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Guruswamy", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 12, "can_split": False, "gender": "M", "status": "Active"},
            {"name": "Sheikh Vali Babu", "primary_lang": "hi", "secondary_langs": ["te"], "calls_per_hour": 12, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Baloji", "primary_lang": "te", "secondary_langs": [], "calls_per_hour": 11, "can_split": False, "gender": "M", "status": "Active"},
            {"name": "waghmare", "primary_lang": "te", "secondary_langs": ["hi", "ka"], "calls_per_hour": 13, "can_split": True, "gender": "M", "status": "Active"},
            {"name": "Deepika", "primary_lang": "ka", "secondary_langs": ["hi"], "calls_per_hour": 12, "can_split": False, "gender": "F", "status": "Maternity"}
        ]

    def get_available_languages(self):
        """Get all unique languages from champions"""
        languages = set()
        for champ in self.champions:
            if champ['primary_lang']:
                languages.add(champ['primary_lang'])
            for lang in champ['secondary_langs']:
                languages.add(lang)
        return sorted(list(languages))

    def calculate_hourly_capacity(self, num_agents, aht_seconds):
        hourly_capacity = (num_agents * 3600) / aht_seconds
        return round(hourly_capacity)

    def predict_al(self, forecasted_calls, num_agents, aht_seconds):
        capacity = self.calculate_hourly_capacity(num_agents, aht_seconds)
        answered_calls = min(capacity, forecasted_calls)
        if forecasted_calls > 0:
            predicted_al = (answered_calls / forecasted_calls) * 100
        else:
            predicted_al = 100
        return round(predicted_al, 1), int(capacity), int(answered_calls)

    def agents_needed_for_target(self, forecasted_calls, target_al, aht_seconds):
        if forecasted_calls <= 0:
            return 0
        required_capacity = (target_al / 100) * forecasted_calls
        agents_required = (required_capacity * aht_seconds) / 3600
        return max(1, np.ceil(agents_required))

    def analyze_roster_sufficiency(self, forecasted_calls, scheduled_agents, aht_seconds, target_al):
        predicted_al, capacity, answered_calls = self.predict_al(forecasted_calls, scheduled_agents, aht_seconds)
        min_agents_required = self.agents_needed_for_target(forecasted_calls, target_al, aht_seconds)

        recommendation = {
            'forecasted_calls': forecasted_calls,
            'scheduled_agents': scheduled_agents,
            'current_capacity': capacity,
            'answered_calls': answered_calls,
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
        elif predicted_al >= 90:
            recommendation['status'] = '🟡 YELLOW (Close to Goal)'
            recommendation['recommendation'] = f'Close to target. Adding 1 agent could help secure {target_al}%.'
        elif predicted_al >= 85:
            recommendation['status'] = '🟠 ORANGE (At Risk)'
            recommendation['recommendation'] = f'Add {int(recommendation["agents_deficit"])} agent(s) to reach {target_al}%.'
        else:
            recommendation['status'] = '🔴 RED (Critical)'
            recommendation['recommendation'] = f'Immediately add {int(recommendation["agents_deficit"])} agent(s) to reach {target_al}%.'

        return recommendation

    def calculate_hourly_al_analysis(self, roster_df, analysis_data):
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
                    predicted_al, capacity, answered = self.predict_al(forecasted_calls, agents_at_hour, self.AVERAGE_HANDLING_TIME_SECONDS)

                    key = f"{day}_{hour}"
                    hourly_al_results[key] = {
                        'day': day,
                        'hour': hour,
                        'agents': agents_at_hour,
                        'forecast': forecasted_calls,
                        'capacity': capacity,
                        'answered': answered,
                        'predicted_al': predicted_al,
                        'status': self.get_al_status(predicted_al)
                    }

        return hourly_al_results

    def is_agent_working_at_hour(self, row, hour):
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
        if al_value >= self.TARGET_AL:
            return "✅ GOOD"
        elif al_value >= 90:
            return "🟡 WARNING"
        elif al_value >= 85:
            return "🟠 AT RISK"
        else:
            return "🔴 CRITICAL"

    def create_template_file(self):
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

        leave_data = pd.DataFrame({
            'Champion': [champ['name'] for champ in self.champions],
            'Sick_Leave': [0] * len(self.champions),
            'Casual_Leave': [0] * len(self.champions),
            'Period_Leave': [0] * len(self.champions),
            'Annual_Leave': [0] * len(self.champions),
            'Comp_Off': [0] * len(self.champions),
            'Maternity_Leave': [1 if champ['status'] == 'Maternity' else 0 for champ in self.champions]
        })

        instructions = pd.DataFrame({
            'Instruction': [
                'INSTRUCTIONS:',
                '1. Fill in your call volume data in the appropriate sheets',
                '2. For best results, use the Hourly_Data sheet with calls per hour',
                '3. If you only have daily totals, use the Daily_Data sheet',
                '4. Add leave information in the Leave_Data sheet (0=no leave, 1=on leave)',
                '5. Save the file and upload it back to the app',
                '6. The app will analyze your data and generate an optimized roster',
                '',
                'HOURLY_DATA SHEET:',
                '- Hour: Operation hour (7 to 21)',
                '- Calls: Number of calls received in that hour',
                '',
                'DAILY_DATA SHEET:',
                '- Date: Date of the data (YYYY-MM-DD)',
                '- Total_Calls: Total calls received that day',
                '- Peak_Hour: Hour with highest call volume (7-21)',
                '- Peak_Volume: Number of calls during peak hour',
                '',
                'LEAVE_DATA SHEET:',
                '- Champion: Name of the agent',
                '- Sick_Leave: 1 if on sick leave, 0 otherwise',
                '- Casual_Leave: 1 if on casual leave, 0 otherwise',
                '- Period_Leave: 1 if on period leave, 0 otherwise',
                '- Annual_Leave: 1 if on annual leave, 0 otherwise',
                '- Comp_Off: 1 if on comp off, 0 otherwise',
                '- Maternity_Leave: 1 if on maternity leave, 0 otherwise'
            ]
        })

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            instructions.to_excel(writer, sheet_name='Instructions', index=False)
            hourly_data.to_excel(writer, sheet_name='Hourly_Data', index=False)
            daily_data.to_excel(writer, sheet_name='Daily_Data', index=False)
            leave_data.to_excel(writer, sheet_name='Leave_Data', index=False)

        return excel_buffer.getvalue()

    def analyze_excel_data(self, uploaded_file):
        try:
            try:
                xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
            except:
                if uploaded_file.name.endswith('.xls'):
                    st.error("XLS format detected. Please convert your file to XLSX format.")
                    st.info("You can convert XLS to XLSX by opening in Excel and saving as XLSX.")
                    return self.get_sample_data()
                else:
                    xls = pd.ExcelFile(uploaded_file)
                    
            analysis_data = {}
            leave_data = {}

            if 'Hourly_Data' in xls.sheet_names:
                try:
                    df = pd.read_excel(uploaded_file, sheet_name='Hourly_Data', engine='openpyxl')
                except:
                    df = pd.read_excel(uploaded_file, sheet_name='Hourly_Data')

                if 'Hour' in df.columns and 'Calls' in df.columns:
                    hourly_volume = dict(zip(df['Hour'], df['Calls']))
                    peak_hours = df.nlargest(4, 'Calls')['Hour'].tolist()
                    total_daily_calls = df['Calls'].sum()

                    analysis_data = {
                        'hourly_volume': hourly_volume,
                        'peak_hours': peak_hours,
                        'total_daily极速飞艇开_calls': total_daily_calls
                    }

            elif 'Daily_Data' in xls.sheet_names:
                try:
                    df = pd.read_excel(uploaded_file, sheet_name='Daily_Data', engine='openpyxl')
                except:
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

                    analysis_data = {
                        'hourly_volume': hourly_volume,
                        'peak_hours': peak_hours,
                        'total_daily_calls': avg_daily_calls
                    }

            if 'Leave_Data' in xls.sheet_names:
                try:
                    df = pd.read_excel(uploaded_file, sheet_name='Leave_Data', engine='openpyxl')
                except:
                    df = pd.read_excel(uploaded_file, sheet_name='Leave_Data')
                    
                if 'Champion' in df.columns:
                    for _, row in df.iterrows():
                        champion = row['Champion']
                        leave_data[champion] = {
                            'sick_leave': row.get('Sick_Leave', 0),
                            'casual_leave': row.get('Casual_Leave', 0),
                            'period_leave': row.get('Period_Leave', 0),
                            'annual_leave': row.get('Annual_Leave', 0),
                            'comp_off': row.get('Comp_Off', 0),
                            'maternity_leave': row.get('Maternity_Leave', 0)
                        }

            if not analysis_data:
                st.warning("No recognized data format. Using sample data.")
                analysis_data = self.get_sample_data()

            analysis_data['leave_data'] = leave_data
            return analysis_data

        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            st.info("Using sample data instead.")
            sample_data = self.get_sample_data()
            sample_data['leave_data'] = {}
            return sample_data

    def get_sample_data(self):
        return {
            'hourly_volume': {
                7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
                13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
                18: 224.9, 19: 179.3, 20: 113.9, 21: 0
            },
            'peak_hours': [11, 12, 13, 14],
            'total_daily_calls': 3130
        }

    def optimize_roster_for_call_flow(self, analysis_data, available_champions, selected_languages=None):
        hourly_volume = analysis_data['hourly_volume']
        
        required_agents_per_hour = {}
        for hour, calls in hourly_volume.items():
            required_agents_per_hour[hour] = self.agents_needed_for_target(calls, self.TARGET_AL, self.AVERAGE_HANDLING_TIME_SECONDS)
        
        prob = pulp.LpProblem("RosterOptimization", pulp.LpMinimize)
        
        shifts = self.shift_patterns
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Create variables for each champion and day
        x = pulp.LpVariable.dicts("assign", 
                                 ((champ['name'], day) 
                                  for champ in available_champions 
                                  for day in days),
                                 cat='Binary')
        
        # Create shift assignment variables
        y = pulp.LpVariable.dicts("shift_assign",
                                 ((champ['name'], day, shift_idx)
                                  for champ in available_champions
                                  for day in days
                                  for shift_idx in range(len(shifts))),
                                 cat='Binary')
        
        # Objective: Minimize the maximum deviation from required agents
        max_deviation = pulp.LpVariable("max_deviation", lowBound=0, cat='Continuous')
        
        # Add constraints to minimize maximum deviation
        for hour in self.operation_hours:
            agents_at_hour = pulp.lpSum([
                y[champ['name'], day, shift_idx] 
                for champ in available_champions 
                for day in days 
                for shift_idx, shift in enumerate(shifts)
                if self.is_hour_in_shift(hour, shift)
            ])
            
            # Constraint: agents at hour should be close to required
            prob += agents_at_hour >= required_agents_per_hour[hour] - max_deviation
            prob += agents_at_hour <= required_agents_per_hour[hour] + max_deviation
        
        # Each champion works at most one shift per day
        for champ in available_champions:
            for day in days:
                prob += pulp.lpSum([y[champ['name'], day, shift_idx] for shift_idx in range(len(shifts))]) <= 1
                # Link x and y variables
                prob += x[champ['name'], day] == pulp.lpSum([y[champ['name'], day, shift_idx] for shift_idx in range(len(shifts))])
        
        # Each champion works 5 days per week
        for champ in available_champions:
            prob += pulp.lpSum([x[champ['name'], day] for day in days]) == 5
        
        # Female champions don't work late shifts (after 8 PM)
        for champ in available_champions:
            if champ['gender'] == 'F':
                for day in days:
                    for shift_idx, shift in enumerate(shifts):
                        if shift['times'][-1] >= 20:
                            prob += y[champ['name'], day, shift_idx] == 0
        
        # Ensure proper coverage during peak hours
        peak_hours = analysis_data.get('peak_hours', [11, 12, 13, 14])
        for hour in peak_hours:
            agents_at_hour = pulp.lpSum([
                y[champ['name'], day, shift_idx] 
                for champ in available_champions 
                for day in days 
                for shift_idx, shift in enumerate(shifts)
                if self.is_hour_in_shift(hour, shift)
            ])
            prob += agents_at_hour >= required_agents_per_hour[hour] * 0.9  # At least 90% of required during peak
        
        prob += max_deviation
        
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
        except:
            return self.generate_fallback_roster(available_champions, days, shifts)
        
        # Extract the solution
        roster_data = []
        for champ in available_champions:
            for day in days:
                for shift_idx in range(len(shifts)):
                    if pulp.value(y[champ['name'], day, shift_idx]) == 1:
                        shift = shifts[shift_idx]
                        roster_data.append({
                            'Day': day,
                            'Champion': champ['name'],
                            'Primary Language': champ['primary_lang'].upper(),
                            'Secondary Languages': ', '.join([lang.upper() for lang in champ['secondary_langs']]),
                            'Shift Type': 'Split' if shift['type'] == 'split' else 'Straight',
                            'Start Time': shift['display'],
                            'End Time': f"{shift['times'][-1]:02d}:00",
                            'Duration': f'{shift["hours"]} hours',
                            'Calls/Hour Capacity': champ['calls_per_hour'],
                            'Can Split': 'Yes' if champ['can_split'] else 'No',
                            'Gender': champ['gender'],
                            'Status': champ['status']
                        })
        
        return pd.DataFrame(roster_data)

    def generate_fallback_roster(self, available_champions, days, shifts):
        roster_data = []
        
        straight_shifts = [s for s in shifts if s['type'] == 'straight']
        split_shifts = [s for s in shifts if s['type'] == 'split']
        
        # Distribute champions evenly across days
        for i, champ in enumerate(available_champions):
            work_days = random.sample(days, 5)  # Each champ works 5 days
            
            for day in work_days:
                if champ['can_split'] and i % 3 == 0:
                    shift = random.choice(split_shifts)
                else:
                    shift = random.choice(straight_shifts)
                
                roster_data.append({
                    'Day': day,
                    'Champion': champ['name'],
                    'Primary Language': champ['primary_lang'].upper(),
                    'Secondary Languages': ', '.join([lang.upper() for lang in champ['secondary_langs']]),
                    'Shift Type': 'Split' if shift['type'] == 'split' else 'Straight',
                    'Start Time': shift['display'],
                    'End Time': f"{shift['times'][-1]:02d}:00",
                    'Duration': f'{shift["hours"]} hours',
                    'Calls/Hour Capacity': champ['calls_per_hour'],
                    'Can Split': 'Yes' if champ['can_split'] else 'No',
                    'Gender': champ['gender'],
                    'Status': champ['status']
                })
        
        return pd.DataFrame(roster_data)

    def is_hour_in_shift(self, hour, shift):
        if shift['type'] == 'straight':
            return shift['times'][0] <= hour < shift['times'][1]
        else:
            return (shift['times'][0] <= hour < shift['times'][1]) or (shift['times'][2] <= hour < shift['times'][3])

    def generate_roster(self, analysis_data, manual_splits=None, selected_languages=None):
        try:
            available_champions = self.get_available_champions(analysis_data.get('leave_data', {}))
            
            # Filter by selected languages if any
            if selected_languages:
                available_champions = [champ for champ in available_champions 
                                     if champ['primary_lang'] in selected_languages 
                                     or any(lang in selected_languages for lang in champ['secondary_langs'])]
            
            roster_df = self.optimize_roster_for_call_flow(analysis_data, available_champions, selected_languages)
            
            # Ensure no blank cells - fill missing days
            roster_df = self.fill_missing_days(roster_df, available_champions)
            
            roster_df = self.apply_special_rules(roster_df)

            if manual_splits:
                roster_df = self.apply_manual_splits(roster_df, manual_splits)

            active_split_champs = getattr(st.session_state, 'active_split_champs', 4)
            roster_df, week_offs = self.assign_weekly_offs(roster_df, max_offs_per_day=4, min_split_champs=active_split_champs)

            return roster_df, week_offs

        except Exception as e:
            st.error(f"Error generating roster: {str(e)}")
            return None, None

    def fill_missing_days(self, roster_df, available_champions):
        """Ensure every champion has shifts for all working days"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        shifts = self.shift_patterns
        
        # Get current assignments
        current_assignments = {}
        for _, row in roster_df.iterrows():
            champ = row['Champion']
            day = row['Day']
            if champ not in current_assignments:
                current_assignments[champ] = {}
            current_assignments[champ][day] = row.to_dict()
        
        # Fill missing days
        new_roster_data = []
        for champ in available_champions:
            champ_days = current_assignments.get(champ, {})
            work_days = list(champ_days.keys())
            
            # If champion has less than 5 days, add missing days
            if len(work_days) < 5:
                missing_days = [day for day in days if day not in work_days]
                days_to_add = random.sample(missing_days, 5 - len(work_days))
                
                for day in days_to_add:
                    # Assign appropriate shift
                    straight_shifts = [s for s in shifts if s['type'] == 'straight']
                    split_shifts = [s for s in shifts if s['type'] == 'split']
                    
                    if champ['can_split'] and random.random() < 0.3:
                        shift = random.choice(split_shifts)
                    else:
                        shift = random.choice(straight_shifts)
                    
                    new_roster_data.append({
                        'Day': day,
                        'Champion': champ['name'],
                        'Primary Language': champ['primary_lang'].upper(),
                        'Secondary Languages': ', '.join([lang.upper() for lang in champ['secondary_langs']]),
                        'Shift Type': 'Split' if shift['type'] == 'split' else 'Straight',
                        'Start Time': shift['display'],
                        'End Time': f"{shift['times'][-1]:02d}:00",
                        'Duration': f'{shift["hours"]} hours',
                        'Calls/Hour Capacity': champ['calls_per_hour'],
                        'Can Split': 'Yes' if champ['can_split'] else 'No',
                        'Gender': champ['gender'],
                        'Status': champ['status']
                    })
            
            # Add existing assignments
            for day, assignment in champ_days.items():
                new_roster_data.append(assignment)
        
        return pd.DataFrame(new_roster_data)

    def get_available_champions(self, leave_data, specific_date=None):
        available_champs = []
        
        for champ in self.champions:
            # Skip maternity leave champions
            if champ['status'] == 'Maternity':
                continue
                
            champ_leave = leave_data.get(champ['name'], {})
            on_leave = False
            
            if specific_date:
                date_leave = champ_leave.get('date_specific', {}).get(specific_date, "")
                if date_leave and date_leave != "Second Half":
                    on_leave = True
            
            if not on_leave:
                on_leave = any([
                    champ_leave.get('sick_leave', 0) == 1,
                    champ_leave.get('casual_leave', 0) == 1,
                    champ_leave.get('period_leave', 0) == 1,
                    champ_leave.get('annual_leave', 0) == 1,
                    champ_leave.get('comp_off', 0) == 1,
                    champ_leave.get('maternity_leave', 0) == 1
                ])
            
            if not on_leave:
                available_champs.append(champ)
        
        return available_champs

    def calculate_coverage(self, roster_df, analysis_data):
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

    def show_editable_roster(self, roster_df, week_offs, leave_data):
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

        st.subheader("🏥 Edit Leave Information")
        
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_dates = [(week_start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
        week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        st.info(f"**Current Week:** {week_dates[0]} to {week_dates[-1]}")
        
        leave_editor_data = []
        for champ in self.champions:
            champ_name = champ['name']
            champ_data = {"Champion": champ_name}
            
            for leave_type in ["Sick Leave", "Casual Leave", "Period Leave", "Annual Leave", "Comp Off", "Maternity Leave"]:
                champ_data[leave_type] = leave_data.get(champ_name, {}).get(leave_type.lower().replace(' ', '_'), 0)
            
            for i, day in enumerate(week_days):
                date_str = week_dates[i]
                champ_data[f"{day} ({date_str})"] = leave_data.get(champ_name, {}).get('date_specific', {}).get(date_str, "")
            
            leave_editor_data.append(champ_data)
        
        leave_df = pd.DataFrame(leave_editor_data)
        
        column_config = {
            "Champion": st.column_config.SelectboxColumn(
                "Champion",
                options=[champ["name"] for champ in self.champions],
                required=True
            )
        }
        
        for leave_type in ["Sick Leave", "Casual Leave", "Period Leave", "Annual Leave", "Comp Off", "Maternity Leave"]:
            column_config[leave_type] = st.column_config.CheckboxColumn(leave_type)
        
        for i, day in enumerate(week_days):
            date_str = week_dates[i]
            column_config[f"{day} ({date_str})"] = st.column_config.SelectboxColumn(
                f"{day}",
                options=["", "Full Day", "First Half", "Second Half", "Emergency Leave"],
                help=f"Leave for {date_str}"
            )
        
        edited_leave_data = st.data_editor(
            leave_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=500
        )
        
        new_leave_data = {}
        for _, row in edited_leave_data.iterrows():
            champ_name = row["Champion"]
            leave_info = {
                'sick_leave': row["Sick Leave"],
                'casual_leave': row["Casual Leave"],
                'period_leave': row["Period Leave"],
                'annual_leave': row["Annual Leave"],
                'comp_off': row["Comp Off"],
                'maternity_leave': row["Maternity Leave"],
                'date_specific': {}
            }
            
            for i, day in enumerate(week_days):
                date_str = week_dates[i]
                leave_status = row[f"{day} ({date_str})"]
                if leave_status:
                    leave_info['date_specific'][date_str] = leave_status
            
            new_leave_data[champ_name] = leave_info

        return edited_df, new_week_offs, new_leave_data

    def assign_weekly_offs(self, roster_df, max_offs_per_day=4, min_split_champs=4):
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
        revathi_mask = roster_df['Champion'] == 'Revathi'
        roster_df.loc[revathi_mask, 'Shift Type'] = 'Split'

        for idx in roster_df[revathi_mask].index:
            pattern = self.shift_patterns[6]
            roster_df.at[idx, 'Start Time'] = pattern['display']
            roster_df.at[idx, 'End Time'] = f"{pattern['times'][3]:02d}:00"
            roster_df.at[idx, 'Duration'] = '9.5 hours (with break)'

        return roster_df

    def apply_manual_splits(self, roster_df, manual_splits):
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
        if can_split_only:
            split_champs = [champ["name"] for champ in self.champions if champ["can_split"]]

            filtered_roster = roster_df.copy()
            split_mask = filtered_roster['Shift Type'] == 'Split'
            filtered_roster = filtered_roster[~split_mask | filtered_roster['Champion'].isin(split_champs)]

            return filtered_roster
        else:
            return roster_df

    def format_roster_for_display(self, roster_df, week_offs, leave_data):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_dates = [(week_start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]

        display_df = pd.DataFrame()
        display_df['Name'] = [champ['name'] for champ in self.champions]
        display_df['Status'] = [champ['status'] for champ in self.champions]
        display_df['Primary Language'] = [champ['primary_lang'].upper() for champ in self.champions]
        display_df['Secondary Languages'] = [', '.join([lang.upper() for lang in champ['secondary_langs']]) for champ in self.champions]

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

        for champ_name, leave_info in leave_data.items():
            champ_idx = display_df[display_df['Name'] == champ_name].index
            if len(champ_idx) > 0:
                for i, day in enumerate(days):
                    date_str = week_dates[i]
                    date_leave = leave_info.get('date_specific', {}).get(date_str, "")
                    
                    if date_leave:
                        badge_class = ""
                        if date_leave == "Full Day":
                            badge_class = "badge-fullday"
                        elif date_leave == "First Half":
                            badge_class = "badge-firsthalf"
                        elif date_leave == "Second Half":
                            badge_class = "badge-secondhalf"
                        elif date_leave == "Emergency Leave":
                            badge_class = "badge-emergency"
                        
                        leave_badge = f'<span class="leave-badge {badge_class}">{date_leave[:2]}</span>'
                        display_df.at[champ_idx[0], day] = leave_badge
                    elif display_df.at[champ_idx[0], day] == "":
                        leave_badges = []
                        if leave_info.get('sick_leave', 0):
                            leave_badges.append('<span class="leave-badge badge-sl">SL</span>')
                        if leave_info.get('casual_leave', 0):
                            leave_badges.append('<span class="leave-badge badge-cl">CL</span>')
                        if leave_info.get('period_leave', 0):
                            leave_badges.append('<span class="leave-badge badge-pl">PL</span>')
                        if leave_info.get('annual_leave', 0):
                            leave_badges.append('<span class="leave-badge badge-al">AL</span>')
                        if leave_info.get('comp_off', 0):
                            leave_badges.append('<span class="leave-badge badge-co">CO</span>')
                        if leave_info.get('maternity_leave', 0):
                            leave_badges.append('<span class="leave-badge badge-maternity">MT</span>')
                        
                        if leave_badges:
                            display_df.at[champ_idx[0], day] = " ".join(leave_badges)

        return display_df

    def calculate_late_hour_coverage(self, roster_df):
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
                'status': '✅ Target Met' if expected_al >= self.MIN_AL else '❌ Below Target'
            }

        return validation_results

# Initialize session state
def initialize_session_state():
    if 'manual_splits' not in st.session_state:
        st.session_state.manual_splits = []
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'roster_df' not in st.session_state:
        st.session_state.roster_df = None
    if 'week_offs' not in st.session_state:
        st.session_state.week_offs = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'answer_rate' not in st.session_state:
        st.session_state.answer_rate = None
    if 'daily_rates' not in st.session_state:
        st.session_state.daily_rates = None
    if 'hourly_al_results' not in st.session_state:
        st.session_state.hourly_al_results = None
    if 'late_hour_coverage' not in st.session_state:
        st.session_state.late_hour_coverage = None
    if 'formatted_roster' not in st.session_state:
        st.session_state.formatted_roster = None
    if 'active_split_champs' not in st.session_state:
        st.session_state.active_split_champs = 4
    if 'leave_data' not in st.session_state:
        st.session_state.leave_data = {}
    if 'selected_languages' not in st.session_state:
        st.session_state.selected_languages = []

# Main application
def main():
    st.markdown('<h1 class="main-header">📞 Call Center Roster Optimizer</h1>', unsafe_allow_html=True)
    
    initialize_session_state()
    
    optimizer = CallCenterRosterOptimizer()

    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Configuration")

        st.subheader("Target Answer Level")
        st.info(f"Current target: {optimizer.TARGET_AL}% (Minimum: {optimizer.MIN_AL}%)")

        st.subheader("Language Filter")
        all_languages = optimizer.get_available_languages()
        selected_languages = st.multiselect(
            "Filter by Language",
            options=all_languages,
            default=st.session_state.selected_languages,
            help="Select languages to include in the roster"
        )
        st.session_state.selected_languages = selected_languages

        st.subheader("Active Split Shift Champions")
        active_split_champs = st.slider(
            "Minimum Active Split Shift Champions per Day",
            min_value=3,
            max_value=8,
            value=4,
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
            type=['xlsx'],
            help="Upload your filled-in call volume data Excel file (XLSX format only)"
        )
        
        st.info("If you have XLS files, please convert them to XLSX format.")
        
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔧 Manual Split Shift Assignment")

        st.markdown('<div class="split-shift-option">', unsafe_allow_html=True)
        st.subheader("Assign Specific Split Shifts")

        available_champs = [champ["name"] for champ in optimizer.champions if champ["can_split"]]
        selected_champ = st.selectbox("Select Champion", available_champs)

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day = st.selectbox("Select Day", days)

        pattern_options = [f"{pattern['name']} ({pattern['display']})" 
                        for pattern in optimizer.shift_patterns if pattern['type'] == 'split']
        selected_pattern_idx = st.selectbox("Select Split Shift Pattern", range(len(pattern_options)), format_func=lambda x: pattern_options[x])

        if st.button("➕ Add Manual Split Assignment"):
            new_assignment = {
                'champion': selected_champ,
                'day': selected_day,
                'pattern': [p for p in optimizer.shift_patterns if p['type'] == 'split'][selected_pattern_idx]
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
        st.header("ℹ️ Champion Information")
        
        st.subheader("Total Champions")
        total_champs = len(optimizer.champions)
        active_champs = len([c for c in optimizer.champions if c['status'] == 'Active'])
        maternity_champs = len([c for c in optimizer.champions if c['status'] == 'Maternity'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", total_champs)
        with col2:
            st.metric("Active", active_champs)
        with col3:
            st.metric("Maternity", maternity_champs)
        
        st.subheader("Language Distribution")
        lang_dist = {}

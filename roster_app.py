import streamlit as st
import pandas as pd
import numpy as np
import pulpneği
from datetime import datetime, timedelta
import random
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
    padding-bottom>    0.5rem;
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
        self.champions = []
        self.AVERAGE_HANDLING_TIME_SECONDS = 202  # 3 minutes 22 seconds
        self.TARGET_AL = 95  # Updated to 95% as per requirement
        self.split_shift_patterns = [
            {"name": "7-12 & 4-9", "times": (7, 12, 16, 21), "display": "07:00 to 12:00 & 16:30 to 21:00"},
            {"name": "8-1 & 5-9", "times": (8, 13, 17, 21), "display": "08:00 to 13:00 & 17:00 to 21:00"},
            {"name": "10-3 & 5-9", "times": (10, 15, 17, 21), "display": "10:00 to 15:00 & 17:00 to 21:00"},
            {"name": "12-9", "times": (12, 21), "display": "12:00 to 21:00"},
            {"name": "11-8", "times": (11, 20), "display": "11:00 to 20:00"},
        ]
        self.leaves = {}  # Store leave data {champion: {day: leave_type}}

    def load_champions(self, uploaded_file=None):
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                required_columns = ['name', 'primary_lang', 'calls_per_hour', 'can_split']
                if not all(col in df.columns for col in required_columns):
                    st.error("Excel file must contain columns: name, primary_lang, calls_per_hour, can_split")
                    return self.get_default_champions()
                
                self.champions = []
                for _, row in df.iterrows():
                    secondary_langs = row.get('secondary_langs', '')
                    if isinstance(secondary_langs, str):
                        secondary_langs = [lang.strip() for lang in secondary_langs.split(',') if lang.strip()]
                    else:
                        secondary_langs = []
                    self.champions.append({
                        'name': str(row['name']),
                        'primary_lang': str(row['primary_lang']),
                        'secondary_langs': secondary_langs,
                        'calls_per_hour': float(row['calls_per_hour']),
                        'can_split': bool(row['can_split'])
                    })
                return self.champions
            except Exception as e:
                st.error(f"Error reading champions file: {str(e)}")
                return self.get_default_champions()
        return self.get_default_champions()

    def get_default_champions(self):
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
            {"name": "Dundesh", "primary_lang": "ka", "secondary_langs": ["te", "hi"], "calls_per_hour": 12, "can_split": False},
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
        predicted_al = (answered_calls / forecasted_calls) * 100 if forecasted_calls > 0 else 100
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
        elif predicted_al >= 90:
            recommendation['status'] = '🟡 YELLOW (Close to Goal)'
            recommendation['recommendation'] = f'Close to target. Adding 1 agent could help secure {target_al}%.'
        elif predicted_al >= 80:
            recommendation['status'] = '🟠 ORANGE (At Risk)'
            recommendation['recommendation'] = f'Add {int(recommendation["agents_deficit"])} agent(s) to reach {target_al}%.'
        else:
            recommendation['status'] = '🔴 RED (Critical)'
            recommendation['recommendation'] = f'Immediately add {int(recommendation["agents_deficit"])} agent(s) to reach {target_al}%.'

        return recommendation

    def create_template_file(self, template_type='call_volume'):
        """Create a template Excel file for call volume or champions data."""
        if template_type == 'call_volume':
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
        else:  # champions template
            champs_data = pd.DataFrame({
                'name': ['Agent Name'],
                'primary_lang': ['Primary Language (e.g., ka, hi, te)'],
                'secondary_langs': ['Secondary Languages (comma-separated, e.g., hi,te,ta)'],
                'calls_per_hour': [0],
                'can_split': [False]
            })
            instructions = pd.DataFrame({
                'Instruction': [
                    'INSTRUCTIONS:',
                    '1. Fill in champion details in the Champions sheet',
                    '2. Columns required: name, primary_lang, secondary_langs, calls_per_hour, can_split',
                    '3. name: Champion’s name (text)',
                    '4. primary_lang: Primary language code (e.g., ka, hi, te)',
                    '5. secondary_langs: Comma-separated list of secondary language codes',
                    '6. calls_per_hour: Number of calls the champion can handle per hour (number)',
                    '7. can_split: Whether the champion can work split shifts (True/False)',
                    '8. Save the file and upload it back to the app'
                ]
            })
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                instructions.to_excel(writer, sheet_name='Instructions', index=False)
                champs_data.to_excel(writer, sheet_name='Champions', index=False)
        
        return excel_buffer.getvalue()

    def analyze_excel_data(self, uploaded_file):
        """Analyze uploaded Excel file for call volume."""
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
                    peak_hours = [11, 12, 13, 14] if 'Peak_Hour' not in df.columns else df['Peak_Hour'].mode()[0]
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
        """Return sample call volume data."""
        return {
            'hourly_volume': {
                7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
                13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
                18: 224.9, 19: 179.3, 20: 113.9, 21: 0
            },
            'peak_hours': [11, 12, 13, 14],
            'total_daily_calls': 2975
        }

    def apply_leave(self, roster_df):
        """Apply leave data to roster."""
        if not self.leaves:
            return roster_df
        updated_roster = roster_df.copy()
        for champ, leave_days in self.leaves.items():
            for day, leave_type in leave_days.items():
                updated_roster = updated_roster[
                    ~((updated_roster['Champion'] == champ) & (updated_roster['Day'] == day))
                ]
                updated_roster = updated_roster.append({
                    'Day': day,
                    'Champion': champ,
                    'Primary Language': '',
                    'Secondary Languages': '',
                    'Shift Type': leave_type,
                    'Start Time': 'N/A',
                    'End Time': 'N/A',
                    'Duration': 'N/A',
                    'Calls/Hour Capacity': 0,
                    'Can Split': False
                }, ignore_index=True)
        return updated_roster

    def generate_daily_roster(self, day, champions, straight_shifts, split_shifts, analysis_data, manual_splits=None):
        """Generate roster for a single day with specific shift requirements."""
        daily_roster = []
        available_champions = [c for c in champions if day not in self.leaves.get(c['name'], {})]

        # Calculate required agents for 95% AL
        total_calls = analysis_data['total_daily_calls']
        agents_needed = int(np.ceil(self.agents_needed_for_target(total_calls, self.TARGET_AL, self.AVERAGE_HANDLING_TIME_SECONDS)))
        straight_shifts = min(straight_shifts, len(available_champions))
        split_shifts = min(split_shifts, len(available_champions) - straight_shifts)

        # Straight shifts
        straight_start_times = [7, 8, 9, 10, 11]
        straight_counts = [max(1, straight_shifts // len(straight_start_times))] * len(straight_start_times)
        straight_idx = 0
        for i, champ in enumerate(available_champions[:straight_shifts]):
            while straight_counts[straight_idx % len(straight_counts)] <= 0:
                straight_idx += 1
            start_time = straight_start_times[straight_idx % len(straight_start_times)]
            straight_counts[straight_idx % len(straight_counts)] -= 1
            straight_idx += 1
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

        # Split shifts
        split_champs = available_champions[straight_shifts:straight_shifts + split_shifts]
        if manual_splits:
            for assignment in manual_splits:
                if assignment['day'] == day:
                    champ = next((c for c in split_champs if c['name'] == assignment['champion']), None)
                    if champ:
                        pattern = assignment['pattern']
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
                        split_champs.remove(champ)
        
        # Assign remaining split shifts
        for i, champ in enumerate(split_champs):
            pattern = self.split_shift_patterns[i % len(self.split_shift_patterns)]
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

    def generate_roster(self, straight_shifts, split_shifts, analysis_data, manual_splits=None):
        """Generate optimized roster based on shift preferences and leaves."""
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
            roster_df = self.apply_leave(roster_df)
            roster_df, week_offs = self.assign_weekly_offs(roster_df, max_offs_per_day=4, min_split_champs=4)

            return roster_df, week_offs
        except Exception as e:
            st.error(f"Error generating roster: {str(e)}")
            return None, None

    def apply_special_rules(self, roster_df):
        """Apply special rules like Revathi always on split shift."""
        revathi_mask = roster_df['Champion'] == 'Revathi'
        roster_df.loc[revathi_mask, 'Shift Type'] = 'Split'
        for idx in roster_df[revathi_mask].index:
            pattern = self.split_shift_patterns[0]
            roster_df.at[idx, 'Start Time'] = pattern['display']
            roster_df.at[idx, 'End Time'] = f"{pattern['times'][3]:02d}:00"
            roster_df.at[idx, 'Duration'] = '9 hours (with break)'
        return roster_df

    def assign_weekly_offs(self, roster_df, max_offs_per_day=4, min_split_champs=4):
        """Assign weekly off days to champions with limit per day."""
        week_offs = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        champs = roster_df['Champion'].unique()
        offs_per_day = {day: 0 for day in days}
        
        for champ in champs:
            if champ not in self.leaves:
                available_days = [d for d in days if offs_per_day[d] < max_offs_per_day]
                if available_days:
                    day_off = random.choice(available_days)
                    week_offs[champ] = day_off
                    offs_per_day[day_off] += 1
                    roster_df = roster_df[~((roster_df['Champion'] == champ) & (roster_df['Day'] == day_off))]
                else:
                    week_offs[champ] = 'No day off'
        
        return roster_df, week_offs

    def calculate_coverage(self, roster_df, analysis_data):
        """Calculate coverage metrics."""
        if roster_df is None or analysis_data is None:
            return None

        total_capacity = 0
        for _, row in roster_df.iterrows():
            if row['Shift Type'] in ['SL', 'PL', 'CO']:
                continue
            if row['Shift Type'] == 'Straight':
                hours_worked = 9
            else:
                shifts = row['Start Time'].split(' & ')
                hours_worked = sum(int(t.split(' to ')[1].split(':')[0]) - int(t.split(' to ')[0].split(':')[0]) for t in shifts)
            total_capacity += row['Calls/Hour Capacity'] * hours_worked

        required_capacity = analysis_data['total_daily_calls'] * 7 * 1.05  # 5% buffer for 95% AL
        utilization_rate = min(100, (required_capacity / total_capacity) * 100) if total_capacity > 0 else 0
        expected_answer_rate = min(100, (total_capacity / (analysis_data['total_daily_calls'] * 7)) * 100) if analysis_data['total_daily_calls'] > 0 else 0

        return {
            'total_capacity': total_capacity,
            'required_capacity': required_capacity,
            'utilization_rate': utilization_rate,
            'expected_answer_rate': expected_answer_rate
        }

    def calculate_hourly_al_analysis(self, roster_df, analysis_data):
        """Calculate AL predictions for each hour of each day."""
        if roster_df is None or analysis_data is None:
            return None
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hourly_al_results = {}
        for day in days:
            daily_roster = roster_df[roster_df['Day'] == day]
            for hour in self.operation_hours:
                if hour not in analysis_data['hourly_volume']:
                    continue
                agents_at_hour = sum(1 for _, row in daily_roster.iterrows() if self.is_agent_working_at_hour(row, hour))
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
        """Check if an agent is working at a specific hour based on their shift."""
        try:
            if row['Shift Type'] in ['SL', 'PL', 'CO']:
                return False
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
        """Get status classification for AL value."""
        if al_value >= self.TARGET_AL:
            return "✅ GOOD"
        elif al_value >= 90:
            return "🟡 WARNING"
        elif al_value >= 80:
            return "🟠 AT RISK"
        else:
            return "🔴 CRITICAL"

    def format_roster_for_display(self, roster_df, week_offs):
        """Format roster for display with leaves."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        champs = roster_df['Champion'].unique()
        display_df = pd.DataFrame({'Name': champs})
        display_df['Shift'] = ""
        for day in days:
            display_df[day] = ""
        
        for _, row in roster_df.iterrows():
            champ_name = row['Champion']
            day = row['Day']
            if row['Shift Type'] in ['SL', 'PL', 'CO']:
                shift_display = row['Shift Type']
            else:
                shift_display = row['Start Time']
            champ_idx = display_df[display_df['Name'] == champ_name].index
            if len(champ_idx) > 0:
                display_df.at[champ_idx[0], day] = shift_display
                if display_df.at[champ_idx[0], 'Shift'] == "":
                    display_df.at[champ_idx[0], 'Shift'] = row['Start Time'] if row['Shift Type'] not in ['SL', 'PL', 'CO'] else 'N/A'
        
        for champ, off_day in week_offs.items():
            if off_day in days:
                champ_idx = display_df[display_df['Name'] == champ].index
                if len(champ_idx) > 0:
                    display_df.at[champ_idx[0], off_day] = "WO"
        
        return display_df

    def show_editable_roster(self, roster_df, week_offs):
        """Display an editable roster table with week offs and leaves."""
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
                    options=["Straight", "Split", "SL", "PL", "CO"],
                    required=True
                ),
                "Start Time": st.column_config.TextColumn(
                    "Start Time",
                    help="Format: HH:MM to HH:MM or HH:MM to HH:MM & HH:MM to HH:MM for split shifts; N/A for leaves"
                ),
                "End Time": st.column_config.TextColumn(
                    "End Time"
                )
            },
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True
        )

        week_off_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "No day off"]
        week_off_editor_data = [{'Champion': champ, 'Current Day Off': off_day} for champ, off_day in week_offs.items()]
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
                    options=week_off_days,
                    required=True
                )
            },
            hide_index=True,
            use_container_width=True
        )

        new_week_offs = {row['Champion']: row['Current Day Off'] for _, row in edited_week_offs.iterrows() if row['Current Day Off'] != "No day off"}
        return edited_df, new_week_offs

    def manage_leaves(self):
        """Manage leave assignments."""
        st.subheader("📅 Leave Management")
        leave_types = ['SL', 'PL', 'CO']
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        champs = [champ['name'] for champ in self.champions]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_champ = st.selectbox("Select Champion", champs, key="leave_champ")
        with col2:
            selected_day = st.selectbox("Select Day", days, key="leave_day")
        with col3:
            leave_type = st.selectbox("Leave Type", leave_types, key="leave_type")
        
        if st.button("Add Leave"):
            if selected_champ not in self.leaves:
                self.leaves[selected_champ] = {}
            self.leaves[selected_champ][selected_day] = leave_type
            st.success(f"Added {leave_type} leave for {selected_champ} on {selected_day}")
        
        if self.leaves:
            st.subheader("Current Leaves")
            leave_data = []
            for champ, leaves in self.leaves.items():
                for day, ltype in leaves.items():
                    leave_data.append({'Champion': champ, 'Day': day, 'Leave Type': ltype})
            leave_df = pd.DataFrame(leave_data)
            st.dataframe(leave_df, use_container_width=True)
            
            if st.button("Clear All Leaves"):
                self.leaves = {}
                st.success("All leaves cleared.")
                st.rerun()

def initialize_session_state():
    """Initialize session state variables."""
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
    if 'formatted_roster' not in st.session_state:
        st.session_state.formatted_roster = None
    if 'active_split_champs' not in st.session_state:
        st.session_state.active_split_champs = 4

def main():
    st.markdown('<h1 class="main-header">📞 Call Center Roster Optimizer</h1>', unsafe_allow_html=True)
    initialize_session_state()
    optimizer = CallCenterRosterOptimizer()

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Configuration")
        
        st.subheader("Champions Data")
        champs_file = st.file_uploader("Upload Champions Data (Excel)", type=['xlsx', 'xls'])
        if champs_file:
            optimizer.champions = optimizer.load_champions(champs_file)
            st.success("✅ Champions data uploaded successfully!")
        
        st.markdown('<div class="template-download">', unsafe_allow_html=True)
        st.subheader("Download Champions Template")
        champs_template = optimizer.create_template_file(template_type='champions')
        st.download_button(
            "📥 Download Champions Template",
            champs_template,
            "champions_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Shift Allocation")
        total_champs = len(optimizer.champions)
        straight_shifts = st.slider(
            "Regular Shifts (9 hours continuous)", 
            min_value=0, 
            max_value=total_champs, 
            value=max(0, total_champs - 5),
            help="Number of champions working straight 9-hour shifts"
        )
        split_shifts = st.slider(
            "Split Shifts (with break)", 
            min_value=0, 
            max_value=total_champs, 
            value=min(5, total_champs),
            help="Number of champions working split shifts"
        )
        if straight_shifts + split_shifts > total_champs:
            st.warning(f"⚠️ Only {total_champs} champions available! Reducing split shifts...")
            split_shifts = total_champs - straight_shifts
        
        st.metric("Total Champions Used", f"{straight_shifts + split_shifts}/{total_champs}")
        st.session_state.active_split_champs = st.slider(
            "Minimum Active Split Shift Champions per Day",
            min_value=4,
            max_value=split_shifts,
            value=min(4, split_shifts)
        )

        st.markdown('<div class="template-download">', unsafe_allow_html=True)
        st.subheader("Call Volume Data")
        uploaded_file = st.file_uploader("Upload Call Volume Data", type=['xlsx', 'xls'])
        st.download_button(
            "📥 Download Call Volume Template",
            optimizer.create_template_file(),
            "call_volume_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="split-shift-option">', unsafe_allow_html=True)
        st.header("🔧 Manual Split Shift Assignment")
        available_champs = [champ["name"] for champ in optimizer.champions if champ["can_split"]]
        selected_champ = st.selectbox("Select Champion", available_champs)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_day = st.selectbox("Select Day", days)
        pattern_options = [f"{p['name']} ({p['display']})" for p in optimizer.split_shift_patterns]
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

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        optimizer.manage_leaves()
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("📈 Configuration")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Regular Shifts", straight_shifts)
        st.metric("Split Shifts", split_shifts)
        st.metric("Total Coverage", f"{straight_shifts + split_shifts} champions")
        if uploaded_file:
            st.session_state.analysis_data = optimizer.analyze_excel_data(uploaded_file)
            st.success("✅ Call volume data uploaded successfully!")
            st.metric("Daily Call Volume", f"{st.session_state.analysis_data['total_daily_calls']:,.0f}")
            st.metric("Weekly Call Volume", f"{st.session_state.analysis_data['total_daily_calls'] * 7:,.0f}")
            st.metric("Peak Hours", ", ".join([f"{h}:00" for h in st.session_state.analysis_data['peak_hours']]))
        else:
            st.session_state.analysis_data = optimizer.get_sample_data()
            st.metric("Estimated Daily Calls", "2,975")
            st.metric("Estimated Weekly Calls", "20,825")
        st.markdown('</div>', unsafe_allow_html=True)

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
                if roster_df is not None:
                    st.session_state.roster_df = roster_df
                    st.session_state.week_offs = week_offs
                    st.session_state.metrics = optimizer.calculate_coverage(roster_df, st.session_state.analysis_data)
                    st.session_state.answer_rate = optimizer.calculate_answer_rate(roster_df, st.session_state.analysis_data)
                    st.session_state.daily_rates = optimizer.calculate_daily_answer_rates(roster_df, st.session_state.analysis_data)
                    st.session_state.hourly_al_results = optimizer.calculate_hourly_al_analysis(roster_df, st.session_state.analysis_data)
                    st.session_state.formatted_roster = optimizer.format_roster_for_display(roster_df, week_offs)
                    st.success("✅ Roster generated successfully!")
                else:
                    st.error("❌ Failed to generate roster. Please check your settings.")

    # Display results
    if st.session_state.roster_df is not None and st.session_state.metrics is not None:
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

        st.markdown('<div class="section-header"><h2>📅 Roster Schedule</h2></div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.formatted_roster, use_container_width=True)

        st.markdown('<div class="section-header"><h2>🛠️ Manual Adjustments</h2></div>', unsafe_allow_html=True)
        if st.checkbox("Enable manual editing of roster"):
            edited_roster, edited_week_offs = optimizer.show_editable_roster(st.session_state.roster_df, st.session_state.week_offs)
            st.session_state.roster_df = edited_roster
            st.session_state.week_offs = edited_week_offs
            if st.button("Update Metrics"):
                st.session_state.metrics = optimizer.calculate_coverage(edited_roster, st.session_state.analysis_data)
                st.session_state.answer_rate = optimizer.calculate_answer_rate(edited_roster, st.session_state.analysis_data)
                st.session_state.daily_rates = optimizer.calculate_daily_answer_rates(edited_roster, st.session_state.analysis_data)
                st.session_state.hourly_al_results = optimizer.calculate_hourly_al_analysis(edited_roster, st.session_state.analysis_data)
                st.session_state.formatted_roster = optimizer.format_roster_for_display(edited_roster, edited_week_offs)
                st.success("Metrics updated!")

        st.markdown('<div class="section-header"><h2>💾 Download Roster</h2></div>', unsafe_allow_html=True)
        csv = st.session_state.formatted_roster.to_csv(index=False)
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            st.session_state.formatted_roster.to_excel(writer, index=False, sheet_name='Roster')
        excel_data = excel_buffer.getvalue()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download as CSV", csv, "roster.csv", "text/csv", use_container_width=True)
        with col2:
            st.download_button(
                "📥 Download as Excel",
                excel_data,
                "roster.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

if __name__ == "__main__":
    main()

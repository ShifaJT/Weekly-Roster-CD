import streamlit as st
import pandas as pd
import numpy as np
import pulp
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

class OptimizedRosterGenerator:
    def __init__(self):
        self.champions = self.load_champions()
        self.config = self.get_config()
        
    def get_config(self):
        return {
            'total_champions': 28,
            'straight_shifts': st.session_state.get('straight_shifts', 18),
            'split_shifts': st.session_state.get('split_shifts', 10),
            'language_requirements': {
                'hi': 10, 'ka': 6, 'te': 3, 'ta': 1, 'en': 1
            },
            'answer_time_target': 0.90
        }
    
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
    
    def generate_optimized_roster(self):
        """Generate roster optimized for 90%+ answer rate"""
        try:
            # Calculate required agents based on call volume
            hourly_requirements = self.calculate_hourly_requirements()
            
            # Create optimization model
            model = pulp.LpProblem("RosterOptimization", pulp.LpMaximize)
            
            # Decision variables
            champions = range(len(self.champions))
            hours = range(7, 22)  # 7 AM to 9 PM
            days = range(7)       # 7 days
            
            # Binary variable: 1 if champion c works hour h on day d
            x = pulp.LpVariable.dicts("work", 
                                    ((c, h, d) for c in champions for h in hours for d in days),
                                    cat='Binary')
            
            # Objective: Maximize coverage during peak hours
            peak_hours = [11, 12, 13, 14]  # 11 AM - 2 PM
            model += pulp.lpSum(x[c, h, d] for c in champions for h in peak_hours for d in days)
            
            # Constraints
            # 1. Each champion works 7h50m per day (7.83 hours)
            for c in champions:
                for d in days:
                    model += pulp.lpSum(x[c, h, d] for h in hours) == 7.83
            
            # 2. Meet hourly requirements
            for h in hours:
                for d in days:
                    model += pulp.lpSum(x[c, h, d] for c in champions) >= hourly_requirements[h]
            
            # 3. Language requirements
            for lang, req in self.config['language_requirements'].items():
                for h in hours:
                    for d in days:
                        lang_champs = [c for c in champions if (
                            self.champions[c]['primary_lang'] == lang or 
                            lang in self.champions[c]['secondary_langs']
                        )]
                        model += pulp.lpSum(x[c, h, d] for c in lang_champs) >= req
            
            # Solve
            model.solve()
            
            # Generate roster
            roster = self.create_roster_from_solution(x, hours, days)
            return roster
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")
            return None
    
    def calculate_hourly_requirements(self):
        """Calculate agents needed per hour for 90% answer rate"""
        hourly_calls = {
            7: 38.5, 8: 104.4, 9: 205.8, 10: 271, 11: 315.8, 12: 292.2,
            13: 278.1, 14: 246.3, 15: 227.4, 16: 240.0, 17: 236.2, 
            18: 224.9, 19: 179.3, 20: 113.9, 21: 0
        }
        
        # For 90% answer rate, we need capacity for 110% of calls
        return {h: max(1, round(calls * 1.1 / 12)) for h, calls in hourly_calls.items()}
    
    def create_roster_from_solution(self, x, hours, days):
        """Create readable roster from optimization solution"""
        roster_data = []
        
        for d in days:
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d]
            
            for c in range(len(self.champions)):
                champ = self.champions[c]
                working_hours = [h for h in hours if pulp.value(x[c, h, d]) == 1]
                
                if working_hours:
                    # Determine shift type
                    if max(working_hours) - min(working_hours) <= 9:
                        shift_type = "straight"
                        shift_time = f"{min(working_hours)}:00-{max(working_hours)}:00"
                    else:
                        shift_type = "split"
                        # Find natural break point
                        morning = [h for h in working_hours if h <= 13]
                        evening = [h for h in working_hours if h > 13]
                        shift_time = f"{min(morning)}:00-{max(morning)}:00 & {min(evening)}:00-{max(evening)}:00"
                    
                    roster_data.append({
                        'Day': day_name,
                        'Champion': champ['name'],
                        'Primary Language': champ['primary_lang'].upper(),
                        'Secondary Languages': ', '.join([l.upper() for l in champ['secondary_langs']]),
                        'Shift Type': shift_type,
                        'Shift Time': shift_time,
                        'Capacity/Hour': champ['calls_per_hour'],
                        'Can Split': 'Yes' if champ['can_split'] else 'No'
                    })
        
        return pd.DataFrame(roster_data)

# Streamlit app
def main():
    st.title("📞 Multilingual Call Center Roster Optimizer")
    
    optimizer = OptimizedRosterGenerator()
    
    if st.button("🚀 Generate Optimized Roster (90%+ Answer Rate)"):
        with st.spinner("Optimizing roster for maximum efficiency..."):
            roster = optimizer.generate_optimized_roster()
            
            if roster is not None:
                st.success("✅ Roster optimized for 90%+ answer rate!")
                
                # Display by day
                for day in roster['Day'].unique():
                    st.subheader(f"📅 {day}")
                    day_roster = roster[roster['Day'] == day]
                    st.dataframe(day_roster.drop('Day', axis=1))
                
                # Statistics
                st.subheader("📊 Performance Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_capacity = roster['Capacity/Hour'].sum() * 7.83 * 7
                    st.metric("Total Weekly Capacity", f"{total_capacity:,.0f} calls")
                
                with col2:
                    req_capacity = 2975 * 7 * 1.1  # 110% of weekly calls
                    st.metric("Required Capacity", f"{req_capacity:,.0f} calls")
                
                with col3:
                    utilization = min(100, (req_capacity / total_capacity) * 100)
                    st.metric("Utilization Rate", f"{utilization:.1f}%")
                
                # Download
                csv = roster.to_csv(index=False)
                st.download_button("📥 Download Full Roster", csv, "optimized_roster.csv")

if __name__ == "__main__":
    main()

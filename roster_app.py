import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta

# --------------------
# CONFIG
# --------------------
st.set_page_config(page_title="Weekly Roster Generator", layout="wide")

OPERATING_HOURS = list(range(7, 21))  # 07:00 to 20:00
SHIFT_LABELS = {
    'straight': [
        (7, 16), (8, 17), (9, 18), (10, 19), (11, 20), (12, 21)
    ],
    'split': [
        ((7, 11), (16.5, 21)),
        ((8, 12), (16.5, 21)),
        ((9, 13), (16.5, 21)),
        ((10, 14), (16.5, 21))
    ]
}
LEAVE_TYPES = ["", "Special Leave", "Casual Leave", "Planned Leave", "Sick Leave", "Comp Off"]

# --------------------
# FUNCTIONS
# --------------------
def create_champions_template():
    buf = io.BytesIO()
    df = pd.DataFrame({
        "name": ["Example1", "Example2"],
        "primary_lang": ["ka", "hi"],
        "secondary_langs": ["te,hi", "ka"],
        "calls_per_hour": [12, 10],
        "can_split": [True, False]
    })
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Champions")
    buf.seek(0)
    return buf

def create_hourly_calls_template():
    buf = io.BytesIO()
    df = pd.DataFrame({
        "Date": ["2025-08-01", "2025-08-01", "2025-08-01", "2025-08-01"],
        "Hour": [7, 8, 9, 10],
        "Calls": [38, 109, 184, 278]
    })
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Hourly_Data")
    buf.seek(0)
    return buf

def prepare_weekday_patterns(hourly_df):
    hourly_df['Date'] = pd.to_datetime(hourly_df['Date'])
    hourly_df['Weekday'] = hourly_df['Date'].dt.day_name()
    pattern = hourly_df.groupby(['Weekday', 'Hour'])['Calls'].mean().reset_index()
    return pattern

def predict_next_week_calls(pattern):
    today = datetime.today()
    next_week_dates = [today + timedelta(days=i) for i in range(1, 8)]
    
    predictions = []
    for date in next_week_dates:
        weekday = date.strftime("%A")
        for hour in OPERATING_HOURS:
            row = pattern[(pattern['Weekday'] == weekday) & (pattern['Hour'] == hour)]
            calls = row['Calls'].values[0] if not row.empty else 0
            predictions.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Hour": hour,
                "Calls": round(calls)
            })
    return pd.DataFrame(predictions)

def assign_shifts(champions, predicted_calls):
    roster = []
    unique_dates = predicted_calls['Date'].unique()
    
    for date in unique_dates:
        day_data = predicted_calls[predicted_calls['Date'] == date]
        total_calls = day_data['Calls'].sum()
        avg_calls_per_hour = champions['calls_per_hour'].mean()
        required_agents = int(np.ceil(total_calls / (avg_calls_per_hour * len(OPERATING_HOURS))))
        
        champ_idx = 0
        for i in range(required_agents):
            champ = champions.iloc[champ_idx % len(champions)]
            shift_type, shift_label = get_shift_label(champ)
            roster.append({
                "Date": date,
                "Champion": champ['name'],
                "Shift": shift_label,
                "Leave": ""
            })
            champ_idx += 1
    return pd.DataFrame(roster)

def get_shift_label(champ):
    if champ['can_split']:
        s1, s2 = SHIFT_LABELS['split'][0]
        return "split", f"Split {format_time(s1[0])}-{format_time(s1[1])} & {format_time(s2[0])}-{format_time(s2[1])}"
    else:
        s = SHIFT_LABELS['straight'][0]
        return "straight", f"Straight {format_time(s[0])}-{format_time(s[1])}"

def format_time(hour):
    h = int(hour)
    m = int((hour - h) * 60)
    return f"{h:02d}:{m:02d}"

def calculate_al_prediction(roster_df, predicted_calls):
    results = []
    for date in predicted_calls['Date'].unique():
        calls = predicted_calls[predicted_calls['Date'] == date]['Calls'].sum()
        champs_on_duty = len(roster_df[roster_df['Date'] == date])
        avg_calls_per_agent = 470 / 300  # 5-min AHT assumption
        capacity = champs_on_duty * avg_calls_per_agent
        al_percent = (capacity / calls) * 100 if calls > 0 else 0
        results.append({
            "Date": date,
            "Total Calls": calls,
            "Champs": champs_on_duty,
            "Predicted AL%": round(al_percent, 2)
        })
    summary_df = pd.DataFrame(results)
    overall_al = round(summary_df['Predicted AL%'].mean(), 2)
    return summary_df, overall_al

def download_excel(df, file_name="roster.xlsx"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Roster")
    buf.seek(0)
    return buf

# --------------------
# UI
# --------------------
st.title("📅 Weekly Roster Generator (Next Week Prediction)")

with st.sidebar:
    st.header("📂 Upload Files")
    champions_file = st.file_uploader("Upload Champions File", type=["xlsx"])
    calls_file = st.file_uploader("Upload Past 30 Days Hourly Calls File", type=["xlsx"])
    
    st.subheader("⬇ Download Templates")
    st.download_button("Download Champions Template", create_champions_template(), file_name="champions_template.xlsx")
    st.download_button("Download Hourly Calls Template", create_hourly_calls_template(), file_name="hourly_calls_template.xlsx")

if champions_file and calls_file:
    champions_df = pd.read_excel(champions_file)
    hourly_df = pd.read_excel(calls_file)
    
    st.success("✅ Files uploaded successfully!")
    
    with st.expander("View Champions Data"):
        st.dataframe(champions_df)
    with st.expander("View Past Hourly Calls Data"):
        st.dataframe(hourly_df)
    
    # Compute weekday patterns and predict next week
    st.subheader("📈 Predicting Next Week Calls Based on Past 30 Days...")
    pattern = prepare_weekday_patterns(hourly_df)
    predicted_calls = predict_next_week_calls(pattern)
    
    with st.expander("View Predicted Hourly Calls for Next Week"):
        st.dataframe(predicted_calls)
    
    # Generate roster
    st.subheader("✅ Generated Weekly Roster (Next Week)")
    roster_df = assign_shifts(champions_df, predicted_calls)
    
    edited_roster = st.data_editor(
        roster_df,
        num_rows="dynamic",
        use_container_width=True,
        key="editable_roster",
        column_config={
            "Leave": st.column_config.SelectboxColumn("Leave", options=LEAVE_TYPES)
        }
    )
    
    # AL Prediction
    st.subheader("📊 AL Prediction Summary")
    al_df, overall_al = calculate_al_prediction(edited_roster, predicted_calls)
    st.dataframe(al_df, use_container_width=True)
    st.metric("Overall Weekly AL%", f"{overall_al}%")
    
    # Download Roster
    st.download_button(
        "⬇ Download Roster (Excel)",
        download_excel(edited_roster),
        file_name="weekly_roster.xlsx"
    )

else:
    st.info("Please upload both Champions file and Past Hourly Calls file to generate the roster.")

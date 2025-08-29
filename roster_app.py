import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import datetime as dt

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Weekly Roster Planner", layout="wide")
st.title("📅 Weekly Roster Planner with AL% Target and Leaves")

# Default values
SHIFT_PATTERNS = [
    "07:00 to 16:00",
    "08:00 to 17:00",
    "09:00 to 18:00",
    "10:00 to 19:00",
    "11:00 to 20:00",
    "12:00 to 21:00"
]

SPLIT_SHIFTS = [
    "07:00 to 11:00 & 16:30 to 21:00",
    "08:00 to 12:00 & 16:30 to 21:00",
    "09:00 to 13:00 & 16:30 to 21:00"
]

ACTIVE_MINUTES = 470  # 7h50m active
AHT = 5  # Average Handling Time in minutes

# ---------------- Sidebar Controls ---------------- #
target_al = st.sidebar.slider("🎯 Target AL %", 80, 100, 95)
max_split = st.sidebar.number_input("Max Split-Shift Champs", min_value=0, value=3)
st.sidebar.write("Default Active Time: 7h50m | AHT: 5 mins")

# ---------------- Templates ---------------- #
def create_hourly_template():
    today = dt.date.today()
    start_date = today - dt.timedelta(days=30)
    dates = pd.date_range(start=start_date, end=today)
    rows = []
    for d in dates:
        for h in range(7, 21):
            rows.append([d.strftime("%Y-%m-%d"), h, 0])
    df = pd.DataFrame(rows, columns=["Date", "Hour", "Calls"])
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Hourly_Data")
    buf.seek(0)
    return buf

def create_champions_template():
    data = [
        ["Name", "Primary_Lang", "Secondary_Langs", "Can_Split(Yes/No)"],
        ["Revathi", "ka", "en", "No"],
        ["Pasang", "en", "hi", "Yes"]
    ]
    df = pd.DataFrame(data[1:], columns=data[0])
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Champions")
    buf.seek(0)
    return buf

st.sidebar.download_button("⬇ Download Hourly Calls Template", create_hourly_template(), file_name="hourly_calls_template.xlsx")
st.sidebar.download_button("⬇ Download Champions Template", create_champions_template(), file_name="champions_template.xlsx")

# ---------------- Upload Section ---------------- #
st.subheader("📤 Upload Required Files")
hourly_file = st.file_uploader("Upload Hourly Calls Excel", type=["xlsx"])
champ_file = st.file_uploader("Upload Champions Excel", type=["xlsx"])

if hourly_file and champ_file:
    # ---------------- Calls Data ---------------- #
    calls_df = pd.read_excel(hourly_file)
    if not {"Date", "Hour", "Calls"}.issubset(calls_df.columns):
        st.error("❌ Invalid hourly file format. Ensure columns: Date, Hour, Calls")
        st.stop()

    calls_df["Date"] = pd.to_datetime(calls_df["Date"], errors="coerce")

    # ---------------- Champions Data ---------------- #
    champs_df = pd.read_excel(champ_file)
    if not {"Name", "Primary_Lang", "Secondary_Langs", "Can_Split(Yes/No)"}.issubset(champs_df.columns):
        st.error("❌ Invalid champions file format.")
        st.stop()

    champions = champs_df.to_dict("records")
    st.write(f"✅ Loaded {len(champions)} champions")

    # ---------------- Forecast Next Week ---------------- #
    st.write("✅ Uploaded Calls Data Preview", calls_df.head())
    pattern = calls_df.copy()
    pattern["Weekday"] = pattern["Date"].dt.day_name()
    weekday_pattern = pattern.groupby(["Weekday", "Hour"], as_index=False)["Calls"].mean()

    # Create next week dates (Mon-Sun)
    today = dt.date.today()
    next_monday = today + dt.timedelta(days=(7 - today.weekday()))
    next_week = [next_monday + dt.timedelta(days=i) for i in range(7)]

    forecast_rows = []
    for d in next_week:
        wd = d.strftime("%A")
        for h in range(7, 21):
            call_val = weekday_pattern[(weekday_pattern["Weekday"] == wd) & (weekday_pattern["Hour"] == h)]["Calls"]
            forecast_rows.append([d, h, round(call_val.values[0] if not call_val.empty else 0)])
    forecast_df = pd.DataFrame(forecast_rows, columns=["Date", "Hour", "Calls"])

    st.subheader("📊 Forecasted Calls for Next Week")
    st.dataframe(forecast_df)

    # ---------------- Calculate Daily Total ---------------- #
    daily_calls = forecast_df.groupby("Date")["Calls"].sum().reset_index().rename(columns={"Calls": "Total_Calls"})

    # ---------------- Week-Off Assignment ---------------- #
    def assign_week_off(champ_list):
        week_off_plan = {}
        days = len(next_week)
        for i, champ in enumerate(champ_list):
            off_day = i % days
            week_off_plan[champ["Name"]] = next_week[off_day]
        return week_off_plan

    week_offs = assign_week_off(champions)

    # ---------------- Shift Assignment Logic ---------------- #
    def calculate_answered(champ_count):
        return (champ_count * ACTIVE_MINUTES) / AHT

    roster_data = []
    al_summary = []

    for idx, row in daily_calls.iterrows():
        date = row["Date"]
        total_calls = row["Total_Calls"]
        shifts = []
        split_assigned = 0

        while True:
            answered_calls = calculate_answered(len(shifts))
            achieved_al = (answered_calls / total_calls) * 100 if total_calls > 0 else 0
            if achieved_al >= target_al or len(shifts) >= len(champions):
                break
            if split_assigned < max_split and len(shifts) % 3 == 0:
                shifts.append(np.random.choice(SPLIT_SHIFTS))
                split_assigned += 1
            else:
                shifts.append(np.random.choice(SHIFT_PATTERNS))

        champs_for_day = champions[:len(shifts)]
        for champ, shift in zip(champs_for_day, shifts):
            if week_offs[champ["Name"]] == date:
                roster_data.append({"Name": champ["Name"], "Date": date, "Shift": "WO"})
            else:
                roster_data.append({"Name": champ["Name"], "Date": date, "Shift": shift})

        al_summary.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Total Calls": total_calls,
            "Answered Calls": int(calculate_answered(len(shifts))),
            "Champs Assigned": len(shifts),
            "Target AL%": target_al,
            "Achieved AL%": round(achieved_al, 2)
        })

    roster_df = pd.DataFrame(roster_data)
    pivot_roster = roster_df.pivot(index="Name", columns="Date", values="Shift").reset_index()

    # ---------------- Display AL Summary ---------------- #
    st.subheader("📈 AL% Summary")
    al_df = pd.DataFrame(al_summary)
    st.dataframe(al_df)

    # ---------------- Display Color-coded Roster ---------------- #
    st.subheader("📅 Weekly Roster")
    st.write("You can manually edit in Excel after download for SL/PL/CO")

    st.dataframe(pivot_roster)

    # ---------------- Download Roster ---------------- #
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Roster")
        return output.getvalue()

    st.download_button("⬇ Download Final Roster", data=to_excel(pivot_roster), file_name="weekly_roster.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

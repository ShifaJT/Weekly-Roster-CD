import io
import math
import datetime as dt
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Config
# ----------------------------
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DEFAULT_HOURLY_PROFILE = np.array([
    0.01, 0.01, 0.01, 0.01, 0.01,
    0.02, 0.03, 0.04, 0.05, 0.06,
    0.07, 0.07, 0.06, 0.05, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.09,
    0.07, 0.05, 0.04, 0.03, 0.02,
])
DEFAULT_HOURLY_PROFILE = DEFAULT_HOURLY_PROFILE / DEFAULT_HOURLY_PROFILE.sum()
START_HOUR = 7
END_HOUR = 22
STRAIGHT_LENGTH_HOURS = 9
SPLIT_OPTIONS = [(7, 5, 16, 4), (8, 5, 17, 4), (9, 5, 18, 4)]
LEAVE_TYPES = ["Special Leave", "Casual Leave", "Planned Leave", "Sick Leave", "Comp Off"]

# ----------------------------
# Helpers
# ----------------------------
def capacity_per_agent_per_hour(aht_seconds: int) -> float:
    return 3600.0 / max(1, aht_seconds)

def required_agents_for_calls(calls: float, aht_seconds: int, target_al_pct: float) -> int:
    if calls <= 0:
        return 0
    cap_per_agent = capacity_per_agent_per_hour(aht_seconds)
    required_capacity = calls / max(0.01, (target_al_pct / 100.0))
    return math.ceil(required_capacity / cap_per_agent)

def expand_daily_to_hourly(daily_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in daily_df.iterrows():
        day = r["Day"]
        total = float(r["Calls"]) if pd.notnull(r["Calls"]) else 0.0
        for hr in range(24):
            calls = total * DEFAULT_HOURLY_PROFILE[hr]
            rows.append({"Day": day, "Hour": hr, "Calls": round(calls, 2)})
    return pd.DataFrame(rows)

def ensure_hourly_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Day"] = df["Day"].astype(str).str.strip().str.capitalize()
    short_map = {"Mon":"Monday","Tue":"Tuesday","Wed":"Wednesday","Thu":"Thursday","Fri":"Friday","Sat":"Saturday","Sun":"Sunday"}
    df["Day"] = df["Day"].replace(short_map)
    df = df[df["Day"].isin(DAYS)]
    df["Hour"] = df["Hour"].astype(int)
    df["Calls"] = pd.to_numeric(df["Calls"], errors="coerce").fillna(0.0)
    df = df[(df["Hour"] >= 0) & (df["Hour"] <= 23)]
    return df.reset_index(drop=True)

def build_hourly_requirements(hourly_calls: pd.DataFrame, aht_sec: int, target_al: float) -> pd.DataFrame:
    df = hourly_calls.copy()
    df["RequiredAgents"] = df["Calls"].apply(lambda c: required_agents_for_calls(c, aht_sec, target_al))
    return df

def empty_roster(champ_names: List[str]) -> Dict[str, Dict[str, Dict]]:
    roster = {d: {} for d in DAYS}
    for d in DAYS:
        for nm in champ_names:
            roster[d][nm] = {"shift": "", "leave": "", "is_split": False}
    return roster

def hourly_coverage_from_roster(roster_day: Dict[str, Dict]) -> np.ndarray:
    coverage = np.zeros(24, dtype=int)
    for champ, info in roster_day.items():
        shift = info.get("shift", "")
        if not shift or info.get("leave"):
            continue
        if shift.startswith("Straight"):
            s, e = [int(x) for x in shift.split()[1].split("-")]
            for h in range(s, e):
                coverage[h] += 1
        elif shift.startswith("Split"):
            parts = [p for p in shift.split()[1:] if p != '&']
            for prt in parts:
                s, e = [int(x) for x in prt.split("-")]
                for h in range(s, e):
                    coverage[h] += 1
    return coverage

def pick_best_straight(start_candidates: List[int], deficit: np.ndarray) -> Tuple[int, int]:
    best_s, best_gain = None, -1
    for s in start_candidates:
        e = s + STRAIGHT_LENGTH_HOURS
        if e > 24:
            continue
        gain = deficit[s:e].clip(min=0).sum()
        if gain > best_gain:
            best_gain = gain
            best_s = s
    return best_s, best_gain

def pick_best_split(deficit: np.ndarray) -> Tuple[Tuple[int,int,int,int], float]:
    best_opt, best_gain = None, -1
    for (s1, l1, s2, l2) in SPLIT_OPTIONS:
        e1, e2 = s1 + l1, s2 + l2
        gain = deficit[s1:e1].clip(min=0).sum() + deficit[s2:e2].clip(min=0).sum()
        if gain > best_gain:
            best_gain = gain
            best_opt = (s1, l1, s2, l2)
    return best_opt, best_gain

def assign_shifts_for_day(day: str, req_df: pd.DataFrame, champions: pd.DataFrame, roster_day: Dict[str, Dict], split_limit: int, prior_split_count: Dict[str, int]) -> None:
    required = np.zeros(24, dtype=int)
    for _, r in req_df[req_df["Day"] == day].iterrows():
        required[int(r["Hour"])] = int(r["RequiredAgents"])
    coverage = hourly_coverage_from_roster(roster_day)
    deficit = required - coverage
    available = [c for c in champions["name"].tolist() if not roster_day[c]["leave"]]
    seed = DAYS.index(day)
    available = available[seed:] + available[:seed]
    straight_starts = list(range(START_HOUR, END_HOUR - STRAIGHT_LENGTH_HOURS + 1))
    for champ in available:
        if (deficit <= 0).all():
            break
        can_split = bool(champions.loc[champions["name"] == champ, "can_split"].iloc[0])
        evening_deficit = deficit[17:22].clip(min=0).sum()
        use_split = can_split and (prior_split_count.get(champ, 0) < split_limit) and (evening_deficit > 0)
        if use_split:
            opt, gain = pick_best_split(deficit)
            if gain <= 0:
                s, gain = pick_best_straight(straight_starts, deficit)
                if s is None or gain <= 0:
                    continue
                shift_label = f"Straight {s}-{s+STRAIGHT_LENGTH_HOURS}"
                for h in range(s, s+STRAIGHT_LENGTH_HOURS):
                    deficit[h] -= 1
                roster_day[champ]["shift"] = shift_label
                roster_day[champ]["is_split"] = False
            else:
                s1, l1, s2, l2 = opt
                e1, e2 = s1 + l1, s2 + l2
                shift_label = f"Split {s1}-{e1} & {s2}-{e2}"
                for h in range(s1, e1):
                    deficit[h] -= 1
                for h in range(s2, e2):
                    deficit[h] -= 1
                roster_day[champ]["shift"] = shift_label
                roster_day[champ]["is_split"] = True
                prior_split_count[champ] = prior_split_count.get(champ, 0) + 1
        else:
            s, gain = pick_best_straight(straight_starts, deficit)
            if s is None or gain <= 0:
                continue
            shift_label = f"Straight {s}-{s+STRAIGHT_LENGTH_HOURS}"
            for h in range(s, s+STRAIGHT_LENGTH_HOURS):
                deficit[h] -= 1
            roster_day[champ]["shift"] = shift_label
            roster_day[champ]["is_split"] = False

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Roster Optimizer", layout="wide")
st.title("📞 Call Center Roster Optimizer")

with st.sidebar:
    target_al = st.number_input("AL Target (%)", 70, 100, 95)
    aht_sec = st.number_input("AHT (seconds)", 60, 1000, 200)
    split_limit_per_agent_week = st.number_input("Max split shifts per week", 0, 10, 3)
    st.subheader("Upload Champions & Call Data")
    champ_file = st.file_uploader("Upload Champions Excel", type=["xlsx"])
    call_file = st.file_uploader("Upload Calls Excel", type=["xlsx"])

# Download templates
def create_champions_template():
    df = pd.DataFrame({
        "name": ["Revathi", "Anjali"],
        "primary_lang": ["ka", "hi"],
        "secondary_langs": ["te,hi", "ka"],
        "calls_per_hour": [12, 11],
        "can_split": [True, True]
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Champions", index=False)
    return buf.getvalue()

def create_calls_template():
    daily = pd.DataFrame({"Day": DAYS, "Calls": [2000, 1800, 2200, 2100, 2300, 2400, 2000]})
    hourly = []
    for d in DAYS:
        for h in range(24):
            hourly.append({"Day": d, "Hour": h, "Calls": 100})
    hourly_df = pd.DataFrame(hourly)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        daily.to_excel(writer, sheet_name="Daily_Data", index=False)
        hourly_df.to_excel(writer, sheet_name="Hourly_Data", index=False)
    return buf.getvalue()

st.sidebar.download_button("Download Champions Template", create_champions_template(), file_name="champions_template.xlsx")
st.sidebar.download_button("Download Calls Template", create_calls_template(), file_name="calls_template.xlsx")

# Load champions
def load_champions_from_excel(file) -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet)
    return df

champ_df = pd.DataFrame()
if champ_file:
    champ_df = load_champions_from_excel(champ_file)
    champ_df = champ_df.rename(columns=lambda x: x.strip().lower())
else:
    st.warning("Upload Champions file.")

# Load calls data
hourly_calls = None
if call_file:
    xls = pd.ExcelFile(call_file)
    if "Hourly_Data" in xls.sheet_names:
        hdf = pd.read_excel(xls, "Hourly_Data")
        hourly_calls = ensure_hourly_schema(hdf)
    elif "Daily_Data" in xls.sheet_names:
        ddf = pd.read_excel(xls, "Daily_Data")
        hourly_calls = ensure_hourly_schema(expand_daily_to_hourly(ddf))
else:
    st.warning("Upload Calls file.")

if champ_df.empty or hourly_calls is None:
    st.stop()

# Requirements
target_al = float(target_al)
req_df = build_hourly_requirements(hourly_calls, aht_sec, target_al)

# Leaves
if "leaves" not in st.session_state:
    st.session_state.leaves = {d: {} for d in DAYS}

colL, colR = st.columns(2)
with colL:
    leave_day = st.selectbox("Day", DAYS)
    leave_champs = st.multiselect("Champions on leave", champ_df["name"].tolist())
with colR:
    leave_type = st.selectbox("Leave Type", LEAVE_TYPES)
    if st.button("Apply Leave"):
        for nm in leave_champs:
            st.session_state.leaves[leave_day][nm] = leave_type

leave_rows = []
for d in DAYS:
    for nm, lt in st.session_state.leaves[d].items():
        leave_rows.append({"Day": d, "Champion": nm, "Leave Type": lt})
leave_df = pd.DataFrame(leave_rows)
st.dataframe(leave_df)

# Download leave status
def create_leave_status_file():
    rows = []
    for d in DAYS:
        for nm in champ_df["name"].tolist():
            status = "Active"
            lt = ""
            if nm in st.session_state.leaves[d]:
                status = "On Leave"
                lt = st.session_state.leaves[d][nm]
            rows.append({"Day": d, "Champion": nm, "Status": status, "Leave Type": lt})
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Leave_Status", index=False)
    return buf.getvalue()

st.download_button("Download Leave Status Template", create_leave_status_file(), file_name="leave_status.xlsx")

# Upload leave status update
uploaded_leave_file = st.file_uploader("Upload Updated Leave Status", type=["xlsx"])
if uploaded_leave_file:
    leave_update_df = pd.read_excel(uploaded_leave_file)
    for _, r in leave_update_df.iterrows():
        day, champ, lt = r["Day"], r["Champion"], r.get("Leave Type", "")
        if day in st.session_state.leaves and champ in champ_df["name"].tolist():
            if pd.notnull(lt) and lt != "":
                st.session_state.leaves[day][champ] = lt
            elif champ in st.session_state.leaves[day]:
                del st.session_state.leaves[day][champ]

# Build roster
champ_names = champ_df["name"].tolist()
roster = empty_roster(champ_names)
for d in DAYS:
    for nm in champ_names:
        if nm in st.session_state.leaves[d]:
            roster[d][nm]["leave"] = st.session_state.leaves[d][nm]

prior_split_count = {nm: 0 for nm in champ_names}
for day in DAYS:
    assign_shifts_for_day(day, req_df, champ_df, roster[day], split_limit_per_agent_week, prior_split_count)

# Summary
summary_rows = []
for day in DAYS:
    required = req_df[req_df["Day"] == day].sort_values("Hour")["RequiredAgents"].values
    coverage = hourly_coverage_from_roster(roster[day])
    cap_per_agent = capacity_per_agent_per_hour(aht_sec)
    calls = hourly_calls[hourly_calls["Day"] == day].sort_values("Hour")["Calls"].values
    cap = coverage * cap_per_agent
    al_hourly = np.where(calls > 0, np.minimum(100.0, (cap / calls) * 100.0), 100.0)
    al_day = float(np.round(np.mean(al_hourly), 2)) if len(al_hourly) else 100.0
    summary_rows.append({"Day": day, "Peak Agents Needed": required.max(), "Assigned Agents": sum(1 for v in roster[day].values() if v["shift"]), "Avg Hourly AL%": al_day})
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df)

# Display roster
roster_tables = []
for day in DAYS:
    rows = []
    for nm in champ_names:
        info = roster[day][nm]
        rows.append({"Champion": nm, "Shift": info["shift"], "Leave": info["leave"], "Split?": info["is_split"]})
    df_day = pd.DataFrame(rows)
    st.subheader(day)
    st.dataframe(df_day)
    df_day["Day"] = day
    roster_tables.append(df_day)
full_roster_df = pd.concat(roster_tables, ignore

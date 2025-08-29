# streamlit_app.py
# Streamlit Call Center Roster Optimizer
# - Daily-call based staffing
# - 95% AL target (editable in UI)
# - Leave management (Special, Casual, Planned, Sick, Comp Off)
# - Balanced straight vs split shift allocation
# - Downloadable roster & leave reports

import io
import math
import json
import datetime as dt
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# ---------- Constants --------
# -----------------------------
DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

# Default hourly distribution (percent of daily calls) if only daily totals are provided
# Roughly bell-shaped with evening peak
DEFAULT_HOURLY_PROFILE = np.array([
    0.01, 0.01, 0.01, 0.01, 0.01,  # 00-04
    0.02, 0.03, 0.04, 0.05, 0.06,  # 05-09
    0.07, 0.07, 0.06, 0.05, 0.05,  # 10-14
    0.06, 0.07, 0.08, 0.09, 0.09,  # 15-19
    0.07, 0.05, 0.04, 0.03, 0.02,  # 20-24
])
DEFAULT_HOURLY_PROFILE = DEFAULT_HOURLY_PROFILE / DEFAULT_HOURLY_PROFILE.sum()

# Working hours window (you can extend to 24h if needed)
START_HOUR = 7
END_HOUR = 22  # exclusive upper bound for shift planning

# Straight and split shift patterns (start times within window)
STRAIGHT_LENGTH_HOURS = 9
SPLIT_BLOCK_1 = (5, )   # hours length block-1
SPLIT_BLOCK_2 = (4, )   # hours length block-2 (total 9)

# Predefined split start options geared to cover evening peaks
SPLIT_OPTIONS = [  # (block1_start, block1_len, block2_start, block2_len)
    (7, 5, 16, 4),
    (8, 5, 17, 4),
    (9, 5, 18, 4),
]

LEAVE_TYPES = ["Special Leave", "Casual Leave", "Planned Leave", "Sick Leave", "Comp Off"]

# -----------------------------
# ---------- Helpers ----------
# -----------------------------

def capacity_per_agent_per_hour(aht_seconds: int) -> float:
    """Calls one agent can handle per hour given AHT in seconds."""
    return 3600.0 / max(1, aht_seconds)


def required_agents_for_calls(calls: float, aht_seconds: int, target_al_pct: float) -> int:
    """Minimum agents needed in an hour so that AL >= target.
    We ensure capacity >= calls / (target_al).
    """
    if calls <= 0:
        return 0
    cap_per_agent = capacity_per_agent_per_hour(aht_seconds)
    required_capacity = calls / max(0.01, (target_al_pct / 100.0))
    agents = math.ceil(required_capacity / cap_per_agent)
    return max(0, agents)


def expand_daily_to_hourly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """If user provides Daily totals, expand to Hourly using DEFAULT_HOURLY_PROFILE."""
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
    # Normalize Day title case to match DAYS list
    df["Day"] = df["Day"].astype(str).str.strip().str.capitalize()
    # Some users may input Mon/Tue... try to map minimally
    short_map = {"Mon":"Monday","Tue":"Tuesday","Wed":"Wednesday","Thu":"Thursday","Fri":"Friday","Sat":"Saturday","Sun":"Sunday"}
    df["Day"] = df["Day"].replace(short_map)
    # Filter to known days only
    df = df[df["Day"].isin(DAYS)]
    df["Hour"] = df["Hour"].astype(int)
    df["Calls"] = pd.to_numeric(df["Calls"], errors="coerce").fillna(0.0)
    # keep only business hours for planning
    df = df[(df["Hour"] >= 0) & (df["Hour"] <= 23)]
    return df.reset_index(drop=True)


def default_champions() -> List[Dict]:
    """Editable starter list. Users can change in the UI."""
    return [
        {"name": "Revathi", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 11, "can_split": True},
        {"name": "Anjali", "primary_lang": "hi", "secondary_langs": ["en"], "calls_per_hour": 10, "can_split": True},
        {"name": "Rakesh", "primary_lang": "te", "secondary_langs": ["hi"], "calls_per_hour": 10, "can_split": False},
        {"name": "Guruswamy", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 12, "can_split": False},
        {"name": "Vishal", "primary_lang": "hi", "secondary_langs": ["en"], "calls_per_hour": 9, "can_split": True},
        {"name": "Kavya", "primary_lang": "ka", "secondary_langs": ["te"], "calls_per_hour": 10, "can_split": True},
        {"name": "Rahul", "primary_lang": "en", "secondary_langs": ["hi"], "calls_per_hour": 9, "can_split": True},
        {"name": "Sneha", "primary_lang": "hi", "secondary_langs": ["ka"], "calls_per_hour": 9, "can_split": False},
    ]


# -----------------------------
# ---- Allocation Engine -------
# -----------------------------

def build_hourly_requirements(hourly_calls: pd.DataFrame, aht_sec: int, target_al: float) -> pd.DataFrame:
    """Return DataFrame with required agents per day-hour."""
    df = hourly_calls.copy()
    df["RequiredAgents"] = df["Calls"].apply(lambda c: required_agents_for_calls(c, aht_sec, target_al))
    return df


def empty_roster(champ_names: List[str]) -> Dict[str, Dict[str, Dict]]:
    """Return {day: {champ: {"shift":"", "leave": "", "is_split": False}}} for the week."""
    roster = {d: {} for d in DAYS}
    for d in DAYS:
        for nm in champ_names:
            roster[d][nm] = {"shift": "", "leave": "", "is_split": False}
    return roster


def hourly_coverage_from_roster(roster_day: Dict[str, Dict], start_hour=START_HOUR, end_hour=END_HOUR) -> np.ndarray:
    """Compute agents present for each hour based on day roster."""
    coverage = np.zeros(24, dtype=int)
    for champ, info in roster_day.items():
        shift = info.get("shift", "")
        if not shift or info.get("leave"):
            continue
        # shift formats: "Straight 9-18" or "Split 7-12 & 16-20"
        if shift.startswith("Straight"):
            try:
                hours = shift.split()[1]  # e.g., '9-18'
                s, e = hours.split("-")
                s, e = int(s), int(e)
                for h in range(s, e):
                    coverage[h] += 1
            except Exception:
                pass
        elif shift.startswith("Split"):
            try:
                body = shift.split()[1:]  # ['7-12', '&', '16-20']
                parts = [p for p in body if p != '&']
                for prt in parts:
                    s, e = prt.split("-")
                    s, e = int(s), int(e)
                    for h in range(s, e):
                        coverage[h] += 1
            except Exception:
                pass
    return coverage


def pick_best_straight(start_candidates: List[int], deficit: np.ndarray) -> Tuple[int, int]:
    """Pick straight shift start that covers maximum deficit across 9h block."""
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
    """Pick split option that covers most deficit."""
    best_opt, best_gain = None, -1
    for (s1, l1, s2, l2) in SPLIT_OPTIONS:
        e1, e2 = s1 + l1, s2 + l2
        if e1 > 24 or e2 > 24:
            continue
        gain = deficit[s1:e1].clip(min=0).sum() + deficit[s2:e2].clip(min=0).sum()
        if gain > best_gain:
            best_gain = gain
            best_opt = (s1, l1, s2, l2)
    return best_opt, best_gain


def assign_shifts_for_day(day: str,
                           req_df: pd.DataFrame,
                           champions: pd.DataFrame,
                           roster_day: Dict[str, Dict],
                           split_limit_per_agent_week: int,
                           prior_split_count: Dict[str, int]) -> None:
    """Greedy assignment to cover per-hour RequiredAgents with a balanced mix of straight & split.
    Updates roster_day in-place.
    """
    # Calculate current coverage & deficit
    required = np.zeros(24, dtype=int)
    for _, r in req_df[req_df["Day"] == day].iterrows():
        required[int(r["Hour"])] = int(r["RequiredAgents"]) if not np.isnan(r["RequiredAgents"]) else 0

    coverage = hourly_coverage_from_roster(roster_day)
    deficit = required - coverage

    # Available champs (not on leave)
    available = [c for c in champions["name"].tolist() if not roster_day[c]["leave"]]

    # Rotate champions for fairness (start position changes each day)
    seed = DAYS.index(day)
    available = available[seed:] + available[:seed]

    straight_starts = list(range(START_HOUR, END_HOUR - STRAIGHT_LENGTH_HOURS + 1))

    for champ in available:
        if (deficit <= 0).all():
            break
        can_split = bool(champions.loc[champions["name"] == champ, "can_split"].iloc[0])
        # Prefer split if evening deficit is large and agent can split and has not exceeded weekly split limit
        evening_deficit = deficit[17:22].clip(min=0).sum()  # 5-9 PM focus
        use_split = can_split and (prior_split_count.get(champ, 0) < split_limit_per_agent_week) and (evening_deficit > 0)

        if use_split:
            (s1, l1, s2, l2), gain = pick_best_split(deficit)
            if gain <= 0:
                # fallback to straight
                s, gain = pick_best_straight(straight_starts, deficit)
                if s is None or gain <= 0:
                    continue
                shift_label = f"Straight {s}-{s+STRAIGHT_LENGTH_HOURS}"
                for h in range(s, s+STRAIGHT_LENGTH_HOURS):
                    deficit[h] -= 1
                roster_day[champ]["shift"] = shift_label
                roster_day[champ]["is_split"] = False
            else:
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


# -----------------------------
# ----------- UI --------------
# -----------------------------

st.set_page_config(page_title="Roster Optimizer", layout="wide")
st.title("📞 Call Center Roster Optimizer")
st.caption("Daily demand → staff needed → balanced roster. Includes leaves & 95% AL target by default.")

with st.sidebar:
    st.header("⚙️ Settings")
    target_al = st.number_input("AL Target (%)", min_value=70, max_value=100, value=95, step=1)
    aht_sec = st.number_input("AHT (seconds)", min_value=60, max_value=1000, value=200, step=10)
    split_limit_per_agent_week = st.number_input("Max split shifts per agent (per week)", min_value=0, max_value=10, value=3)

    st.markdown("---")
    st.subheader("👥 Champions")
    if "champions_df" not in st.session_state:
        st.session_state.champions_df = pd.DataFrame(default_champions())
    st.info("Edit the table below to match your team. Add/remove rows as needed.")
    st.session_state.champions_df = st.data_editor(
        st.session_state.champions_df,
        num_rows="dynamic",
        use_container_width=True,
        key="champions_editor",
    )

    st.markdown("---")
    st.subheader("📄 Upload Calls Data (Excel)")
    st.caption("Use a file with either a sheet named 'Hourly_Data' (Day, Hour, Calls) or 'Daily_Data' (Day, Calls)")
    uploaded = st.file_uploader("Upload .xlsx", type=["xlsx"])

# Prepare calls dataframe
hourly_calls = None
if uploaded:
    try:
        xls = pd.ExcelFile(uploaded)
        if "Hourly_Data" in xls.sheet_names:
            hdf = pd.read_excel(xls, "Hourly_Data")
            needed = {"Day", "Hour", "Calls"}
            if not needed.issubset(set(hdf.columns)):
                st.error("'Hourly_Data' must have columns: Day, Hour, Calls")
            else:
                hourly_calls = ensure_hourly_schema(hdf)
        elif "Daily_Data" in xls.sheet_names:
            ddf = pd.read_excel(xls, "Daily_Data")
            if not {"Day", "Calls"}.issubset(set(ddf.columns)):
                st.error("'Daily_Data' must have columns: Day, Calls")
            else:
                hourly_calls = ensure_hourly_schema(expand_daily_to_hourly(ddf))
        else:
            st.error("Excel must contain 'Hourly_Data' or 'Daily_Data' sheet")
    except Exception as e:
        st.exception(e)

if hourly_calls is None:
    st.warning("No call data yet. Using a small demo week (Daily totals) – please upload your file for real results.")
    demo = pd.DataFrame({
        "Day": DAYS,
        "Calls": [2000, 1600, 1800, 1900, 2100, 2300, 1500]
    })
    hourly_calls = ensure_hourly_schema(expand_daily_to_hourly(demo))

# ---------- Build Requirements ---------
req_df = build_hourly_requirements(hourly_calls, aht_sec, target_al)

# ---------- Leave Management ---------
st.markdown("### 🛌 Leave Management")
if "leaves" not in st.session_state:
    # leaves structure: {day: {champ_name: leave_type}}
    st.session_state.leaves = {d: {} for d in DAYS}

colL, colR = st.columns(2)
with colL:
    leave_day = st.selectbox("Day", DAYS, index=0)
    champ_opts = st.session_state.champions_df["name"].dropna().astype(str).tolist()
    leave_champs = st.multiselect("Select champions on leave", champ_opts)
with colR:
    leave_type = st.selectbox("Leave Type", LEAVE_TYPES)
    if st.button("Apply Leave"):
        for nm in leave_champs:
            st.session_state.leaves[leave_day][nm] = leave_type

# Show leave summary
leave_rows = []
for d in DAYS:
    for nm, lt in st.session_state.leaves[d].items():
        leave_rows.append({"Day": d, "Champion": nm, "Leave Type": lt})
leave_df = pd.DataFrame(leave_rows) if leave_rows else pd.DataFrame(columns=["Day","Champion","Leave Type"])
st.dataframe(leave_df, use_container_width=True, height=160)

# ---------- Roster Allocation ---------
champ_df = st.session_state.champions_df.dropna(subset=["name"]).copy()
champ_df["can_split"] = champ_df["can_split"].fillna(False).astype(bool)
champ_names = champ_df["name"].astype(str).tolist()

roster = empty_roster(champ_names)

# Apply leaves to roster
for d in DAYS:
    for nm in champ_names:
        if nm in st.session_state.leaves[d]:
            roster[d][nm]["leave"] = st.session_state.leaves[d][nm]

# Track weekly split counts for fairness
prior_split_count = {nm: 0 for nm in champ_names}

for day in DAYS:
    assign_shifts_for_day(day, req_df, champ_df, roster[day], int(split_limit_per_agent_week), prior_split_count)

# ---------- KPIs & Validation ----------
summary_rows = []
for day in DAYS:
    required = req_df[req_df["Day"] == day].sort_values("Hour")
    required_series = required["RequiredAgents"].values
    coverage = hourly_coverage_from_roster(roster[day])
    # compute per-hour AL (cap/calls)
    day_calls = hourly_calls[hourly_calls["Day"] == day].sort_values("Hour")["Calls"].values
    cap_per_agent = capacity_per_agent_per_hour(aht_sec)
    cap = coverage * cap_per_agent
    al_hourly = np.where(day_calls > 0, np.minimum(100.0, (cap / day_calls) * 100.0), 100.0)
    al_day = float(np.round(np.mean(al_hourly), 2)) if len(al_hourly) else 100.0

    required_peak = int(required_series.max()) if len(required_series) else 0
    assigned_count = sum(1 for v in roster[day].values() if v["shift"] and not v["leave"]) 

    summary_rows.append({
        "Day": day,
        "Peak Agents Needed": required_peak,
        "Assigned Agents": assigned_count,
        "Avg Hourly AL%": al_day,
    })

summary_df = pd.DataFrame(summary_rows)

st.markdown("### 📊 Day-wise Summary")
st.dataframe(summary_df, use_container_width=True)

# Highlight under/over staffing recommendations
reco_msgs = []
for _, r in summary_df.iterrows():
    if r["Assigned Agents"] < r["Peak Agents Needed"]:
        reco_msgs.append(f"🔺 {r['Day']}: Need +{int(r['Peak Agents Needed'] - r['Assigned Agents'])} more agent(s) to hit {int(target_al)}% AL at peak.")
    elif r["Assigned Agents"] > r["Peak Agents Needed"] + 2:
        reco_msgs.append(f"🔻 {r['Day']}: Consider reducing {int(r['Assigned Agents'] - r['Peak Agents Needed'])} agent(s); you are well above requirement.")

if reco_msgs:
    st.warning("\n".join(reco_msgs))
else:
    st.success("Roster looks balanced vs. demand.")

# ---------- Render Roster Tables ----------
st.markdown("### 🗓️ Roster (Editable per day)")

roster_tables = []
for day in DAYS:
    rows = []
    for nm in champ_names:
        info = roster[day][nm]
        rows.append({
            "Champion": nm,
            "Shift": info["shift"],
            "Leave": info["leave"],
            "Split?": info["is_split"],
        })
    df_day = pd.DataFrame(rows)
    st.subheader(day)
    st.dataframe(df_day, use_container_width=True, height=250)
    df_day["Day"] = day
    roster_tables.append(df_day)

full_roster_df = pd.concat(roster_tables, ignore_index=True)[["Day","Champion","Shift","Leave","Split?"]]

# ---------- Downloads ----------
def to_excel_bytes(df_map: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for sheet, df in df_map.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    return buf.getvalue()

st.markdown("### ⬇️ Download")
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download Roster (CSV)", data=full_roster_df.to_csv(index=False), file_name="roster.csv", mime="text/csv")
with col2:
    excel_bytes = to_excel_bytes({
        "Roster": full_roster_df,
        "Leaves": leave_df,
        "Summary": summary_df,
        "Hourly_Requirements": req_df,
    })
    st.download_button("Download Planner (Excel)", data=excel_bytes, file_name="roster_planner.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- Hourly View (Optional) ----------
st.markdown("### ⏱️ Hourly Requirement vs Coverage (select a day)")
s_day = st.selectbox("Select day for hourly view", DAYS)
req_series = req_df[req_df["Day"] == s_day].sort_values("Hour")
coverage = hourly_coverage_from_roster(roster[s_day])
hr_view = pd.DataFrame({
    "Hour": list(range(24)),
    "RequiredAgents": req_series.set_index("Hour")["RequiredAgents"].reindex(range(24), fill_value=0).values,
    "CoverageAgents": coverage,
})
st.line_chart(hr_view.set_index("Hour"))

st.caption("Tip: Edit champions, apply leaves, and upload your calls file again to replan on the fly.")

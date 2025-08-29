# roster_app.py
"""
Complete Streamlit Roster Planner
- Upload Champions Excel (first sheet must include 'name'; optional columns: primary_lang, secondary_langs, calls_per_hour, can_split)
- Upload Calls Excel with sheet 'Hourly_Data' (Day, Hour, Calls) OR 'Daily_Data' (Day, Calls)
- Active time: 7h50m (28200s). Shift duration: 9 hours.
- Uses 30-minute slots (48 slots/day) for split-shift precision.
- Editable roster grid (Champion rows x Mon..Sun columns). Manual edits applied and AL recalculated.
- Leave management: download blank template, download current leave status, upload leave updates to apply.
- Downloads: sample roster Excel, long CSV, planner Excel.
"""
import io
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --------------------- CONFIG ---------------------
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Use 30-minute slots for precision: 48 slots per day (slot 0 = 00:00-00:29, slot 14 = 07:00-07:29, etc.)
SLOTS_PER_HOUR = 2
SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR

# AHT default and active time
DEFAULT_AHT_SECONDS = 202  # can be changed in UI
ACTIVE_TIME_SECONDS = 7 * 3600 + 50 * 60  # 7h50m = 28200 sec
SHIFT_DURATION_HOURS = 9
SHIFT_DURATION_SLOTS = SHIFT_DURATION_HOURS * SLOTS_PER_HOUR  # 18 slots

# Default hourly distribution (for converting daily to hourly). 24 values (per hour)
DEFAULT_HOURLY_PROFILE = np.array([
    0.01, 0.01, 0.01, 0.01, 0.01,
    0.02, 0.03, 0.04, 0.05, 0.06,
    0.07, 0.07, 0.06, 0.05, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.09,
    0.07, 0.05, 0.04, 0.03, 0.02
])
DEFAULT_HOURLY_PROFILE = DEFAULT_HOURLY_PROFILE / DEFAULT_HOURLY_PROFILE.sum()

# Shift options requested (straight and split). We'll represent times as decimals: 16.5 = 16:30
STRAIGHT_OPTIONS = [
    (7.0, 16.0), (8.0, 17.0), (9.0, 18.0), (10.0, 19.0), (11.0, 20.0), (12.0, 21.0)
]
SPLIT_OPTIONS = [
    ((7.0, 11.0), (16.5, 21.0)),  # 07:00-11:00 & 16:30-21:00
    ((8.0, 12.0), (16.5, 21.0)),
    ((9.0, 13.0), (16.5, 21.0)),
    ((10.0, 14.0), (16.5, 21.0)),
    # also include some longer single block splits if requested earlier
    ((7.0, 12.5), (16.5, 21.0)),  # 07:00-12:30 & 16:30-21:00
    ((10.0, 15.0), (16.5, 21.0)),  # example (10-15 & 16.5-21) but will clip if overlap
]

LEAVE_TYPES = ["Special Leave", "Casual Leave", "Planned Leave", "Sick Leave", "Comp Off"]

# --------------------- UTILITIES ---------------------
def time_to_slot(t: float) -> int:
    """
    Convert decimal hours t (e.g., 7.5 for 07:30) to slot index (0..47).
    """
    # handle t like 16.5 -> 16.5*2 = 33
    return int(round(t * SLOTS_PER_HOUR))

def slot_to_time_str(slot: int) -> str:
    """Return string HH:MM for slot start"""
    hour = slot // SLOTS_PER_HOUR
    minute = (slot % SLOTS_PER_HOUR) * (60 // SLOTS_PER_HOUR)
    return f"{hour:02d}:{minute:02d}"

def slots_span_from_range(start_dec: float, end_dec: float) -> Tuple[int, int]:
    """Given decimal hours start and end, return integer slot indices [s, e)"""
    s = time_to_slot(start_dec)
    e = time_to_slot(end_dec)
    if e <= s:
        # clip to same day end
        e = min(s + SHIFT_DURATION_SLOTS, SLOTS_PER_DAY)
    return s, e

def per_agent_capacity_per_hour(aht_seconds: int) -> float:
    """Number of calls an agent handles in one full productive hour (if fully active)"""
    return 3600.0 / max(1, aht_seconds)

def effective_capacity_per_slot(aht_seconds: int) -> float:
    """Effective capacity per 30-minute slot considering active fraction across the shift."""
    cap_per_hour = per_agent_capacity_per_hour(aht_seconds)
    # active fraction relative to SHIFT_DURATION_HOURS
    active_fraction = ACTIVE_TIME_SECONDS / (SHIFT_DURATION_HOURS * 3600.0)
    # per-hour effective capacity
    eff_per_hour = cap_per_hour * active_fraction
    # per slot (30 min) capacity
    return eff_per_hour / SLOTS_PER_HOUR

def ensure_calls_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept Hourly_Data (Day, Hour, Calls) or Daily_Data (Day, Calls).
    Return DataFrame with columns Day, Hour (0-23), Calls.
    """
    df = df.copy()
    if set(["Day", "Hour", "Calls"]).issubset(df.columns):
        # normalize day names
        df["Day"] = df["Day"].astype(str).str.strip().str.capitalize()
        short = {"Mon":"Monday","Tue":"Tuesday","Wed":"Wednesday","Thu":"Thursday","Fri":"Friday","Sat":"Saturday","Sun":"Sunday"}
        df["Day"] = df["Day"].replace(short)
        df = df[df["Day"].isin(DAYS)].copy()
        df["Hour"] = df["Hour"].astype(int)
        df["Calls"] = pd.to_numeric(df["Calls"], errors="coerce").fillna(0.0)
        # ensure all hours present for each day
        rows=[]
        for d in DAYS:
            tmp = df[df["Day"]==d].set_index("Hour") if not df[df["Day"]==d].empty else pd.DataFrame()
            for h in range(24):
                val = float(tmp.loc[h,"Calls"]) if (not tmp.empty and h in tmp.index) else 0.0
                rows.append({"Day":d,"Hour":h,"Calls":val})
        return pd.DataFrame(rows)
    elif set(["Day","Calls"]).issubset(df.columns):
        # Daily totals — expand using DEFAULT_HOURLY_PROFILE
        rows=[]
        df = df.copy()
        df["Day"] = df["Day"].astype(str).str.strip().str.capitalize()
        df = df[df["Day"].isin(DAYS)].copy()
        for _,r in df.iterrows():
            total = float(r["Calls"]) if pd.notnull(r["Calls"]) else 0.0
            day = r["Day"]
            for h in range(24):
                rows.append({"Day":day,"Hour":h,"Calls": round(total * DEFAULT_HOURLY_PROFILE[h],2)})
        return pd.DataFrame(rows)
    else:
        raise ValueError("Calls sheet must have either (Day,Hour,Calls) OR (Day,Calls).")

def hourly_to_slot_calls(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert hourly calls (Day, Hour 0..23, Calls) into slot calls (Day, Slot 0..47, Calls_per_slot).
    We split each hour's calls equally into two slots (simple approach).
    """
    rows=[]
    for _,r in hourly_df.iterrows():
        day = r["Day"]
        h = int(r["Hour"])
        calls = float(r["Calls"])
        # split equally into two slots
        s1 = h * SLOTS_PER_HOUR
        rows.append({"Day":day,"Slot":s1,"Calls":calls/2.0})
        rows.append({"Day":day,"Slot":s1+1,"Calls":calls/2.0})
    df = pd.DataFrame(rows)
    # ensure all slots present
    full=[]
    for d in DAYS:
        df_d = df[df["Day"]==d].set_index("Slot") if not df[df["Day"]==d].empty else pd.DataFrame()
        for s in range(SLOTS_PER_DAY):
            val = float(df_d.loc[s,"Calls"]) if (not df_d.empty and s in df_d.index) else 0.0
            full.append({"Day":d,"Slot":s,"Calls":val})
    return pd.DataFrame(full)

def required_agents_per_slot(slot_calls: float, aht: int, target_al_pct: float) -> int:
    """
    For a given slot forecast (calls in 30-min), compute agents required so that AL >= target.
    required_capacity = calls / (target_al / 100).
    agents = ceil(required_capacity / cap_per_slot)
    """
    if slot_calls <= 0:
        return 0
    cap_slot = effective_capacity_per_slot(aht)
    required_capacity = slot_calls / max(0.01, (target_al_pct / 100.0))
    agents = math.ceil(required_capacity / max(1e-9, cap_slot))
    return max(0, agents)

def build_slot_requirements(slot_calls_df: pd.DataFrame, aht:int, target_al:float) -> pd.DataFrame:
    df = slot_calls_df.copy()
    df["RequiredAgents"] = df["Calls"].apply(lambda c: required_agents_per_slot(c,aht,target_al))
    return df

# Excel writer factory (try xlsxwriter first, then openpyxl)
def get_excel_writer(buf):
    try:
        import xlsxwriter  # noqa
        return pd.ExcelWriter(buf, engine="xlsxwriter")
    except Exception:
        try:
            import openpyxl  # noqa
            return pd.ExcelWriter(buf, engine="openpyxl")
        except Exception:
            raise RuntimeError("Please install either 'xlsxwriter' or 'openpyxl' to enable Excel exports.")

# --------------------- STREAMLIT APP ---------------------
st.set_page_config(page_title="Roster Planner (7:00-21:00)", layout="wide")
st.title("📞 Call Center Roster Planner — 07:00 to 21:00 | 9h shifts | active 7h50m")
st.caption("Upload champions & calls, manage leaves, auto-generate balanced roster (split & straight), edit manually and download.")

# Sidebar — settings, templates, uploads
with st.sidebar:
    st.header("Settings")
    target_al = st.number_input("AL Target (%)", min_value=70, max_value=100, value=95)
    aht_sec = st.number_input("AHT (seconds)", min_value=30, max_value=1200, value=DEFAULT_AHT_SECONDS)
    split_limit = st.number_input("Max split shifts per agent (week)", min_value=0, max_value=7, value=3)
    st.markdown("---")
    st.subheader("Templates & Uploads")
    st.write("Download templates to fill and upload back.")
    # Champions template
    def champions_template_bytes():
        df = pd.DataFrame({
            "name":["Revathi","Pasang","Kavya S"],
            "primary_lang":["ka","ka","ka"],
            "secondary_langs":["hi,te","te,ta","te"],
            "calls_per_hour":[14,13,15],
            "can_split":[True,False,False]
        })
        buf = io.BytesIO()
        with get_excel_writer(buf) as writer:
            df.to_excel(writer, sheet_name="Champions", index=False)
        return buf.getvalue()
    st.download_button("Download Champions Template", champions_template_bytes(), file_name="champions_template.xlsx")

    # Calls template
    def calls_template_bytes():
        daily = pd.DataFrame({"Day":DAYS, "Calls":[2000,1800,2200,2100,2300,2400,2000]})
        hourly_rows=[]
        for d in DAYS:
            for h in range(24):
                hourly_rows.append({"Day":d,"Hour":h,"Calls":100 if 7<=h<21 else 10})
        hourly = pd.DataFrame(hourly_rows)
        buf = io.BytesIO()
        with get_excel_writer(buf) as writer:
            daily.to_excel(writer, sheet_name="Daily_Data", index=False)
            hourly.to_excel(writer, sheet_name="Hourly_Data", index=False)
        return buf.getvalue()
    st.download_button("Download Calls Template", calls_template_bytes(), file_name="calls_template.xlsx")

    # Blank leave template
    def leave_template_bytes(names=None):
        if names is None:
            names = ["Revathi","Pasang","Kavya S"]
        rows=[]
        for d in DAYS:
            for n in names:
                rows.append({"Day":d,"Champion":n,"Leave Type":""})
        df = pd.DataFrame(rows)
        buf = io.BytesIO()
        with get_excel_writer(buf) as writer:
            df.to_excel(writer, sheet_name="Leave_Status", index=False)
        return buf.getvalue()
    st.download_button("Download Blank Leave Template", leave_template_bytes(), file_name="leave_template.xlsx")

    st.markdown("---")
    st.caption("Upload champions file first, then calls file. After upload, use the main UI to manage leaves, edit roster and download outputs.")
    champions_file = st.file_uploader("Upload Champions Excel (first sheet used)", type=["xlsx"])
    calls_file = st.file_uploader("Upload Calls Excel (Hourly_Data or Daily_Data)", type=["xlsx"])

# Validate uploads
if champions_file is None:
    st.warning("Please upload Champions file (use Champions Template if needed).")
    st.stop()
if calls_file is None:
    st.warning("Please upload Calls file (Hourly_Data or Daily_Data).")
    st.stop()

# Load champions
try:
    x = pd.ExcelFile(champions_file)
    champ_sheet = x.sheet_names[0]
    champs_df = pd.read_excel(x, champ_sheet)
except Exception as e:
    st.error(f"Failed reading champions file: {e}")
    st.stop()

# Normalize champions
champs_df.columns = [c.strip() for c in champs_df.columns]
if "name" not in [c.lower() for c in champs_df.columns]:
    # try lowercase names
    if "name" not in champs_df.columns and "Name" not in champs_df.columns:
        st.error("Champions sheet must contain a 'name' column (case-insensitive).")
        st.stop()
# Make column names lowercase keys for safe access
champs_df.columns = [c.lower() for c in champs_df.columns]
# Ensure required columns exist
for col in ["primary_lang","secondary_langs","calls_per_hour","can_split"]:
    if col not in champs_df.columns:
        champs_df[col] = None
# Normalize types
champs_df["name"] = champs_df["name"].astype(str).str.strip()
# boolean convert
try:
    champs_df["can_split"] = champs_df["can_split"].astype(bool)
except Exception:
    champs_df["can_split"] = champs_df["can_split"].apply(lambda v: str(v).strip().lower() in ("true","1","yes"))


# Load calls file and normalize to hourly
try:
    x = pd.ExcelFile(calls_file)
    if "Hourly_Data" in x.sheet_names:
        calls_raw = pd.read_excel(x, "Hourly_Data")
        hourly_calls_df = ensure_calls_hourly(calls_raw)
    elif "Daily_Data" in x.sheet_names:
        daily_raw = pd.read_excel(x, "Daily_Data")
        hourly_df = ensure_calls_hourly(daily_raw)
        hourly_calls_df = hourly_df
    else:
        st.error("Calls file must contain either 'Hourly_Data' or 'Daily_Data' sheet.")
        st.stop()
except Exception as e:
    st.error(f"Failed reading calls file: {e}")
    st.stop()

# Convert hourly to slot-level calls
slot_calls_df = hourly_to_slot_calls(hourly_calls_df)

# Build slot requirements per target AL and AHT
slot_req_df = build_slot_requirements(slot_calls_df, int(aht_sec), float(target_al))

# Session-state leaves
if "leaves" not in st.session_state:
    st.session_state["leaves"] = {d: {} for d in DAYS}

# Ability to download current leave status and upload updates
def current_leave_status_bytes():
    rows=[]
    names = champs_df["name"].tolist()
    for d in DAYS:
        for n in names:
            status = "Active"
            lt = ""
            if n in st.session_state["leaves"].get(d, {}):
                status = "On Leave"; lt = st.session_state["leaves"][d][n]
            rows.append({"Day":d,"Champion":n,"Status":status,"Leave Type":lt})
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with get_excel_writer(buf) as writer:
        df.to_excel(writer, sheet_name="Leave_Status", index=False)
    return buf.getvalue()

st.download_button("Download Current Leave Status", current_leave_status_bytes(), file_name="current_leave_status.xlsx")

uploaded_leave_updates = st.file_uploader("Upload Leave Updates (sheet with Day,Champion,Leave Type)", type=["xlsx"])
if uploaded_leave_updates is not None:
    try:
        xl = pd.ExcelFile(uploaded_leave_updates)
        sheet = xl.sheet_names[0]
        ldf = pd.read_excel(xl, sheet)
        # normalize columns
        ldf.columns = [c.strip() for c in ldf.columns]
        for _,r in ldf.iterrows():
            day = str(r.get("Day","")).strip()
            champ = str(r.get("Champion","")).strip()
            lt = r.get("Leave Type","")
            if day in DAYS and champ in champs_df["name"].tolist():
                if pd.notna(lt) and str(lt).strip() != "":
                    st.session_state["leaves"].setdefault(day, {})
                    st.session_state["leaves"][day][champ] = str(lt).strip()
                else:
                    # remove leave if exists
                    if champ in st.session_state["leaves"].get(day, {}):
                        del st.session_state["leaves"][day][champ]
        st.success("Applied leave updates from file.")
    except Exception as e:
        st.error(f"Failed to apply leave updates: {e}")

# Build initial empty roster and apply leaves
champ_names = champs_df["name"].tolist()
def empty_roster(champ_names: List[str]):
    r = {d: {} for d in DAYS}
    for d in DAYS:
        for n in champ_names:
            r[d][n] = {"shift":"","leave":"","is_split":False}
    return r

roster = empty_roster(champ_names)
# apply leaves
for d in DAYS:
    for n in champ_names:
        if n in st.session_state["leaves"].get(d, {}):
            roster[d][n]["leave"] = st.session_state["leaves"][d][n]

# Assignment: slot-level greedy per day
# compute required per slot for day
prior_split_count = {n: 0 for n in champ_names}
cap_slot = effective_capacity_per_slot(int(aht_sec))

for d in DAYS:
    # build required array for day per slot
    required_slots = slot_req_df[slot_req_df["Day"]==d].sort_values("Slot")["RequiredAgents"].values
    if len(required_slots) == 0:
        required_slots = np.zeros(SLOTS_PER_DAY, dtype=int)
    else:
        required_slots = np.asarray(required_slots, dtype=int)
    # current coverage from roster (initially zero except manual leaves)
    coverage = np.zeros(SLOTS_PER_DAY, dtype=int)
    # mark available champions for day (not on leave)
    available = [n for n in champ_names if not roster[d][n]["leave"]]
    # rotate available for fairness by day index
    rot = DAYS.index(d)
    available = available[rot:] + available[:rot]
    # for each champ in available assign best shift covering deficit
    deficit = required_slots - coverage
    straight_starts = [s for s,e in STRAIGHT_OPTIONS]
    # convert straight start decimals to slot indices for candidates
    straight_slot_candidates = [time_to_slot(s) for s in straight_starts]
    for champ in available:
        if (deficit <= 0).all():
            break
        can_split = bool(champs_df.loc[champs_df["name"]==champ, "can_split"].iloc[0])
        evening_def = deficit[ (17*SLOTS_PER_HOUR):(21*SLOTS_PER_HOUR) ].clip(min=0).sum()
        use_split = can_split and (prior_split_count.get(champ,0) < int(split_limit)) and (evening_def > 0)
        if use_split:
            # evaluate SPLIT_OPTIONS to find best gain
            best_opt=None; best_gain=-1
            for (s1_e1, s2_e2) in SPLIT_OPTIONS:
                s1, e1 = s1_e1[0], s1_e1[1] if isinstance(s1_e1, tuple) else (s1_e1[0], s1_e1[1])
                # but SPLIT_OPTIONS defined as ((s1,e1),(s2,e2)) — ensure unpack
            # SPLIT_OPTIONS is tuple of pairs; fix iteration:
            for opt in SPLIT_OPTIONS:
                if isinstance(opt[0], tuple):
                    (s1,e1),(s2,e2) = opt
                else:
                    s1,e1 = opt[0], opt[1]
                    s2,e2 = opt[2], opt[3]
                s1_slot, e1_slot = time_to_slot(s1), time_to_slot(e1)
                s2_slot, e2_slot = time_to_slot(s2), time_to_slot(e2)
                s1_slot = max(0, min(SLOTS_PER_DAY, s1_slot))
                e1_slot = max(0, min(SLOTS_PER_DAY, e1_slot))
                s2_slot = max(0, min(SLOTS_PER_DAY, s2_slot))
                e2_slot = max(0, min(SLOTS_PER_DAY, e2_slot))
                gain = deficit[s1_slot:e1_slot].clip(min=0).sum() + deficit[s2_slot:e2_slot].clip(min=0).sum()
                if gain > best_gain:
                    best_gain = gain; best_opt = (s1_slot,e1_slot,s2_slot,e2_slot, s1,e1,s2,e2)
            if best_gain <= 0:
                # fallback to straight
                best_s=None; best_gain_s=-1
                for s_dec in [s for s,e in STRAIGHT_OPTIONS]:
                    s_slot = time_to_slot(s_dec)
                    e_slot = s_slot + SHIFT_DURATION_SLOTS
                    if e_slot > SLOTS_PER_DAY:
                        continue
                    gain = deficit[s_slot:e_slot].clip(min=0).sum()
                    if gain > best_gain_s:
                        best_gain_s=gain; best_s=s_slot
                if best_s is None or best_gain_s <= 0:
                    continue
                # assign straight
                s_slot = best_s; e_slot = s_slot + SHIFT_DURATION_SLOTS
                roster[d][champ]["shift"] = f"Straight {slot_to_time_str(s_slot)}-{slot_to_time_str(e_slot)}"
                roster[d][champ]["is_split"] = False
                for h in range(s_slot, e_slot):
                    deficit[h] -= 1
            else:
                # assign best split
                s1_slot,e1_slot,s2_slot,e2_slot, s1_dec,e1_dec,s2_dec,e2_dec = best_opt
                roster[d][champ]["shift"] = f"Split {slot_to_time_str(s1_slot)}-{slot_to_time_str(e1_slot)} & {slot_to_time_str(s2_slot)}-{slot_to_time_str(e2_slot)}"
                roster[d][champ]["is_split"] = True
                prior_split_count[champ] = prior_split_count.get(champ,0)+1
                for h in range(s1_slot,e1_slot):
                    deficit[h] -= 1
                for h in range(s2_slot,e2_slot):
                    deficit[h] -= 1
        else:
            # pick best straight start
            best_s=None; best_gain=-1
            for s_dec in [s for s,e in STRAIGHT_OPTIONS]:
                s_slot = time_to_slot(s_dec)
                e_slot = s_slot + SHIFT_DURATION_SLOTS
                if e_slot > SLOTS_PER_DAY:
                    continue
                gain = deficit[s_slot:e_slot].clip(min=0).sum()
                if gain > best_gain:
                    best_gain=gain; best_s=s_slot
            if best_s is None or best_gain <= 0:
                continue
            s_slot = best_s; e_slot = s_slot + SHIFT_DURATION_SLOTS
            roster[d][champ]["shift"] = f"Straight {slot_to_time_str(s_slot)}-{slot_to_time_str(e_slot)}"
            roster[d][champ]["is_split"] = False
            for h in range(s_slot,e_slot):
                deficit[h] -= 1

# --------------------- KPI & AL Prediction ---------------------
cap_slot = effective_capacity_per_slot(int(aht_sec))
# compute weekly totals and per-day predicted AL
weekly_calls_per_day = slot_calls_df.groupby("Day", as_index=False)["Calls"].sum().rename(columns={"Calls":"WeeklyCalls"})
total_weekly_calls = weekly_calls_per_day["WeeklyCalls"].sum()
avg_daily_calls = weekly_calls_per_day["WeeklyCalls"].mean()
peak_day = weekly_calls_per_day.loc[weekly_calls_per_day["WeeklyCalls"].idxmax(),"Day"] if not weekly_calls_per_day.empty else ""

summary_rows=[]
weekly_capacity_sum = 0.0
for d in DAYS:
    # assigned agents count (agents with shift and not on leave)
    assigned_agents = sum(1 for v in roster[d].values() if v["shift"] and not v["leave"])
    # coverage per slot
    cov_slots = np.zeros(SLOTS_PER_DAY, dtype=int)
    for n in champ_names:
        info = roster[d][n]
        if info["leave"] or not info["shift"]:
            continue
        shift_str = info["shift"]
        if shift_str.startswith("Straight"):
            # parse times
            try:
                times = shift_str.split()[1]
                s_str, e_str = times.split("-")
                s_slot = int(s_str.split(":")[0]) * SLOTS_PER_HOUR + (int(s_str.split(":")[1])//30)
                e_slot = int(e_str.split(":")[0]) * SLOTS_PER_HOUR + (int(e_str.split(":")[1])//30)
            except Exception:
                s_slot = 0; e_slot = 0
            for s in range(s_slot, e_slot):
                if 0 <= s < SLOTS_PER_DAY:
                    cov_slots[s] += 1
        elif shift_str.startswith("Split"):
            try:
                parts = shift_str.replace("Split","").strip().split("&")
                for p in parts:
                    p = p.strip()
                    s_str,e_str = p.split("-")
                    s_slot = int(s_str.split(":")[0]) * SLOTS_PER_HOUR + (int(s_str.split(":")[1])//30)
                    e_slot = int(e_str.split(":")[0]) * SLOTS_PER_HOUR + (int(e_str.split(":")[1])//30)
                    for s in range(s_slot, e_slot):
                        if 0 <= s < SLOTS_PER_DAY:
                            cov_slots[s] += 1
            except Exception:
                pass
    # compute slot-level AL
    day_slot_calls = slot_calls_df[slot_calls_df["Day"]==d].sort_values("Slot")["Calls"].values
    if len(day_slot_calls)==0:
        day_slot_calls = np.zeros(SLOTS_PER_DAY)
    cap = cov_slots * cap_slot  # effective capacity in calls per slot
    with np.errstate(divide='ignore', invalid='ignore'):
        al_slots = np.where(day_slot_calls>0, np.minimum(100.0, (cap / day_slot_calls) * 100.0), 100.0)
    day_al = float(np.round(np.mean(al_slots),2)) if len(al_slots)>0 else 100.0
    summary_rows.append({"Day":d, "WeeklyCalls": float(day_slot_calls.sum()), "AssignedAgents": assigned_agents, "Predicted_AL%": day_al})
    weekly_capacity_sum += cap.sum()

weekly_al_pct = float(np.round(min(100.0, (weekly_capacity_sum / max(1.0, total_weekly_calls)) * 100.0),2)) if total_weekly_calls>0 else 100.0

summary_df = pd.DataFrame(summary_rows)

# --------------------- UI OUTPUT ---------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Weekly Calls (total)", f"{int(total_weekly_calls):,}")
col2.metric("Average Daily Calls", f"{int(avg_daily_calls):,}")
col3.metric("Peak Day", f"{peak_day}")
col4.metric("Predicted Weekly AL%", f"{weekly_al_pct}%")

st.markdown("---")
st.header("Day-wise Summary")
st.dataframe(summary_df, use_container_width=True)

# Recommendations
reco=[]
for _,r in summary_df.iterrows():
    d = r["Day"]
    peak_needed = int(slot_req_df[slot_req_df["Day"]==d]["RequiredAgents"].max()) if not slot_req_df[slot_req_df["Day"]==d].empty else 0
    assigned = int(r["AssignedAgents"])
    if assigned < peak_needed:
        reco.append(f"🔺 {d}: Need +{peak_needed-assigned} agent(s) at peak to reach {int(target_al)}% AL.")
    elif assigned > peak_needed + 2:
        reco.append(f"🔻 {d}: Consider reducing {assigned-peak_needed} agent(s); currently above requirement.")
if reco:
    st.warning("\n".join(reco))
else:
    st.success("Roster is balanced vs forecast.")

# --------------------- Editable Roster (Sample Format) ---------------------
st.markdown("---")
st.header("Roster (editable) — sample format")

# Build sample pivot: champions as rows, columns Mon..Sun with shift or WO
sample_rows=[]
for n in champ_names:
    row = {"Champion": n}
    for d in DAYS:
        info = roster[d][n]
        cell = "WO" if info["leave"] else (info["shift"] if info["shift"] else "")
        row[d] = cell
    sample_rows.append(row)
sample_df = pd.DataFrame(sample_rows)

st.caption("Edit any cell (enter shift string like 'Straight 07:00-16:00' or 'Split 07:00-11:00 & 16:30-21:00', or enter 'WO' to mark leave). Click 'Apply Manual Edits' to save.")
edited = st.data_editor(sample_df, num_rows="dynamic", use_container_width=True, key="roster_editor")

if st.button("Apply Manual Edits"):
    # apply edits back to roster structure
    for _,r in edited.iterrows():
        nm = r["Champion"]
        for d in DAYS:
            val = str(r[d]).strip()
            if val.upper() == "WO":
                roster[d][nm]["leave"] = "ManualLeave"
                roster[d][nm]["shift"] = ""
                roster[d][nm]["is_split"] = False
            elif val == "" or val.lower() in ("nan","none"):
                roster[d][nm]["leave"] = ""
                roster[d][nm]["shift"] = ""
                roster[d][nm]["is_split"] = False
            else:
                # treat as shift string
                roster[d][nm]["leave"] = ""
                roster[d][nm]["shift"] = val
                roster[d][nm]["is_split"] = "Split" in val
    st.success("Manual edits applied. Recomputing AL and summary...")
    # recompute summary (same method as above)
    # (For brevity, just refresh the page by re-running; but we recompute here)
    # In Streamlit, simply rerun by setting a small session flag
    st.experimental_rerun()

# --------------------- Downloads ---------------------
st.markdown("---")
st.header("Downloads")

# full long roster
full_long = []
for d in DAYS:
    for n in champ_names:
        info = roster[d][n]
        full_long.append({"Day":d,"Champion":n,"Shift":info["shift"],"Leave":info["leave"],"Split?":info["is_split"]})
full_long_df = pd.DataFrame(full_long)

# sample pivot (Champion rows, Mon..Sun)
sample_pivot = full_long_df.pivot(index="Champion", columns="Day", values="Shift").reset_index()
# set 'WO' where leave present
for _,row in full_long_df.iterrows():
    if row["Leave"]:
        sample_pivot.loc[sample_pivot["Champion"]==row["Champion"], row["Day"]] = "WO"

def to_excel_bytes(sheets: Dict[str,pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with get_excel_writer(buf) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()

st.download_button("Download Roster (sample format) - Excel", to_excel_bytes({"Roster_Sample": sample_pivot}), file_name="roster_sample.xlsx")
st.download_button("Download Full Roster (long) - CSV", full_long_df.to_csv(index=False).encode("utf-8"), file_name="roster_long.csv")
st.download_button("Download Planner (Excel) - Roster/Leaves/Summary/SlotRequirements",
                  to_excel_bytes({
                      "Roster": full_long_df,
                      "Leaves": pd.DataFrame([{"Day":d,"Champion":k,"Leave Type":v} for d in DAYS for k,v in st.session_state["leaves"].get(d,{}).items()]),
                      "Summary": summary_df,
                      "Slot_Requirements": slot_req_df
                  }),
                  file_name="roster_planner.xlsx")

# Hourly & slot view
st.markdown("---")
st.subheader("Hourly & Slot view")
sel_day = st.selectbox("Select day for hourly/slot view", DAYS)
# hourly aggregated required and assigned
hourly_req = slot_req_df[slot_req_df["Day"]==sel_day].groupby(slot_req_df["Slot"]//SLOTS_PER_HOUR, as_index=False)["RequiredAgents"].max()
hourly_req = hourly_req.rename(columns={"Slot":"Hour","RequiredAgents":"RequiredAgentsAtPeakSlot"})
# assigned agents per hour aggregated from slots
assigned_slots = np.zeros(24)
cov_slots = np.zeros(SLOTS_PER_DAY, dtype=int)
for n in champ_names:
    info = roster[sel_day][n]
    if info["leave"] or not info["shift"]:
        continue
    s = info["shift"]
    if s.startswith("Straight"):
        times = s.split()[1]
        s_str,e_str = times.split("-")
        s_slot = int(s_str.split(":")[0]) * SLOTS_PER_HOUR + (int(s_str.split(":")[1])//30)
        e_slot = int(e_str.split(":")[0]) * SLOTS_PER_HOUR + (int(e_str.split(":")[1])//30)
        for sl in range(s_slot,e_slot):
            if 0<=sl<SLOTS_PER_DAY: cov_slots[sl]+=1
    elif s.startswith("Split"):
        parts = s.replace("Split","").strip().split("&")
        for p in parts:
            p = p.strip()
            if "-" not in p: continue
            s_str,e_str = p.split("-")
            s_slot = int(s_str.split(":")[0]) * SLOTS_PER_HOUR + (int(s_str.split(":")[1])//30)
            e_slot = int(e_str.split(":")[0]) * SLOTS_PER_HOUR + (int(e_str.split(":")[1])//30)
            for sl in range(s_slot,e_slot):
                if 0<=sl<SLOTS_PER_DAY: cov_slots[sl]+=1
# aggregate per hour
for h in range(24):
    assigned_slots[h] = cov_slots[h*2:(h*2+2)].max()  # show peak slot coverage within hour
hr_view = pd.DataFrame({"Hour": list(range(24)), "AssignedAgents": assigned_slots.astype(int)})
# join with hourly_req (which is in hours)
hourly_calls_view = hourly_calls_df[hourly_calls_df["Day"]==sel_day].sort_values("Hour").reset_index(drop=True)
hourly_calls_view = hourly_calls_view.rename(columns={"Calls":"CallsInHour"})
hourly_display = hourly_calls_view.merge(hourly_req, left_on="Hour", right_on="Hour", how="left").fillna(0)
hourly_display = hourly_display.merge(hr_view, on="Hour", how="left")
st.dataframe(hourly_display, use_container_width=True)

st.caption("End of planner. Edit roster and re-apply edits as needed. Use templates to upload data in correct format.")

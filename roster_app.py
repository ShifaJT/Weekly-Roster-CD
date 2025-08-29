# roster_app.py
"""
Date-hourly Roster Planner (single file)
- Upload Champions Excel (first sheet; must have 'name' column)
- Upload Calls Excel with sheet 'Hourly_Data' having columns: Date, Hour (0-23), Calls
- Leave management via roster editor (enter 'WO' or leave type into cell)
- Straight + Split shifts supported (all 9h)
- Active time = 7h50m used for capacity calculations
- Download roster, CSV, planner Excel
"""
import io
import math
import calendar
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- CONFIG ----------------
SHIFT_DURATION_HOURS = 9
ACTIVE_TIME_SECONDS = 7 * 3600 + 50 * 60  # 7h50m = 28200 sec
SLOTS_PER_HOUR = 1  # we will use hourly slots for simplicity (0..23)
SLOTS_PER_DAY = 24

# Straight shift start/end (start_hour, end_hour)
STRAIGHT_SHIFTS = [
    (7, 16), (8, 17), (9, 18), (10, 19), (11, 20), (12, 21)
]

# Split shift options (block1_start, block1_end, block2_start, block2_end)
# All blocks sum to approximate shift working windows; we will allow half-hour by using .5 (interpreted as :30)
SPLIT_OPTIONS = [
    (7.0, 11.0, 16.5, 21.0),  # 07:00-11:00 & 16:30-21:00
    (8.0, 12.0, 16.5, 21.0),
    (9.0, 13.0, 16.5, 21.0),
    (10.0, 14.0, 16.5, 21.0),
    (7.0, 12.5, 16.5, 21.0),  # 07:00-12:30 & 16:30-21:00
]

LEAVE_MARKERS = {"WO", "WO+", "WO-"}  # user can input any leave text; we'll treat non-empty/non-shift as leave

# ---------------- HELPERS ----------------
def parse_time_decimal(t: float) -> Tuple[int, int]:
    """Convert decimal hour (e.g., 16.5) to (hour, minute)"""
    hour = int(math.floor(t))
    minute = 30 if abs(t - hour - 0.5) < 0.01 else 0
    return hour, minute

def slot_index_from_time_decimal(t: float) -> int:
    """Map decimal t to slot index (hour). Half hours -> map to floor hour (simple)."""
    h, m = parse_time_decimal(t)
    return h  # use hour indexing 0..23; half-hours will map to same hour for coverage approximation

def format_time_from_decimal(t: float) -> str:
    h, m = parse_time_decimal(t)
    return f"{h:02d}:{m:02d}"

def per_agent_capacity_per_hour(aht_seconds: int) -> float:
    """Calls handled per full productive hour if agent is fully active that hour."""
    return 3600.0 / max(1, aht_seconds)

def effective_capacity_per_hour(aht_seconds: int) -> float:
    """Effective per-hour capacity considering active fraction across shift (active / shift_length)."""
    cap_full = per_agent_capacity_per_hour(aht_seconds)
    active_fraction = ACTIVE_TIME_SECONDS / (SHIFT_DURATION_HOURS * 3600.0)
    return cap_full * active_fraction

def required_agents_for_calls(calls: float, aht_seconds: int, target_al_pct: float) -> int:
    """Given hourly calls, compute minimal agents needed to hit target AL for that hour."""
    if calls <= 0:
        return 0
    cap_agent = effective_capacity_per_hour(aht_seconds)
    required_capacity = calls / max(0.01, (target_al_pct / 100.0))
    agents = math.ceil(required_capacity / max(1e-9, cap_agent))
    return max(0, agents)

def get_excel_writer(buf):
    try:
        import xlsxwriter  # noqa
        return pd.ExcelWriter(buf, engine="xlsxwriter")
    except Exception:
        try:
            import openpyxl  # noqa
            return pd.ExcelWriter(buf, engine="openpyxl")
        except Exception:
            raise RuntimeError("Install xlsxwriter or openpyxl to enable Excel exports.")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Date-Hourly Roster Planner", layout="wide")
st.title("📅 Date-hourly Roster Planner — 07:00–21:00 shifts | active 7h50m")
st.caption("Upload champions and an hourly calls file (Date,Hour,Calls). Manage leaves directly in the roster table.")

with st.sidebar:
    st.header("Settings & Uploads")
    target_al = st.number_input("AL Target (%)", min_value=70, max_value=100, value=95)
    aht_sec = st.number_input("AHT (seconds)", min_value=30, max_value=1200, value=202)
    max_split_per_week = st.number_input("Max split shifts per agent / week", min_value=0, max_value=7, value=3)
    st.markdown("---")
    st.subheader("Upload Files")
    champions_file = st.file_uploader("Champions Excel (first sheet used)", type=["xlsx"])
    calls_file = st.file_uploader("Calls Excel (Hourly_Data sheet)", type=["xlsx"])
    st.markdown("---")
    st.write("Notes:")
    st.write("- Champion file must have a 'name' column (case-insensitive). Optional: primary_lang, secondary_langs, calls_per_hour, can_split.")
    st.write("- Calls file: sheet 'Hourly_Data' with columns: Date (YYYY-MM-DD), Hour (0-23), Calls (numeric).")

    st.markdown("---")
    st.header("Download Templates")
    def champions_template_bytes():
        df = pd.DataFrame({
            "name": ["Revathi", "Anjali", "Rakesh"],
            "primary_lang": ["ka", "hi", "te"],
            "secondary_langs": ["te,hi", "ka", "ka"],
            "calls_per_hour": [12, 11, 10],
            "can_split": [True, True, False]
        })
        buf = io.BytesIO()
        with get_excel_writer(buf) as writer:
            df.to_excel(writer, sheet_name="Champions", index=False)
        return buf.getvalue()
    st.download_button("Champions template", champions_template_bytes(), file_name="champions_template.xlsx")

    def calls_template_bytes():
        # Example for two dates
        dates = [pd.Timestamp.today().normalize().date(), pd.Timestamp.today().normalize().date() + pd.Timedelta(days=1)]
        rows = []
        for d in dates:
            for h in range(7, 21):  # only business hours sample
                rows.append({"Date": str(d), "Hour": h, "Calls": 100})
        df = pd.DataFrame(rows)
        buf = io.BytesIO()
        with get_excel_writer(buf) as writer:
            df.to_excel(writer, sheet_name="Hourly_Data", index=False)
        return buf.getvalue()
    st.download_button("Calls (hourly) template", calls_template_bytes(), file_name="calls_hourly_template.xlsx")

# Validate uploads
if champions_file is None:
    st.warning("Upload champions file to begin.")
    st.stop()
if calls_file is None:
    st.warning("Upload calls file (Hourly_Data) to begin.")
    st.stop()

# Load champions
try:
    x = pd.ExcelFile(champions_file)
    champ_sheet = x.sheet_names[0]
    champs_df = pd.read_excel(x, champ_sheet)
except Exception as e:
    st.error(f"Error reading champions file: {e}")
    st.stop()

# normalize columns
champs_df.columns = [c.strip() for c in champs_df.columns]
cols_lower = [c.lower() for c in champs_df.columns]
if "name" not in cols_lower:
    st.error("Champions sheet must contain 'name' column.")
    st.stop()

# rename columns to lowercase
champs_df.columns = [c.lower() for c in champs_df.columns]
for optional_col in ["primary_lang", "secondary_langs", "calls_per_hour", "can_split"]:
    if optional_col not in champs_df.columns:
        champs_df[optional_col] = None

champs_df["name"] = champs_df["name"].astype(str).str.strip()
# boolean convert for can_split
def to_bool(v):
    try:
        return bool(v) if isinstance(v, (bool, int)) else str(v).strip().lower() in ("true","1","yes")
    except Exception:
        return False
champs_df["can_split"] = champs_df["can_split"].apply(to_bool)

champ_names = champs_df["name"].tolist()

# Load calls (date+hour)
try:
    x = pd.ExcelFile(calls_file)
    if "Hourly_Data" not in x.sheet_names:
        st.error("Calls file must have sheet 'Hourly_Data' with columns Date, Hour, Calls.")
        st.stop()
    calls_raw = pd.read_excel(x, "Hourly_Data")
except Exception as e:
    st.error(f"Error reading calls file: {e}")
    st.stop()

# Validate calls columns
calls_raw.columns = [c.strip() for c in calls_raw.columns]
required_cols = {"Date","Hour","Calls"}
if not required_cols.issubset(set(calls_raw.columns)):
    st.error("Hourly_Data sheet must have columns: Date, Hour, Calls.")
    st.stop()

# Normalize Date column to date objects and Hour to int
calls_raw["Date"] = pd.to_datetime(calls_raw["Date"]).dt.date
calls_raw["Hour"] = calls_raw["Hour"].astype(int)
calls_raw["Calls"] = pd.to_numeric(calls_raw["Calls"], errors="coerce").fillna(0.0)

# Build list of unique dates (in sorted order)
unique_dates = sorted(calls_raw["Date"].unique())
if len(unique_dates) == 0:
    st.error("No dates found in calls file.")
    st.stop()

# Build hourly calls pivot for each date
# We'll compute required agents per hour per date
def build_requirements(calls_df: pd.DataFrame, aht:int, target_al:float) -> pd.DataFrame:
    rows = []
    for d in sorted(calls_df["Date"].unique()):
        df_day = calls_df[calls_df["Date"] == d]
        # ensure hours 0..23 present
        hours = {h:0.0 for h in range(24)}
        for _,r in df_day.iterrows():
            hours[int(r["Hour"])] = float(r["Calls"])
        for h in range(24):
            calls = hours[h]
            req = required_agents_for_calls(calls, aht, target_al)
            rows.append({"Date": d, "Hour": h, "Calls": calls, "RequiredAgents": req})
    return pd.DataFrame(rows)

req_df = build_requirements(calls_raw, int(aht_sec), float(target_al))

# ---------------- Build initial roster ----------------
# roster structure = { date: { champ_name: { "shift": "", "leave": "", "is_split": False } } }
def empty_roster_for_dates(champ_names: List[str], dates: List) -> Dict:
    roster = {}
    for d in dates:
        roster[d] = {name: {"shift":"","leave":"","is_split":False} for name in champ_names}
    return roster

roster = empty_roster_for_dates(champ_names, unique_dates)

# Apply greedy assignment per date (hourly basis) trying to cover RequiredAgents
# simple greedy: iterate champs and pick best straight or split if helps evening
cap_per_agent_hour = effective_capacity_per_hour = None  # define below

def per_agent_effective_capacity_per_hour(aht_seconds:int) -> float:
    cap_full = per_agent_capacity_per_hour(aht_seconds)
    active_fraction = ACTIVE_TIME_SECONDS / (SHIFT_DURATION_HOURS * 3600.0)
    return cap_full * active_fraction

cap_per_agent_hour = per_agent_effective_capacity_per_hour(int(aht_sec))

# assign function (hourly resolution)
def assign_for_date(date, req_df_day: pd.DataFrame, champs_df: pd.DataFrame, roster_day: Dict[str, Dict], split_limit: int, prior_split_count: Dict[str,int]):
    # required array per hour
    required = np.zeros(24, dtype=int)
    for _,r in req_df_day.iterrows():
        required[int(r["Hour"])] = int(r["RequiredAgents"])
    coverage = np.zeros(24, dtype=int)
    # initial coverage (none assigned except may have manual leave)
    # available champs not on leave
    available = [c for c in champs_df["name"].tolist() if roster_day[c]["leave"] == ""]
    # rotate available by date index for fairness
    idx = unique_dates.index(date)
    available = available[idx:] + available[:idx]
    for champ in available:
        if (required - coverage <= 0).all():
            break
        # decide use split
        can_split = bool(champs_df.loc[champs_df["name"]==champ,"can_split"].iloc[0])
        evening_deficit = (required - coverage)[17:21].clip(min=0).sum()
        use_split = can_split and (prior_split_count.get(champ,0) < split_limit) and (evening_deficit > 0)
        if use_split:
            # evaluate split options and straight options
            best_split = None; best_split_gain = -1
            for s1,e1,s2,e2 in SPLIT_OPTIONS:
                s1_slot = int(math.floor(s1))
                e1_slot = int(math.ceil(e1))
                s2_slot = int(math.floor(s2))
                e2_slot = int(math.ceil(e2))
                # clip to 0..23
                s1_slot = max(0,min(23,s1_slot))
                e1_slot = max(0,min(24,e1_slot))
                s2_slot = max(0,min(23,s2_slot))
                e2_slot = max(0,min(24,e2_slot))
                gain = (required[s1_slot:e1_slot] - coverage[s1_slot:e1_slot]).clip(min=0).sum() + (required[s2_slot:e2_slot] - coverage[s2_slot:e2_slot]).clip(min=0).sum()
                if gain > best_split_gain:
                    best_split_gain = gain; best_split = (s1_slot,e1_slot,s2_slot,e2_slot,s1,e1,s2,e2)
            if best_split is None or best_split_gain <= 0:
                # fallback to straight
                best_straight = None; best_gain = -1
                for s,e in STRAIGHT_SHIFTS:
                    s_slot = int(s); e_slot = int(e)
                    if e_slot - s_slot != SHIFT_DURATION_HOURS:
                        pass
                    gain = (required[s_slot:e_slot] - coverage[s_slot:e_slot]).clip(min=0).sum()
                    if gain > best_gain:
                        best_gain = gain; best_straight = (s_slot,e_slot)
                if best_straight is None or best_gain <= 0:
                    continue
                s_slot,e_slot = best_straight
                roster_day[champ]["shift"] = f"Straight {s_slot:02d}:00-{e_slot:02d}:00"
                roster_day[champ]["is_split"] = False
                for h in range(s_slot,e_slot):
                    coverage[h] += 1
            else:
                s1_slot,e1_slot,s2_slot,e2_slot,s1_dec,e1_dec,s2_dec,e2_dec = best_split
                roster_day[champ]["shift"] = f"Split {format_time_from_decimal(s1_dec)}-{format_time_from_decimal(e1_dec)} & {format_time_from_decimal(s2_dec)}-{format_time_from_decimal(e2_dec)}"
                roster_day[champ]["is_split"] = True
                prior_split_count[champ] = prior_split_count.get(champ,0) + 1
                for h in range(s1_slot,e1_slot):
                    coverage[h] += 1
                for h in range(s2_slot,e2_slot):
                    coverage[h] += 1
        else:
            # pick best straight
            best_straight = None; best_gain = -1
            for s,e in STRAIGHT_SHIFTS:
                s_slot = int(s); e_slot = int(e)
                gain = (required[s_slot:e_slot] - coverage[s_slot:e_slot]).clip(min=0).sum()
                if gain > best_gain:
                    best_gain = gain; best_straight = (s_slot,e_slot)
            if best_straight is None or best_gain <= 0:
                continue
            s_slot,e_slot = best_straight
            roster_day[champ]["shift"] = f"Straight {s_slot:02d}:00-{e_slot:02d}:00"
            roster_day[champ]["is_split"] = False
            for h in range(s_slot,e_slot):
                coverage[h] += 1

# Perform assignment for each date
prior_split_count = {name: 0 for name in champ_names}
for d in unique_dates:
    df_day = req_df[req_df["Date"]==d]
    assign_for_date(d, df_day, champs_df, roster[d], int(max_split_per_week), prior_split_count)

# ---------------- Compute AL predictions (per date and overall) ----------------
cap_per_agent_hour = per_agent_effective_capacity_per_hour = None
def per_agent_effective_capacity_per_hour(aht):
    full = per_agent_capacity_per_hour(aht)  # from earlier helper
    frac = ACTIVE_TIME_SECONDS / (SHIFT_DURATION_HOURS * 3600.0)
    return full * frac

def per_agent_capacity_per_hour(aht_seconds):
    return 3600.0 / max(1, aht_seconds)

cap_per_agent_hour = per_agent_effective_capacity_per_hour(int(aht_sec))

summary_rows = []
total_calls_all_dates = 0.0
total_capacity_all_dates = 0.0

for d in unique_dates:
    # compute coverage per hour
    cov = np.zeros(24, dtype=int)
    for champ in champ_names:
        info = roster[d][champ]
        if info["leave"] or not info["shift"]:
            continue
        shift = info["shift"]
        if shift.startswith("Straight"):
            try:
                times = shift.split()[1]
                start, end = times.split("-")
                s_h = int(start.split(":")[0]); e_h = int(end.split(":")[0])
                for h in range(s_h, e_h):
                    cov[h] += 1
            except Exception:
                pass
        elif shift.startswith("Split"):
            try:
                _, blocks = shift.split(" ",1)
                parts = blocks.split("&")
                for p in parts:
                    p = p.strip()
                    stime, etime = p.split("-")
                    s_h = int(stime.split(":")[0]); e_h = int(etime.split(":")[0])
                    for h in range(s_h, e_h):
                        if 0 <= h < 24:
                            cov[h] += 1
            except Exception:
                pass
    # calls per hour for date
    calls_day = calls_raw[calls_raw["Date"]==d].set_index("Hour")["Calls"].reindex(range(24), fill_value=0).values
    capacity = cov * cap_per_agent_hour
    # per-hour AL in percent
    with np.errstate(divide='ignore', invalid='ignore'):
        al_hour = np.where(calls_day > 0, np.minimum(100.0, (capacity / calls_day) * 100.0), 100.0)
    daily_al = float(np.round(np.mean(al_hour),2)) if len(al_hour)>0 else 100.0
    total_calls_all_dates += calls_day.sum()
    total_capacity_all_dates += capacity.sum()
    assigned_agents = sum(1 for v in roster[d].values() if v["shift"] and not v["leave"])
    summary_rows.append({"Date": d, "TotalCalls": float(calls_day.sum()), "AssignedAgents": int(assigned_agents), "Predicted_AL%": daily_al})

weekly_al = float(np.round(min(100.0, (total_capacity_all_dates / max(1.0, total_calls_all_dates)) * 100.0), 2)) if total_calls_all_dates > 0 else 100.0
summary_df = pd.DataFrame(summary_rows)

# ---------------- UI: display metrics and summary ----------------
st.markdown("## Summary & AL prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Dates covered", f"{len(unique_dates)}")
col2.metric("Total calls (all dates)", f"{int(total_calls_all_dates):,}")
col3.metric("Predicted AL (overall)", f"{weekly_al}%")

st.dataframe(summary_df, use_container_width=True)

# Recommendations
reco = []
for _, r in summary_df.iterrows():
    d = r["Date"]
    peak_needed = int(req_df[req_df["Date"]==d]["RequiredAgents"].max()) if not req_df[req_df["Date"]==d].empty else 0
    assigned = int(r["AssignedAgents"])
    if assigned < peak_needed:
        reco.append(f"🔺 {d}: Add +{peak_needed-assigned} agent(s) at peak to reach {int(target_al)}% AL.")
    elif assigned > peak_needed + 2:
        reco.append(f"🔻 {d}: Consider reducing {assigned-peak_needed} agent(s); you are above requirement.")
if reco:
    st.warning("\n".join(reco))
else:
    st.success("Roster appears balanced vs hourly forecasts.")

# ---------------- Roster Editor (Date-based sample pivot) ----------------
st.markdown("---")
st.header("Roster (editable) — Date × Champions sample format")

# Build pivot where rows = champion, columns = date strings (ISO)
date_cols = [str(d) for d in unique_dates]
pivot_rows = []
for name in champ_names:
    row = {"Champion": name}
    for d in unique_dates:
        info = roster[d][name]
        if info["leave"]:
            row[str(d)] = info["leave"] if info["leave"] else "WO"
        else:
            row[str(d)] = info["shift"]
    pivot_rows.append(row)
pivot_df = pd.DataFrame(pivot_rows)

st.caption("Edit cells to adjust shifts or mark leave. Enter 'WO' or any Leave string to mark leave. Format for shifts: e.g. 'Straight 07:00-16:00' or 'Split 07:00-11:00 & 16:30-21:00'. After edits click 'Apply Manual Edits'.")

edited = st.data_editor(pivot_df, num_rows="dynamic", use_container_width=True, key="roster_editor")

if st.button("Apply Manual Edits"):
    # apply edits into roster structure
    for _, r in edited.iterrows():
        name = r["Champion"]
        for d in unique_dates:
            cell = str(r[str(d)]).strip()
            if cell == "" or cell.lower() in ("nan","none"):
                roster[d][name]["shift"] = ""
                roster[d][name]["leave"] = ""
                roster[d][name]["is_split"] = False
            elif cell.upper() == "WO" or (not cell.startswith("Straight") and not cell.startswith("Split")):
                # mark as leave; accept arbitrary leave text
                roster[d][name]["leave"] = cell if cell.upper() != "WO" else "WO"
                roster[d][name]["shift"] = ""
                roster[d][name]["is_split"] = False
            else:
                roster[d][name]["leave"] = ""
                roster[d][name]["shift"] = cell
                roster[d][name]["is_split"] = ("Split" in cell)
    st.success("Manual edits applied. Recomputing AL and summary...")
    st.experimental_rerun()

# ---------------- Downloads ----------------
st.markdown("---")
st.header("Downloads")

# Build long roster (Date, Champion, Shift, Leave)
long_rows = []
for d in unique_dates:
    for name in champ_names:
        info = roster[d][name]
        long_rows.append({"Date": d, "Champion": name, "Shift": info["shift"], "Leave": info["leave"], "Split?": info["is_split"]})
long_df = pd.DataFrame(long_rows)

# sample pivot: champion rows and date columns (Shift text)
sample_pivot = long_df.pivot(index="Champion", columns="Date", values="Shift").reset_index()
# put 'WO' where leave present
for _, r in long_df.iterrows():
    if r["Leave"]:
        sample_pivot.loc[sample_pivot["Champion"] == r["Champion"], r["Date"]] = "WO"

def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with get_excel_writer(buf) as writer:
        for name, df in sheets.items():
            # Convert date columns to strings for Excel readability
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()

st.download_button("Download Roster (sample pivot) - Excel", to_excel_bytes({"Roster_Sample": sample_pivot}), file_name="roster_sample.xlsx")
st.download_button("Download Roster (long) - CSV", long_df.to_csv(index=False).encode("utf-8"), file_name="roster_long.csv")
st.download_button("Download Planner (Excel) - Roster+Summary+HourlyReq", to_excel_bytes({"Roster": long_df, "Summary": summary_df, "HourlyRequirements": req_df}), file_name="roster_planner.xlsx")

# Hourly view for a chosen date
st.markdown("---")
st.subheader("Hourly view for a selected Date")
sel_date = st.selectbox("Select Date", unique_dates)
hr_req = req_df[req_df["Date"] == sel_date].sort_values("Hour").reset_index(drop=True)
# build assigned agents per hour from roster
assigned_per_hour = np.zeros(24, dtype=int)
for name in champ_names:
    info = roster[sel_date][name]
    if info["shift"] and not info["leave"]:
        shift = info["shift"]
        if shift.startswith("Straight"):
            try:
                times = shift.split()[1]
                s,e = times.split("-")
                s_h = int(s.split(":")[0]); e_h = int(e.split(":")[0])
                for h in range(s_h, e_h):
                    if 0 <= h < 24: assigned_per_hour[h] += 1
            except Exception:
                pass
        elif shift.startswith("Split"):
            try:
                parts = shift.replace("Split","").strip().split("&")
                for p in parts:
                    stime, etime = p.strip().split("-")
                    s_h = int(stime.split(":")[0]); e_h = int(etime.split(":")[0])
                    for h in range(s_h,e_h):
                        if 0 <= h < 24: assigned_per_hour[h] += 1
            except Exception:
                pass
hourly_display = pd.DataFrame({
    "Hour": list(range(24)),
    "Calls": hr_req["Calls"].values,
    "RequiredAgents": hr_req["RequiredAgents"].values,
    "AssignedAgents": assigned_per_hour
})
st.dataframe(hourly_display, use_container_width=True)

st.caption("Leave management: edit cell for date & champion in the roster above and enter 'WO' or leave text to mark leave. No separate leave template needed.")

# End

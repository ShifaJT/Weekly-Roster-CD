# roster_app.py
"""
Weekly roster planner:
- Upload champions (first sheet must have 'name', optional 'can_split' boolean)
- Upload past 30 days hourly calls (sheet Hourly_Data with Date, Hour, Calls)
- Predict next 7 days (weekday-hour average)
- Auto-assign champions to meet target AL% (Answered Calls / Total Calls)
- Editable roster, inline leave marking, color-coded roster view
- Download roster/excel outputs
"""

import io
import math
from datetime import datetime, timedelta
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Roster Planner — Predict & Plan", layout="wide")
OPERATING_HOURS = list(range(7, 21))  # 7..20 inclusive hours
ACTIVE_MINUTES = 7 * 60 + 50  # 7h50m = 470 minutes effective per shift
STRAIGHT_SHIFTS = [
    (7, 16), (8, 17), (9, 18), (10, 19), (11, 20), (12, 21)
]
SPLIT_SHIFTS = [
    ((7.0, 11.0), (16.5, 21.0)),
    ((8.0, 12.0), (16.5, 21.0)),
    ((9.0, 13.0), (16.5, 21.0)),
    ((10.0, 14.0), (16.5, 21.0)),
    ((7.0, 12.5), (16.5, 21.0)),
]
LEAVE_OPTIONS = ["", "WO", "CO", "Special Leave", "Casual Leave", "Planned Leave", "Sick Leave", "Comp Off"]

# Color map for display (simple)
COLOR_MAP = {
    "WO": "#0B3D91",            # dark blue
    "CO": "#FFD700",            # gold
    "Special Leave": "#FF7F7F", # light red
    "Casual Leave": "#FFB266",  # orange
    "Planned Leave": "#9FE2BF", # mint
    "Sick Leave": "#FF9999",    # red tint
    "Comp Off": "#FFD580",      # tan
    "Straight": "#F5B041",      # orange
    "Split": "#85C1E9",         # light blue
    "Default": "#DDDDDD"        # grey
}

# ---------------- Helpers ----------------
def get_excel_writer(buf):
    try:
        import xlsxwriter  # noqa
        return pd.ExcelWriter(buf, engine="xlsxwriter")
    except Exception:
        try:
            import openpyxl  # noqa
            return pd.ExcelWriter(buf, engine="openpyxl")
        except Exception:
            raise RuntimeError("Install xlsxwriter or openpyxl for Excel exports.")

def create_champions_template_bytes():
    df = pd.DataFrame({
        "name": ["Revathi", "Pasang", "Kavya S"],
        "primary_lang": ["ka", "ka", "ka"],
        "secondary_langs": ["te,hi", "te", "hi"],
        "calls_per_hour": [12, 11, 11],
        "can_split": [True, False, True]
    })
    buf = io.BytesIO()
    with get_excel_writer(buf) as writer:
        df.to_excel(writer, sheet_name="Champions", index=False)
    buf.seek(0)
    return buf.getvalue()

def create_hourly_calls_template_bytes(days=30):
    today = datetime.today().date()
    start = today - timedelta(days=days-1)
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        for h in OPERATING_HOURS:
            rows.append({"Date": d.strftime("%Y-%m-%d"), "Hour": h, "Calls": ""})
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with get_excel_writer(buf) as writer:
        df.to_excel(writer, sheet_name="Hourly_Data", index=False)
    buf.seek(0)
    return buf.getvalue()

def read_champions(file) -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, xls.sheet_names[0])
    # normalize column names to lower
    df.columns = [c.strip() for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    if "name" not in df.columns:
        raise ValueError("Champions sheet must contain 'name' column.")
    for col in ["primary_lang", "secondary_langs", "calls_per_hour", "can_split"]:
        if col not in df.columns:
            df[col] = None
    df["name"] = df["name"].astype(str).str.strip()
    # ensure can_split boolean
    def to_bool(v):
        if pd.isna(v): return False
        if isinstance(v, bool): return v
        s = str(v).strip().lower()
        return s in ("true","1","yes","y")
    df["can_split"] = df["can_split"].apply(to_bool)
    # ensure calls_per_hour numeric
    df["calls_per_hour"] = pd.to_numeric(df["calls_per_hour"], errors="coerce").fillna(df["calls_per_hour"].mean() if not df["calls_per_hour"].isna().all() else 12)
    return df

def read_hourly_calls(file) -> pd.DataFrame:
    xls = pd.ExcelFile(file)
    if "Hourly_Data" not in xls.sheet_names:
        raise ValueError("Calls workbook must contain sheet 'Hourly_Data' with Date, Hour, Calls.")
    df = pd.read_excel(xls, "Hourly_Data")
    df = df.copy()
    if "date" not in [c.lower() for c in df.columns]:
        raise ValueError("Hourly_Data must have 'Date' column.")
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # unify names
    cols_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "date": cols_map[c] = "Date"
        elif lc == "hour": cols_map[c] = "Hour"
        elif lc == "calls": cols_map[c] = "Calls"
    df = df.rename(columns=cols_map)
    if not set(["Date","Hour","Calls"]).issubset(set(df.columns)):
        raise ValueError("Hourly_Data must include columns Date, Hour, Calls (case-insensitive).")
    # normalize values
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Hour"] = df["Hour"].astype(int)
    df["Calls"] = pd.to_numeric(df["Calls"], errors="coerce").fillna(0.0)
    # keep only operating hours
    df = df[df["Hour"].isin(OPERATING_HOURS)].copy()
    return df

def weekday_hour_pattern(hourly_df: pd.DataFrame) -> pd.DataFrame:
    # group by weekday and hour -> mean calls
    tmp = hourly_df.copy()
    tmp["Weekday"] = pd.to_datetime(tmp["Date"]).day_name()
    pattern = tmp.groupby(["Weekday","Hour"], as_index=False)["Calls"].mean()
    return pattern

def predict_next_week(pattern: pd.DataFrame, start_date: datetime.date=None) -> pd.DataFrame:
    if start_date is None:
        start_date = datetime.today().date() + timedelta(days=1)  # predict starting tomorrow
    next7 = [start_date + timedelta(days=i) for i in range(7)]
    rows = []
    for d in next7:
        wd = d.strftime("%A")
        for h in OPERATING_HOURS:
            r = pattern[(pattern["Weekday"]==wd) & (pattern["Hour"]==h)]
            calls = float(r["Calls"].iloc[0]) if not r.empty else 0.0
            rows.append({"Date": d, "Hour": h, "Calls": round(calls)})
    return pd.DataFrame(rows)

def minutes_from_decimal_hour(t: float) -> int:
    # t like 16.5 -> 16:30 -> return minutes from midnight start-of-hour
    hour = int(math.floor(t))
    minute = 30 if abs(t - hour - 0.5) < 0.01 else 0
    return hour*60 + minute

def format_shift_label(shift_tuple) -> str:
    # shift_tuple can be ("Straight", start,end) or ("Split", (s1,e1),(s2,e2))
    if shift_tuple[0] == "Straight":
        s,e = shift_tuple[1], shift_tuple[2]
        return f"{s:02d}:00 to {e:02d}:00"
    else:
        (s1,e1),(s2,e2) = shift_tuple[1], shift_tuple[2]
        def fm(t):
            hour = int(math.floor(t)); minute = int(round((t-hour)*60))
            return f"{hour:02d}:{minute:02d}"
        return f"{fm(s1)} to {fm(e1)} & {fm(s2)} to {fm(e2)}"

# ---------------- Allocation logic ----------------
def allocate_for_week(champs_df, predicted_calls_df, target_al_pct, aht_minutes, max_split_champs):
    """
    For each date, determine number of champs needed and assign shifts.
    Approach:
     - For date: compute total_calls
     - Start with 0 champs and add champs (prefer straight) until AnsweredCalls/TotalCalls >= target_al_pct
     - When adding champion, choose champ who has been used least this week (fairness) and who can be split if we need split
     - For each assigned champ choose shift to maximize coverage of peak hours (simple heuristic: pick shift covering peak hour)
    Returns roster_long: list of dict {Date,Champion,ShiftLabel,IsSplit,Leave}
            summary_df: per-date metrics
    """
    roster_long = []
    summary_rows = []
    champions = champs_df.copy().reset_index(drop=True)
    champ_names = champions["name"].tolist()
    # track usage counts for fairness
    usage_count = {nm: 0 for nm in champ_names}
    # track how many split assignments per champ (respect max_split_champs)
    split_count = {nm: 0 for nm in champ_names}
    # for each date
    for d in sorted(predicted_calls_df["Date"].unique()):
        df_day = predicted_calls_df[predicted_calls_df["Date"]==d]
        total_calls_day = float(df_day["Calls"].sum())
        # compute peak hour
        peak_row = df_day.loc[df_day["Calls"].idxmax()] if not df_day.empty else None
        peak_hour = int(peak_row["Hour"]) if peak_row is not None else 9
        # answered_calls if we have N champs: answered = N * ACTIVE_MINUTES / AHT_minutes
        def answered_calls_for_n(n):
            return (n * ACTIVE_MINUTES) / aht_minutes
        # we will keep list of assigned champs today and chosen shift label
        assigned = []
        # choose order of champs by least used
        # we will iterate adding champs until target met or all champs used
        champs_queue = deque(sorted(champ_names, key=lambda x: usage_count[x]))
        # while not meeting target and there are champs left
        while True:
            # compute current answered with current assigned count
            n_assigned = len(assigned)
            answered = answered_calls_for_n(n_assigned)
            achieved = (answered / total_calls_day) * 100 if total_calls_day>0 else 100.0
            if achieved >= target_al_pct or n_assigned >= len(champ_names):
                break
            # pick next champ from queue who is not already assigned today
            candidate = None
            for _ in range(len(champs_queue)):
                cand = champs_queue.popleft()
                if cand not in [a["Champion"] for a in assigned]:
                    candidate = cand
                    # push back to end for fairness rotation
                    champs_queue.append(cand)
                    break
                champs_queue.append(cand)
            if candidate is None:
                break
            # choose whether to give candidate split or straight
            can_split = bool(champions.loc[champions["name"]==candidate,"can_split"].iloc[0])
            use_split = False
            # heuristic: if evening peak (>=17) and candidate can_split and split_count less than max, prefer split
            if can_split and split_count[candidate] < max_split_champs:
                # check if evening peak or heavy evening volume
                evening_calls = df_day[df_day["Hour"].between(17,20)]["Calls"].sum()
                day_calls = total_calls_day if total_calls_day>0 else 1
                if (evening_calls / day_calls) > 0.20:  # if >20% calls in evening, prefer split
                    use_split = True
            # choose shift to cover peak hour: pick straight option that covers peak or split that covers peak
            if use_split:
                # find best split option covering peak hour in second block ideally
                chosen_split = None
                for (s1,e1),(s2,e2) in SPLIT_SHIFTS:
                    # if peak in either block, it's useful; prefer second block when evening peak
                    if peak_hour >= int(math.floor(s2)) and peak_hour < int(math.ceil(e2)):
                        chosen_split = ((s1,e1),(s2,e2)); break
                if chosen_split is None:
                    chosen_split = SPLIT_SHIFTS[0]
                shift_label = format_shift_label(("Split", chosen_split[0], chosen_split[1]))
                is_split = True
                split_count[candidate] += 1
            else:
                # choose straight shift covering peak hour
                chosen_straight = None
                for s,e in STRAIGHT_SHIFTS:
                    if peak_hour >= s and peak_hour < e:
                        chosen_straight = (s,e); break
                if chosen_straight is None:
                    chosen_straight = STRAIGHT_SHIFTS[0]
                shift_label = format_shift_label(("Straight", chosen_straight[0], chosen_straight[1]))
                is_split = False
            assigned.append({"Date": d, "Champion": candidate, "Shift": shift_label, "IsSplit": is_split, "Leave": ""})
            usage_count[candidate] += 1
            # continue loop until target achieved
        # After loop compute final metrics
        total_answered = answered_calls_for_n(len(assigned))
        achieved_al = (total_answered / total_calls_day) * 100 if total_calls_day>0 else 100.0
        summary_rows.append({
            "Date": d, "TotalCalls": int(total_calls_day),
            "AssignedChamps": len(assigned),
            "AnsweredCalls": round(total_answered),
            "Achieved_AL%": round(achieved_al,2),
            "Target_AL%": target_al_pct
        })
        roster_long.extend(assigned)
    summary_df = pd.DataFrame(summary_rows)
    roster_long_df = pd.DataFrame(roster_long)
    # If no roster rows (e.g., zero calls) fill with empty entries per champ per date optionally
    return roster_long_df, summary_df

# ---------------- UI ----------------
st.title("📊 Weekly Roster Planner — Predict & Plan to hit AL%")

with st.sidebar:
    st.header("Settings")
    target_al = st.slider("Target AL% (Answered / Total)", min_value=80, max_value=100, value=95, step=1)
    aht_seconds = st.number_input("AHT (seconds)", value=300, min_value=30, max_value=2000, step=10)
    aht_minutes = aht_seconds / 60.0
    max_split_champs = st.number_input("Max split-shift assignments per champ per week", min_value=0, max_value=7, value=3)
    days_for_template = st.number_input("Prefill template days", min_value=14, max_value=60, value=30)
    st.markdown("---")
    st.subheader("Files / Templates")
    champ_file = st.file_uploader("Upload Champions Excel", type=["xlsx"])
    calls_file = st.file_uploader("Upload Past Hourly Calls (Hourly_Data sheet)", type=["xlsx"])
    st.download_button("Download Champions Template", create_champions_template_bytes(), file_name="champions_template.xlsx")
    st.download_button("Download Hourly Calls Template (prefilled last X days)", create_hourly_calls_template_bytes(days=int(days_for_template)), file_name="hourly_calls_template.xlsx")

# require uploads
if champ_file is None or calls_file is None:
    st.info("Upload Champions and Past Hourly Calls (Hourly_Data) to proceed. Use templates if needed.")
    st.stop()

# Read inputs
try:
    champs_df = read_champions(champ_file)
    calls_df = read_hourly_calls(calls_file)
except Exception as e:
    st.error(f"Error reading uploaded files: {e}")
    st.stop()

st.success("Files read successfully.")

st.write(f"Champions: {len(champs_df)} | Historical calls rows: {len(calls_df)}")

# require at least some data
if calls_df.empty:
    st.error("Hourly calls file has no rows for operating hours.")
    st.stop()

# Build pattern and predict next week
pattern = weekday_hour_pattern(calls_df)
predicted = predict_next_week(pattern)

st.subheader("Predicted calls for next 7 days (from weekday-hour averages)")
st.dataframe(predicted, use_container_width=True)

# Allocate champions to meet target AL%
roster_long_df, summary_df = allocate_for_week(champs_df, predicted, float(target_al), aht_minutes, int(max_split_champs))

st.subheader("Planned assignments summary (auto-assigned to meet target AL%)")
st.dataframe(summary_df, use_container_width=True)

# If any day failed to meet target (Achieved_AL% < target), show recommendation
fail_days = summary_df[summary_df["Achieved_AL%"] < target_al]
if not fail_days.empty:
    st.warning("Some days did not reach the target AL% even after assigning all champs. See suggestions below.")
    for _, r in fail_days.iterrows():
        st.write(f"- {r['Date']}: Achieved AL% {r['Achieved_AL%']} with {r['AssignedChamps']} champs (Target {r['Target_AL%']}). Consider adding more champs or increasing AHT speed.")

# Build roster pivot (Champion rows, Date columns) for editable view
dates = sorted(predicted["Date"].unique())
champ_names = champs_df["name"].tolist()
# Start pivot with blank or assigned shift
pivot_rows = []
for name in champ_names:
    row = {"Champion": name}
    for d in dates:
        # find assigned shift if exists
        assigned_row = roster_long_df[(roster_long_df["Date"]==d) & (roster_long_df["Champion"]==name)]
        if not assigned_row.empty:
            shift_text = assigned_row.iloc[0]["Shift"]
            row[str(d)] = shift_text
        else:
            row[str(d)] = ""  # user can fill: 'WO' or shift string
    pivot_rows.append(row)
pivot_df = pd.DataFrame(pivot_rows)

st.markdown("### Editable roster (Champion × Date). Edit any cell to set shift text or leave (enter 'WO' or leave type). Click 'Apply Edits' to commit.")
edited = st.data_editor(pivot_df, num_rows="dynamic", use_container_width=True, key="roster_editor")

if st.button("Apply Edits"):
    # Apply manual edits -> override roster_long_df accordingly
    # Build new long df from edited table
    new_long = []
    for _, r in edited.iterrows():
        name = r["Champion"]
        for d in dates:
            cell = str(r[str(d)]).strip()
            if cell == "":
                # empty means no assignment (leave blank)
                new_long.append({"Date": d, "Champion": name, "Shift": "", "IsSplit": "Split" in cell, "Leave": ""})
            elif cell.upper() in ("WO","CO","PLANNED LEAVE","SICK LEAVE","COMP OFF","CASUAL LEAVE","SPECIAL LEAVE"):
                new_long.append({"Date": d, "Champion": name, "Shift": "", "IsSplit": False, "Leave": cell})
            else:
                # treat as shift
                new_long.append({"Date": d, "Champion": name, "Shift": cell, "IsSplit": "Split" in cell, "Leave": ""})
    roster_long_df = pd.DataFrame(new_long)
    # recompute summary (AnsweredCalls & Achieved AL)
    summary_rows = []
    for d in dates:
        df_day = predicted[predicted["Date"]==d]
        total_calls_day = float(df_day["Calls"].sum())
        assigned_count = len(roster_long_df[(roster_long_df["Date"]==d) & (roster_long_df["Shift"]!="")])
        answered = (assigned_count * ACTIVE_MINUTES) / aht_minutes
        achieved = (answered / total_calls_day) * 100 if total_calls_day>0 else 100.0
        summary_rows.append({"Date": d, "TotalCalls": int(total_calls_day), "AssignedChamps": assigned_count, "AnsweredCalls": round(answered), "Achieved_AL%": round(achieved,2), "Target_AL%": target_al})
    summary_df = pd.DataFrame(summary_rows)
    st.success("Manual edits applied. Recomputed summary.")
    st.dataframe(summary_df, use_container_width=True)

# Render a color-coded HTML roster view similar to your image
def html_roster_table(roster_long_df, champions_order, dates):
    # build pivot again but include leave detection for coloring
    # create header with dates and weekday labels
    header_dates = [pd.to_datetime(d).strftime("%a\n%d-%b") for d in dates]
    # create rows
    rows_html = []
    for name in champions_order:
        cells = []
        # name and base-shift (first assigned shift in week) column
        first_shifts = roster_long_df[(roster_long_df["Champion"]==name) & (roster_long_df["Shift"]!="")]
        base_shift = first_shifts.iloc[0]["Shift"] if not first_shifts.empty else ""
        cells.append(f"<td style='padding:6px; font-weight:600; background:#FFFFFF'>{name}</td>")
        cells.append(f"<td style='padding:6px; background:#FFFFFF'>{base_shift}</td>")
        for d in dates:
            row = roster_long_df[(roster_long_df["Date"]==d) & (roster_long_df["Champion"]==name)]
            if row.empty:
                display = ""
                color = COLOR_MAP["Default"]
            else:
                val = row.iloc[0]
                if val["Leave"]:
                    display = val["Leave"]
                    color = COLOR_MAP.get(val["Leave"], COLOR_MAP["WO"])
                elif val["Shift"]:
                    display = val["Shift"]
                    color = COLOR_MAP["Split"] if "Split" in val["Shift"] else COLOR_MAP["Straight"]
                else:
                    display = ""
                    color = COLOR_MAP["Default"]
            cells.append(f"<td style='padding:6px; background:{color}; text-align:center'>{display}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    # build header html
    header_html = "<tr><th style='padding:6px'>Name</th><th style='padding:6px'>Shift (base)</th>"
    for hd in header_dates:
        header_html += f"<th style='padding:6px'>{hd}</th>"
    header_html += "</tr>"
    table_html = f"""
    <table style='border-collapse:collapse; width:100%;'>
      <thead style='background:#f0f0f0; text-align:center'>{header_html}</thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """
    return table_html

st.markdown("---")
st.header("Final roster (color-coded view)")
if roster_long_df.empty:
    st.info("No assignments for the week. Adjust settings or champions.")
else:
    # Ensure roster_long_df has Date as datetime.date
    roster_long_df["Date"] = roster_long_df["Date"].apply(lambda x: x if isinstance(x, (str,)) else x)
    html = html_roster_table(roster_long_df, champ_names, dates)
    st.write("Legend: colored cells indicate shift vs leave. Edit the table above to adjust.")
    st.components.v1.html(html, height=600, scrolling=True)

# Downloads: long roster CSV and Excel planner
st.markdown("---")
st.header("Downloads")
# ensure Dates are strings for Excel
dl_long = roster_long_df.copy()
dl_long["Date"] = dl_long["Date"].astype(str)
buf = io.BytesIO()
with get_excel_writer(buf) as writer:
    dl_long.to_excel(writer, sheet_name="Roster_Long", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    predicted.to_excel(writer, sheet_name="Predicted_Calls", index=False)
buf.seek(0)
st.download_button("Download Planner (Excel)", buf, file_name="roster_planner.xlsx")
st.download_button("Download Roster (CSV)", dl_long.to_csv(index=False).encode("utf-8"), file_name="roster_long.csv")

st.caption("Notes: \n- AL% is computed as (AnsweredCalls / TotalCalls) × 100 with AnsweredCalls = (AssignedChamps × 470 minutes) / AHT_minutes.\n- Splits count toward the same 470 active minutes.\n- The allocation heuristic is greedy and fairness-based; you can edit the roster manually to adjust specific champions or leaves.\n- If some days do not reach the target AL% even with all champions assigned, consider adding more workforce or lowering AHT.")

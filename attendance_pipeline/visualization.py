import os, sys
import webbrowser
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from .utils import BASE_DIR
from .logger import get_logger
from .data_processing import parse_and_aggregate_attendance
from .forecasting import compute_typical_staffing, forecast_staff_by_group

logger = get_logger(__name__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_forecast_sequence(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    seq_len: int = 7,
    horizon: int = 5,
    title: str = "Staffing Forecast"
):
    detailed = forecast_staff_by_group(
        df,
        seq_len=seq_len,
        start_date=start_date,
        horizon=horizon
    )
    detailed["Date"] = pd.to_datetime(detailed["Date"])

    cal      = USFederalHolidayCalendar()
    holidays = set(cal.holidays(start=detailed.Date.min(),
                                end=detailed.Date.max()))
    detailed["dow"] = detailed["Date"].dt.weekday
    mask = (detailed["dow"] >= 5) | (detailed["Date"].isin(holidays))
    detailed.loc[mask, "StaffRequired"] = 0
    detailed.drop(columns="dow", inplace=True)

    table_cols = [
        "Date",
        "Place",
        "AssignedRoom",
        "Level",
        "ForecastHours",
        "ForecastStudents",
        "StaffRequired"
    ]
    header_labels = ["Date","Place","Room","Level","Forecast Hrs","# Students","Staff Required"]
    detailed_tbl = detailed[table_cols].sort_values(table_cols)

    table_traces = []
    table_traces.append(go.Table(
      header=dict(values=header_labels, fill_color="paleturquoise"),
      cells =dict(values=[detailed_tbl[c] for c in table_cols],
                  fill_color="lavender"),
      visible=True
    ))
    for p in detailed.Place.unique():
        sub = detailed_tbl[detailed_tbl.Place==p]
        table_traces.append(go.Table(
          header=dict(values=header_labels, fill_color="paleturquoise"),
          cells =dict(values=[sub[c] for c in table_cols],
                      fill_color="lavender"),
          visible=False
        ))

    places = detailed.Place.unique().tolist()
    levels = detailed.Level.unique().tolist()
    place_level = [
        (p,l)
        for p in places
        for l in levels
        if ((detailed.Place==p)&(detailed.Level==l)).any()
    ]
    place_rooms = (
        detailed[['Place','AssignedRoom','Level']]
        .drop_duplicates()
        .values
        .tolist()
    )

    chart_traces = []
    for p,l in place_level:
        agg = (
            detailed[(detailed.Place==p)&(detailed.Level==l)]
            .groupby('Date')['StaffRequired']
            .sum()
            .reset_index()
        )
        chart_traces.append(go.Scatter(
            x=agg["Date"],
            y=agg["StaffRequired"],
            mode="lines",
            name=f"{p} – {l} (Total)",
            line=dict(dash='dash'),
            visible=True
        ))
    for p, room, lvl in place_rooms:
        sub = detailed[(detailed.Place==p)&(detailed.AssignedRoom==room)]
        chart_traces.append(go.Scatter(
            x=sub["Date"],
            y=sub["StaffRequired"],
            mode="lines+markers",
            name=f"{p} – {room}",
            visible=False
        ))

    fig = make_subplots(
      rows=2, cols=1,
      row_heights=[0.3,0.7],
      specs=[[{"type":"table"}],[{"type":"scatter"}]]
    )
    for t in table_traces:
        fig.add_trace(t, row=1, col=1)
    for tr in chart_traces:
        fig.add_trace(tr, row=2, col=1)

    n_tbl = len(table_traces)
    n_ctr = len(chart_traces)
    buttons = []
    buttons.append(dict(
        label="All",
        method="update",
        args=[{"visible":[True]*n_tbl + [True]*n_ctr},
              {"title":"All"}]
    ))
    for i,p in enumerate(places):
        vis = [False]*(n_tbl + n_ctr)
        vis[i+1] = True
        for j,(pp,ll) in enumerate(place_level):
            if pp==p:
                vis[n_tbl + j] = True
        buttons.append(dict(
            label=p,
            method="update",
            args=[{"visible":vis}, {"title":p}]
        ))

    fig.update_layout(
      updatemenus=[dict(buttons=buttons,
                        direction="down",
                        x=0, y=1.2)],
      title=title,
      margin=dict(t=100)
    )
    fig.update_xaxes(row=2, col=1,
                     rangeslider=dict(visible=True),
                     type="date")
    fig.update_yaxes(row=2, col=1,
                     title_text="Staff Required")

    # save and open
    out = os.path.join(PLOTS_DIR, "forecast_sequence.html")
    plot(fig, filename=out,
         auto_open=False,
         include_plotlyjs="cdn")
    logger.info("Saved %s", out)
    webbrowser.open(f"file://{out}")

    return fig


def plot_typical_staffing_table(
    typical_df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    title:      str = "Typical Staffing"
):
    if start_date is None:
        start_date = pd.to_datetime("today").normalize()
    cal = USFederalHolidayCalendar()
    cbd = CustomBusinessDay(calendar=cal)
    first_bd = (
        start_date
        if (start_date.weekday() < 5 and
            not start_date in cal.holidays(start_date, start_date))
        else
        start_date + cbd
    )
    dates = pd.date_range(first_bd, periods=len(typical_df), freq=cbd)

    new_idx = [
        f"{dow} ({dt:%Y-%m-%d})"
        for dow, dt in zip(typical_df.index, dates)
    ]
    t = typical_df.copy()
    t.index = new_idx

    places = t.columns.get_level_values(0).unique().tolist()
    tables = []
    for i,p in enumerate(places):
        sub = t[p].reset_index()
        tables.append(go.Table(
            header=dict(values=list(sub.columns), fill_color="paleturquoise"),
            cells =dict(values=[sub[c] for c in sub.columns],
                        fill_color="lavender"),
            visible=(i == 0)
        ))

    buttons = []
    for i,p in enumerate(places):
        vis = [j == i for j in range(len(tables))]
        buttons.append(dict(
            label=p,
            method="update",
            args=[{"visible": vis}, {"title": f"{title} – {p}"}]
        ))

    fig = go.Figure(data=tables)
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.2)],
        title=f"{title} – {places[0]}",
        margin=dict(t=80,b=20)
    )

    out = os.path.join(PLOTS_DIR, "typical_staffing.html")
    plot(fig, filename=out, auto_open=False, include_plotlyjs="cdn")
    logger.info("Saved %s", out)
    webbrowser.open(f"file://{out}")

    return fig


def prompt_for_date(prompt="Enter forecast start date (YYYY-MM-DD):"):
    root = tk.Tk()
    root.withdraw()
    date_str = simpledialog.askstring(
        title="Forecast Date",
        prompt=prompt,
        initialvalue=datetime.today().strftime("%Y-%m-%d")
    )
    root.destroy()
    if not date_str:
        messagebox.showerror("No Date", "You must enter a date to continue.")
        sys.exit(1)
    try:
        return pd.to_datetime(date_str).normalize()
    except Exception:
        messagebox.showerror("Bad Date", f"Could not parse “{date_str}” as YYYY-MM-DD.")
        sys.exit(1)

if __name__ == "__main__":
    start = prompt_for_date()
    df_hist = parse_and_aggregate_attendance()

    typical = compute_typical_staffing(df_hist)
    plot_typical_staffing_table(typical, start_date=start)

    plot_forecast_sequence(df_hist, start_date=start, horizon=5)

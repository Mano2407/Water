from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import predict_goal_ml

BASE = Path(__file__).parent
LOG = BASE / "data" / "water_log.csv"
COLS = ["date", "time", "drink", "ml"]
DRINKS = {"Water": "💧", "Juice": "🧃", "Tender coconut": "🥥", "Coffee": "☕", "Tea": "🍵"}

st.set_page_config(page_title="My Water Tracker", page_icon="💧", layout="centered")
st.markdown(f"<style>{(BASE / 'styles.css').read_text()}</style>", unsafe_allow_html=True)


def load_log() -> pd.DataFrame:
    if LOG.exists():
        return pd.read_csv(LOG)
    return pd.DataFrame(columns=COLS)


def save_log(df: pd.DataFrame):
    LOG.parent.mkdir(exist_ok=True)
    df.to_csv(LOG, index=False)


def today_str() -> str:
    return date.today().isoformat()


def today_total(df: pd.DataFrame) -> int:
    return int(df[df["date"] == today_str()]["ml"].sum())


def add_drink(drink: str, ml: int):
    df = load_log()
    before = today_total(df)
    now = datetime.now()
    row = pd.DataFrame([[today_str(), now.strftime("%H:%M"), drink, int(ml)]], columns=COLS)
    save_log(pd.concat([df, row], ignore_index=True))
    goal = int(st.session_state.goal_l * 1000)
    if before < goal <= before + ml:
        st.session_state.celebrate = True


def undo_last():
    df = load_log()
    idx = df.index[df["date"] == today_str()]
    if len(idx):
        save_log(df.drop(idx[-1]))


def reset_today():
    df = load_log()
    save_log(df[df["date"] != today_str()])


def use_recommended():
    st.session_state.goal_l = predict_goal_ml(
        st.session_state.w, st.session_state.a, st.session_state.act, st.session_state.t
    ) / 1000


# ---------------- Sidebar: goal ----------------
st.session_state.setdefault("goal_l", 2.0)
with st.sidebar:
    st.header("🎯 Daily goal")
    st.number_input("Goal (litres)", 0.5, 10.0, step=0.1, key="goal_l")
    with st.expander("Recommend my goal"):
        st.number_input("Weight (kg)", 30.0, 200.0, 60.0, key="w")
        st.number_input("Age", 5, 100, 25, key="a")
        st.select_slider("Activity", options=[0, 1, 2], value=1, key="act",
                         format_func=lambda x: ["Low", "Medium", "High"][x])
        st.number_input("Temperature (°C)", 0.0, 50.0, 30.0, key="t")
        st.button("Use recommended goal", on_click=use_recommended)

st.title("💧 My Water Tracker")
st.caption(date.today().strftime("%A, %d %B %Y"))
tab_track, tab_report = st.tabs(["Tracker", "📊 Daily report"])

# ---------------- Tracker ----------------
with tab_track:
    drink = st.radio("Drink", list(DRINKS), horizontal=True,
                     format_func=lambda d: f"{DRINKS[d]} {d}")
    cols = st.columns(3)
    for col, (label, ml) in zip(cols, [("🥛 Glass · 250 ml", 250), ("🍶 Bottle · 500 ml", 500), ("💦 Sip · 100 ml", 100)]):
        if col.button(label, width="stretch"):
            add_drink(drink, ml)
            st.rerun()

    c1, c2 = st.columns([3, 1.4])
    custom = c1.number_input("Custom ml", 1, 3000, 300, step=50, label_visibility="collapsed")
    if c2.button("＋ Add custom", width="stretch"):
        add_drink(drink, custom)
        st.rerun()

    b1, b2 = st.columns(2)
    b1.button("↩ Undo last", on_click=undo_last, width="stretch")
    b2.button("🔄 Reset today", on_click=reset_today, width="stretch")

    df = load_log()
    goal = int(st.session_state.goal_l * 1000)
    total = today_total(df)
    left = max(0, goal - total)
    pct = min(1.0, total / goal)

    m1, m2, m3 = st.columns(3)
    m1.metric("Goal", f"{goal / 1000:.2f} L")
    m2.metric("Consumed", f"{total} ml")
    m3.metric("Remaining", f"{left} ml")
    st.progress(pct, text=f"{int(pct * 100)}% of daily goal")

    if total >= goal:
        st.markdown('<div class="msg done">🎉 Goal reached! Amazing work staying hydrated today!</div>', unsafe_allow_html=True)
    elif total == 0:
        st.markdown('<div class="msg">Let’s start hydrating!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="msg">Great going, keep sipping! 💪</div>', unsafe_allow_html=True)
    if st.session_state.pop("celebrate", False):
        st.balloons()

# ---------------- Report ----------------
with tab_report:
    df = load_log()
    td = df[df["date"] == today_str()]
    total = int(td["ml"].sum())
    r1, r2, r3 = st.columns(3)
    r1.metric("Total today", f"{total} ml")
    r2.metric("Drinks", len(td))
    r3.metric("Of goal", f"{min(100, round(total / goal * 100))}%")

    st.subheader("By drink")
    if len(td):
        st.bar_chart(td.groupby("drink")["ml"].sum())
    else:
        st.info("No drinks logged yet.")

    st.subheader("Last 7 days")
    days = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    daily = df.groupby("date")["ml"].sum().reindex(days, fill_value=0)
    st.bar_chart(daily)
    met = int((daily >= goal).sum())
    st.caption(f"Average {int(daily.mean())} ml/day · Goal met {met}/7 days")

    st.subheader("Today’s log")
    st.dataframe(td[["time", "drink", "ml"]].iloc[::-1], hide_index=True, width="stretch")
    st.download_button("⬇ Download full report (CSV)", df.to_csv(index=False), "water_report.csv", "text/csv")

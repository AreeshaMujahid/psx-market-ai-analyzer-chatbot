from __future__ import annotations
from openai import OpenAI
from pathlib import Path
import pandas as pd
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import os
import streamlit as st

import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


URL = "https://www.psx.com.pk/market-summary/#main"

CANON = ["SCRIP", "LDCP", "OPEN", "HIGH", "LOW", "CURRENT", "CHANGE", "VOLUME"]
CANON_SET = set(CANON)

# Load environment variables in a file called .env
load_dotenv(override=True)
# FIXED: Load API key from environment variable, not hardcoded
api_key = os.getenv("OPENAI_API_KEY")


def scrape_all_tables(url: str, wait_time: int = 15, headless: bool = True) -> list[pd.DataFrame]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            viewport={"width": 1280, "height": 720},
            locale="en-US",
        )
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(wait_time * 1000)
        html = page.content()
        browser.close()

    try:
        return pd.read_html(html)
    except ValueError:
        return []


def _norm(x) -> str:
    return str(x).replace("\xa0", " ").strip().upper()


def clean_table_to_stock_schema(df: pd.DataFrame) -> pd.DataFrame | None:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df.columns = [_norm(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).str.startswith("UNNAMED")]

    if len(df) > 0:
        row0 = [_norm(v) for v in df.iloc[0].tolist()]
        hits = sum(v in CANON_SET for v in row0)
        if hits >= 4:
            df.columns = row0
            df = df.iloc[1:].reset_index(drop=True)

    df.columns = [_norm(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    present = [c for c in CANON if c in df.columns]
    if len(present) < 6:
        return None

    out = df[present].copy()

    if "SCRIP" in out.columns:
        out["SCRIP"] = out["SCRIP"].astype(str).str.strip()
        out = out[out["SCRIP"].str.upper() != "SCRIP"]

    return out


def save_all_tables_single_sheet(
    tables: list[pd.DataFrame],
    out_dir: str = "psx_output",
    excel_name: str = "psx_stocks_single_sheet.xlsx",
):
    if not tables:
        return None

    cleaned = []
    skipped = 0

    for t in tables:
        fixed = clean_table_to_stock_schema(t)
        if fixed is None:
            skipped += 1
        else:
            cleaned.append(fixed)

    if not cleaned:
        return None

    combined_df = pd.concat(cleaned, ignore_index=True)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    excel_file = out_path / excel_name

    with pd.ExcelWriter(excel_file) as writer:
        combined_df.to_excel(writer, sheet_name="AllData", index=False)

    return combined_df


def messages_for(combined_df: pd.DataFrame, user_question: str):
    system_prompt = """
You are a PSX market analysis assistant. You will be given (1) a user question and (2) computed results produced by Python/pandas from PSX tabular data (columns like SCRIP, LDCP, OPEN, HIGH, LOW, CURRENT, CHANGE, VOLUME).

Rules:
- Use ONLY the provided computed results. Do not invent or assume any prices, volumes, ranks, symbols, sectors, dates, or trends.
- If the question requires data not present in the computed results, say exactly what is missing and what you can answer instead.
- Do NOT provide financial advice, buy/sell recommendations, or future predictions.
- Prefer short, clear explanations with bullet points.
- When citing numbers, repeat them exactly as provided.
Output format:
1) 1–2 line summary
2) Bullets with key observations
3) If needed: a short "Data limitations" note
""".strip()

    df_text = combined_df.to_string(index=False)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question.strip() + "\n\n" + df_text},
    ]


def summarize(combined_df: pd.DataFrame, user_question: str) -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # FIXED: Correct model name
        messages=messages_for(combined_df, user_question),
    )
    return response.choices[0].message.content


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="PSX Analyzer", layout="wide")
st.title("📈 PSX Market Summary — Live Scrape + AI Answer")

if not api_key:
    st.error("OPENAI_API_KEY not found. Put it in .env or set it in your environment.")
    st.stop()

# FIXED: Initialize session state to persist data
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None

user_question = st.text_input(
    "Ask a question (example: 'tell the high and low volume stocks in psx')",
    value="tell the high and low volume stocks in psx",
)

col1, col2 = st.columns(2)

with col1:
    headless = st.checkbox("Headless browser", value=True)
    wait_time = st.slider("Wait time (seconds)", min_value=3, max_value=30, value=15)

with col2:
    if st.button("▶ Run", type="primary"):
        with st.spinner("Scraping PSX tables..."):
            tables = scrape_all_tables(URL, wait_time=wait_time, headless=headless)
            combined_df = save_all_tables_single_sheet(tables)

        if combined_df is None or combined_df.empty:
            st.error("No stock-sector tables detected after cleaning. Try increasing wait time or disabling headless.")
            st.stop()

        # FIXED: Store in session state
        st.session_state.combined_df = combined_df
        st.success(f"Scraped {len(combined_df)} rows")

# FIXED: Display data and answer outside button block
if st.session_state.combined_df is not None:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Live Market Data")
        st.info(f"Scraped **{len(st.session_state.combined_df)}** rows")
        st.dataframe(st.session_state.combined_df, use_container_width=True, height=400)

    with col_right:
        st.subheader("AI Answer")
        
        if st.button("🤖 Get AI Analysis"):
            with st.spinner("Analyzing with AI..."):
                try:
                    answer = summarize(st.session_state.combined_df, user_question)
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"Error calling OpenAI: {str(e)}")

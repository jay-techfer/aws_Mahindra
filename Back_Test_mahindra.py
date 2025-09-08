import streamlit as st
import pandas as pd
import urllib
import pyodbc
import re
from sqlalchemy import create_engine
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import socket
# from dotenv import load_dotenv
# # from streamlit_plotly_events import plotly_events
# from PIL import Image
# from io import BytesIO
# import plotly.io as pio
# from fpdf import FPDF
# import io
# from PIL import Image
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import A4
# from cryptography.fernet import Fernet
import os
import time
import json

# Load session token securely
# if not os.path.exists("session_token.json"):
#     st.error("❌ No active session. Please log in.")
#     st.stop()

# with open("session_token.json", "r") as f:
#     session_data = json.load(f)

# encrypted_data = session_data.get('encrypted_data', None)

# # Your Fernet key (should match the one used in login.py)
# fernet_key = b'Sv_cBtT5H5i_fv3sPvRrAe_2z6WRnqbmq-rmfxUyiGQ='
# cipher_suite = Fernet(fernet_key)

# try:
#     # Decrypt and load session info
#     decrypted_text = cipher_suite.decrypt(encrypted_data.encode()).decode()
#     session_info = json.loads(decrypted_text)

#     username = session_info.get("username")
#     token = session_info.get("token")

#     # st.success(f"✅ Welcome, {username}")

#     # Optionally delete session file after successful load
#     # os.remove("session_token.json")

# except Exception as e:
#     st.error("❌ Decryption failed. Invalid or tampered token.")
#     st.stop()
# 🔹 Configure Gemini
genai.configure(api_key="AIzaSyC0T1vRMxg8r2Ma75sit71SWFHGyKpwRso")
model = genai.GenerativeModel("gemini-1.5-pro")

# 🔹 Streamlit config
st.set_page_config("DataGenie", layout="wide",
                   initial_sidebar_state="expanded")

# Optional: wider sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        max-width: 1000px;
        min-width: 500px;
        overflow-x: auto;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-right: 1rem;
    }

    .canvas-box {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 12px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        min-height: 400px;
    }
    </style>
""", unsafe_allow_html=True)


#  Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_result_df" not in st.session_state:
    st.session_state.query_result_df = pd.DataFrame()
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_query_columns" not in st.session_state:
    st.session_state.last_query_columns = []


# 🔹 SQL Server Config
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

ssms_servers = [
    {
        "name": "VIJAY\\SQLEXPRESS",   # or just "SQLEXPRESS"
        "server": "VIJAY\SQLEXPRESS,52235",       # dynamically set IP
        "username": "sa",
        "password": "abcd123456"
    }
]

ssms_schema_df = pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_ssms_schema():
    data = []
    for s in ssms_servers:
        try:
            base_conn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={
                s['server']};UID={
                s['username']};PWD={
                s['password']};Encrypt=no;"
            dbs = ["AdventureWorks2022"]  # ✅ Only fetch AdventureWorks

            for db in dbs:
                db_conn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={
                    s['server']};Database={db};UID={
                    s['username']};PWD={
                    s['password']};Encrypt=no;"
                engine = create_engine(
                    f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(db_conn)}")
                df = pd.read_sql(
                    "SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS",
                    engine
                )
                df['SERVER'] = s['name']
                df['DATABASE'] = db
                data.append(df)
        except Exception as e:
            st.warning(f"SSMS Error: {e}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()


def build_chat_context():
    conversation = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            conversation.append(f"User: {msg['message']}")
        elif msg["role"] == "assistant":
            conversation.append(f"Assistant: {msg['message']}")
    return "\n".join(conversation)


def gen_join_queries(user_input, ssms_schema, history=""):
    user_input_add = user_input.strip().rstrip('.') 
    schema_description = ""
    if os.path.exists("parameters.json"):
                        with open("parameters.json", "r", encoding="utf-8") as f:
                            param_info = json.load(f)
                            schema_description += "[parameters.json]\n"
                            schema_description += "\n".join([
                                f"- {item['Parameter']}: {item.get('Description', item.get('Descripsition', 'No description'))}"
                                for item in param_info
                            ])
                            schema_description += "\n"
    else:
        schema_description += "[parameters.json]\nSchema details not available.\n"
                
                    # Handle the other flat dictionary JSON files
    flat_json_files = ["parts.json", "labour.json", "verbatim.json"]

    for file in flat_json_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                schema_description += f"[{file}]\n"
                for key, value in data.items():
                    schema_description += f"- {key}: {value}\n"
        else:
            schema_description += f"[{file}]\nSchema details not available.\n"

                    # Prompt for model
        history_text = f"Conversation so far:\n{history}\n\n"            
        prompt = f"""
        Their previous chat was:{history_text}
You are a helpful assistant that generates valid and optimized T-SQL queries for SQL Server.
The user is working with tables called `dbo.sampledata`, `dbo.parts`, `dbo.labour`, and `dbo.verbatim` inside the `Mahindra` database.

Schema details:
{schema_description}

Strict Instructions:
- Do NOT use CTEs (e.g., WITH ... AS) or `ROW_NUMBER()` functions.
- Do NOT use `TOP` in the `SELECT` clause **unless the user specifically requests a limit** (e.g., "top 5", "top 10", etc.).
- Always use `SELECT TOP XX` format instead of `ROW_NUMBER()` for limiting rows.
- Use `COUNT(*) AS Repetition_Count` when showing how many times something occurred.
- For analyzing most frequent entries, use `GROUP BY` followed by `ORDER BY COUNT(*) DESC`.
- Use clear aliases like `Source_Type` and `Item_Description` when combining records from different tables.
- For date filtering, use: `RO_Date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`.
- Do not wrap multiple GROUP BYs inside subqueries unless required — keep it simple and flat.
- Output only the query and nothing else.
- while generating the query use [Mahindra].[dbo].[table_name]

if '{user_input_add}' is 'give me data of Top 5 Repeating VoC / Part / Labour Descriptions' or something like that:
then Return SQL Query :
SELECT 'Customer Verbatim' AS Source_Type, Item_Description, Repetition_Count
FROM (
    SELECT TOP 5 CUSTMR_VERBTM AS Item_Description, COUNT(*) AS Repetition_Count
    FROM [Mahindra].[dbo].[verbatim]
    GROUP BY CUSTMR_VERBTM
    ORDER BY Repetition_Count DESC
) AS VerbatimTop5
UNION ALL
-- Top 5 Parts
SELECT 'Part' AS Source_Type, Item_Description, Repetition_Count
FROM (
    SELECT TOP 5 PART_DESC AS Item_Description, COUNT(*) AS Repetition_Count
    FROM [Mahindra].[dbo].[parts]
    GROUP BY PART_DESC
    ORDER BY Repetition_Count DESC
) AS PartsTop5
UNION ALL
SELECT 'Labor' AS Source_Type, Item_Description, Repetition_Count
FROM (
    SELECT TOP 5 LABR_DESC AS Item_Description, COUNT(*) AS Repetition_Count
    FROM [Mahindra].[dbo].[labour]
    GROUP BY LABR_DESC
    ORDER BY Repetition_Count DESC
) AS LaborTop5;



User Question:
{user_input_add}

Return only the SQL query starting after this line:
-- SSMS Query Start
"""
    return model.generate_content(prompt).text

def detect_mode(user_text):
    classification_prompt = f"""
    You are an intent classifier.
    The user said: "{user_text}"

    Classify the intent into one of the following categories:

    - "Query": When the user asks for raw data, SQL, data fetching, tables, aggregations or database queries.
    - "Descriptive": When the user wants summaries, facts, trends, or general descriptions about what the data shows.
    - "Diagnostic": When the user wants explanations or reasons behind data trends or results.
    - "Predictive": When the user asks for forecasts or predictions based on data.
    - "Prescriptive": When the user wants actionable suggestions, decisions, or recommendations based on the data.

    Examples:
    - "Get total sales by year" → Query
    - "Summarize the yearly sales trend" → Descriptive
    - "Why did sales drop in 2019?" → Diagnostic
    - "Predict 2025 revenue using this data" → Predictive
    - "What should we focus on next year to increase sales?" → Prescriptive
    DO NOT GIVE PYTHON CODE
    Output ONLY one of the following words: Query, Descriptive, Diagnostic, Predictive, Prescriptive.
    """
    response = model.generate_content(classification_prompt).text.strip()
    return response


history_text = build_chat_context()

# 🔹 Fetch Schema Once
if ssms_schema_df.empty:
    with st.spinner("Fetching SSMS schema..."):
        ssms_schema_df = fetch_ssms_schema()

# 🔹 Logo
st.image("techfer_logo_new.png", width=200)

with st.sidebar:
    st.markdown("""
    <h1 style='font-size: 40px; color: #2C3E50; margin-bottom: 10px;'>DataGenie</h1>
    """, unsafe_allow_html=True)

    if not st.session_state.query_result_df.empty:
        new_df1 = st.session_state.query_result_df
        new_df1.reset_index(drop=True, inplace=True)
        new_df1.index = new_df1.index + 1

        # === Initialize States ===
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = "data"
        if "chart_metadata" not in st.session_state:
            st.session_state["chart_metadata"] = []

        # === Tab Buttons ===
        colA, colB = st.columns([1, 4])
        with colA:
            if st.button("📊 Data"):
                st.session_state.active_tab = "data"
        with colB:
            if st.button("📈 Visualize"):
                st.session_state.active_tab = "viz"

        # === Data Tab ===
        if st.session_state.active_tab == "data":
            # if st.session_state.last_query:
            #     with st.expander("📎Query       "):
            #         st.code(st.session_state.last_query, language="sql")
            st.dataframe(new_df1)

        # === Visualization Tab ===
        if st.session_state.active_tab == "viz":

            if "last_df_shape" not in st.session_state or st.session_state["last_df_shape"] != new_df1.shape:
                st.session_state.pop("generated_chart_code", None)
                st.session_state["last_df_shape"] = new_df1.shape

            x_axis_cols = st.multiselect(
                "📌 Select X-axis columns",
                new_df1.columns.tolist(),
                default=st.session_state.get("x_axis_cols", [])
            )
            y_axis_cols = st.multiselect(
                "📌 Select Y-axis columns",
                new_df1.columns.tolist(),
                default=st.session_state.get("y_axis_cols", [])
            )
            # adv_widgets = st.multiselect(
            #     "⚙️ Add advanced Plotly widgets",
            #     ["Range Slider", "Range Selector", "Animation",
            #      "Dropdown Menu", "Hover Compare", "Crossfilter"],
            #     default=st.session_state.get("adv_widgets", [])
            # )
            chart_prompt = st.text_area(
                "📝 Describe the chart you want to generate",
                value=st.session_state.get("chart_prompt", "")
            )

            st.subheader("📈 Gemini Chart Canvas")

            if st.button("🎨 Create Chart"):
                if not x_axis_cols or not y_axis_cols or not chart_prompt:
                    st.warning(
                        "Select X & Y columns and enter chart description.")
                else:
                    st.session_state["x_axis_cols"] = x_axis_cols
                    st.session_state["y_axis_cols"] = y_axis_cols
                    # st.session_state["adv_widgets"] = adv_widgets
                    st.session_state["chart_prompt"] = chart_prompt

                    x_list = ", ".join(x_axis_cols)
                    y_list = ", ".join(y_axis_cols)

                    chart_gen_prompt = f"""
                        You are a Python data visualization assistant.

                        The user wants a chart based on this request: {chart_prompt}

                        Selected columns from the DataFrame named `df`:
                        - X-axis: {x_list}
                        - Y-axis: {y_list}

                        Instructions:
                        - Use the existing DataFrame `df` as-is. Do not create or redefine `df` or generate any mock/sample data.
                        - Use Plotly Express or Plotly Graph Objects.
                        - If widgets are selected, integrate them.
                        - Before plotting, drop any rows where required columns (like X, Y, hierarchy path, or value columns) are null, NaN, or blank strings ('').
                        - Output only the Python code inside a markdown code block.
                        """

                    response = model.generate_content(chart_gen_prompt).text
                    print("chart code : ", response)
                    chart_code = re.search(
                        r"```python(.*?)```", response, re.DOTALL)

                    if chart_code:
                        st.session_state["generated_chart_code"] = chart_code.group(
                            1).strip()
                    else:
                        st.error("⚠️ Couldn't parse chart code.")

            # === Create & Store Charts ===
            if "generated_chart_code" in st.session_state:
                try:
                    exec_globals = {"pd": pd, "df": new_df1,
                                    "px": px, "go": go, "np": np}
                    exec(
                        st.session_state["generated_chart_code"], exec_globals)

                    new_figs = [
                        exec_globals[name]
                        for name in exec_globals
                        if re.match(r"fig\d*$", name) and isinstance(exec_globals[name], go.Figure)
                    ]

                    if new_figs:
                        for fig in new_figs:
                            # ✅ Add this to render new chart
                            st.plotly_chart(fig, use_container_width=True)

                        st.session_state["chart_metadata"].append({
                            "code": st.session_state["generated_chart_code"],
                            "x_cols": x_axis_cols,
                            "y_cols": y_axis_cols
                        })

                    st.session_state.pop("generated_chart_code", None)

                except Exception as e:
                    st.error("❌ Chart rendering failed.")
                    st.exception(e)

            if st.session_state["chart_metadata"]:
                st.subheader("📊 Created Charts")

                # Column Filters
                df = st.session_state.query_result_df
                filter_cols = st.multiselect(
                    "Select columns to filter", df.columns.tolist()
                )

                filters = {}
                for col in filter_cols:
                    unique_vals = sorted(df[col].dropna().unique())
                    selected_vals = st.multiselect(
                        f"Filter {col}", unique_vals, default=unique_vals
                    )
                    filters[col] = selected_vals

                # Apply filters
                filtered_df = df.copy()
                for col, vals in filters.items():
                    filtered_df = filtered_df[filtered_df[col].isin(vals)]

                # Display last 6 charts using filtered_df as df
                grid_cols = st.columns(3)
                # Directly loop over the last 6 chart entries with their true indices
                for display_i, chart_index in enumerate(range(max(0, len(st.session_state["chart_metadata"]) - 6), len(st.session_state["chart_metadata"]))):
                    meta = st.session_state["chart_metadata"][chart_index]
                    exec_globals = {"pd": pd, "df": filtered_df,
                                    "px": px, "go": go, "np": np}
                    exec(meta["code"], exec_globals)
                    fig = next(v for v in exec_globals.values()
                               if isinstance(v, go.Figure))

                    with grid_cols[display_i % 3]:
                        delete_key = f"delete_chart_{chart_index}"
                        if st.button("❌", key=delete_key):
                            st.session_state["chart_metadata"].pop(chart_index)
                            st.rerun()

                        st.plotly_chart(fig, use_container_width=True,
                                        key=f"chart_{display_i}")
            else:
                st.info(
                    "No chart generated yet. Use the controls above to create one.")

        else:
            st.info("Please request data to generate chart!!")


def sanitize_gemini_response(text):
    text = re.sub(r'</?div[^>]*>', '', text)
    return text.strip()


for msg in st.session_state.chat_history:
    if msg["role"] == "separator":
        st.markdown("<hr style='border: 1px solid #ccc;'>",
                    unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        clean_text = sanitize_gemini_response(msg['message'])
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-start; margin-bottom: 20px;'>
            <div style='background-color: #e6f3ff; padding: 10px 14px; border-radius: 15px 15px 15px 0; max-width: 80%; white-space: pre-wrap; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                {clean_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif msg["role"] == "user":
        clean_text = sanitize_gemini_response(msg['message'])
        st.markdown(f"""
        <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
            <div style='background-color: #E8E8E8; padding: 10px 14px; border-radius: 15px 15px 0 15px; max-width: 80%; white-space: pre-wrap; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                {clean_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Bottom Chat Input ---
user_input = st.chat_input("Type your question...")

if user_input:
    mode = detect_mode(user_input)   # <-- Auto mode detection here
    print(mode)
    if mode == "Query":
        # # Store query-related context
        # st.session_state.query_context.append(user_input)
        # === QUERY MODE ===
        schema_text = re.sub(
            r'\s{2,}', ' ', ssms_schema_df.to_string(index=False).strip())
        sql_text = gen_join_queries(
            user_input, schema_text, history_text)
        # print("sql query:", sql_text)
        cleaned_output = sql_text.replace("sql", "").strip()
        match = re.search(r"--\s*SSMS Query Start\s*(.*)",
                          cleaned_output, re.DOTALL | re.IGNORECASE)

        if match:
            query = match.group(1).strip("`").rstrip(";").strip()
            print("query:", query)
            st.session_state.last_query = query

            st.session_state.chat_history.append(
                {"role": "separator", "message": "---"})
            st.session_state.chat_history.append(
                {"role": "user", "message": user_input})

            server_cfg = ssms_servers[0]
            conn_str = (
                f"Driver={{ODBC Driver 17 for SQL Server}};"
                f"Server={server_cfg['server']};"
                f"UID={server_cfg['username']};"
                f"PWD={server_cfg['password']};"
                f"Encrypt=no;TrustServerCertificate=yes;"
            )
            quoted_conn = urllib.parse.quote_plus(conn_str)
            engine = create_engine(
                f"mssql+pyodbc:///?odbc_connect={quoted_conn}")
            df = pd.read_sql(query, engine)
            st.session_state.query_result_df = df
            st.success("✅ Data fetched and ready to analyze")
        else:
            st.error("❌ Could not parse valid SQL query.")

    else:
        # === CHAT MODE ===
        st.session_state.chat_history.append(
            {"role": "user", "message": user_input}
        )

        # ✅ Build conversation history for context

        print("""
                show me :

              """, history_text)
        if not st.session_state.query_result_df.empty:
            df = st.session_state.query_result_df
            prompt = f"""
            Conversation so far:
            {history_text}

                You are a data consultant. Your role is to analyze all the given data and answer the user's question with clear, actionable insights.
                - Do not give python code
                - If Asked for any kind of calculation, consider the whole data and give the answer
                User question: "{user_input}"
                Data:
                {df.to_markdown(index=True)}

                Respond in bullet points with clear insights.
            """
        else:
            prompt = f"""
                Conversation so far:
                {history_text}

                You are a domain expert in all fields and Data Consultant expert.
                User asked: \"{user_input}\"
                Respond in 4-5 bullet points with useful analysis/suggestions.
                """

        reply = model.generate_content(prompt).text
        st.session_state.chat_history.append(
            {"role": "assistant", "message": reply}
        )

    st.rerun()

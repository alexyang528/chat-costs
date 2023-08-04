import streamlit as st
import pandas as pd
# import snowflake.connector
import plotly.express as px

st.set_page_config(
    page_title="Chat Pricing Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Chat Pricing Analysis")

# List of available models
MODELS = [
    "gpt-3.5-4k",
    "gpt-3.5-16k",
    "gpt-4-8k",
    "gpt-4-32k"
]

# Price per 1000 tokens
PROMPT_PRICES = {
    "gpt-3.5-4k": 0.0015,
    "gpt-3.5-16k": 0.003,
    "gpt-4-8k": 0.03,
    "gpt-4-32k": 0.06
}

OUTPUT_PRICES = {
    "gpt-3.5-4k": 0.002,
    "gpt-3.5-16k": 0.004,
    "gpt-4-8k": 0.06,
    "gpt-4-32k": 0.12
}


# @st.cache_data
# def get_prompt_completions():
#     query = """
#         select
#             business_id,
#             chat_bot_name,
#             model_request_id,
#             chat_response_id,
#             prompt_name,
#             raw_response:usage.prompt_tokens as prompt_tokens,
#             raw_response:usage.completion_tokens as output_tokens,
#             raw_response:usage.total_tokens as total_tokens
#         from prod_chat.public.chat_prompt_completions;
#     """

#     conn = snowflake.connector.connect(account="tw61901.us-east-1",
#                                        authenticator="externalbrowser")
    
#     curs = conn.cursor()
#     curs.execute(query)
#     df = curs.fetch_pandas_all()

#     return df

# Fetch data
# DATA = get_prompt_completions()
DATA = pd.read_csv("chat_prompt_completions.csv")

# Select the chatbot
st.sidebar.write("# Filters")
chatbot = st.sidebar.multiselect(label="Chatbot", options=DATA["CHAT_BOT_NAME"].unique(), default=["hitchhikers-chat"])

# Select model for each prompt
st.sidebar.write("# Models per Prompt")
predictStepDigression = st.sidebar.selectbox(label="predictStepDigression", options=MODELS, index=0)
predictGoal = st.sidebar.selectbox(label="predictGoal", options=MODELS, index=0)
predictSearchQuery = st.sidebar.selectbox(label="predictSearchQuery", options=MODELS, index=0)
detectResultContent = st.sidebar.selectbox(label="detectResultContent", options=MODELS, index=1)
answerQuestion = st.sidebar.selectbox(label="answerQuestion", options=MODELS, index=1)
evaluateCondition = st.sidebar.selectbox(label="evaluateCondition", options=MODELS, index=0)
casualResponse = st.sidebar.selectbox(label="casualResponse", options=MODELS, index=0)
parseFields = st.sidebar.selectbox(label="parseFields", options=MODELS, index=2)
collectFields = st.sidebar.selectbox(label="collectFields", options=MODELS, index=0)
noData = st.sidebar.selectbox(label="noData", options=MODELS, index=0)

# Build dictionary of pormpts and models
PROMPT_MODELS = {
    "predictStepDigression": predictStepDigression,
    "predictGoal": predictGoal,
    "predictSearchQuery": predictSearchQuery,
    "detectResultContent": detectResultContent,
    "answerQuestion": answerQuestion,
    "evaluateCondition": evaluateCondition,
    "casualResponse": casualResponse,
    "parseFields": parseFields,
    "collectFields": collectFields,
    "noData": noData
}

# Drop any columsn with null token counts
DATA = DATA.dropna(subset=["PROMPT_TOKENS", "OUTPUT_TOKENS"])

# Convert to numeric
DATA["PROMPT_TOKENS"] = DATA["PROMPT_TOKENS"].astype(int)
DATA["OUTPUT_TOKENS"] = DATA["OUTPUT_TOKENS"].astype(int)
DATA["TOTAL_TOKENS"] = DATA["TOTAL_TOKENS"].astype(int)


# Calculate cost column
DATA["MODEL"] = DATA["PROMPT_NAME"].map(PROMPT_MODELS)
DATA["COST"] = (
    DATA["MODEL"].map(PROMPT_PRICES) * DATA["PROMPT_TOKENS"].astype(int) / 1000
    + DATA["MODEL"].map(OUTPUT_PRICES) * DATA["OUTPUT_TOKENS"].astype(int) / 1000
)

# Filter data and group by response
prompt_logs = DATA[DATA["CHAT_BOT_NAME"].isin(chatbot)]
response_logs = prompt_logs.groupby("CHAT_RESPONSE_ID").agg(
    {
        "BUSINESS_ID": "first",
        "CHAT_BOT_NAME": "first",
        "MODEL_REQUEST_ID": "nunique",
        "PROMPT_NAME": "first",
        "PROMPT_TOKENS": "sum",
        "OUTPUT_TOKENS": "sum",
        "TOTAL_TOKENS": "sum",
        "COST": "sum",
    }
)

col1, col2 = st.columns(2)

with col1:
    st.write("## Per Prompt Completion")
    st.write("Prompt Completions are individual requests made to OpenAI. Request are charged by total number of tokens in the prompt (the input) and the completion (the output).")

    st.write("### Tokens")
    st.metric(label="Average", value="{:.0f}".format(prompt_logs["TOTAL_TOKENS"].mean()))
    fig = px.histogram(prompt_logs, x="TOTAL_TOKENS", nbins=50)
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, height=800, use_container_width=True)

    st.write("### Cost")
    st.metric(label="Average", value="${:.4f}".format(prompt_logs["COST"].mean()))
    fig = px.histogram(prompt_logs, x="COST", nbins=50)
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, height=800, use_container_width=True)

    with st.expander("View Logs"):
        st.dataframe(prompt_logs)
        st.download_button(
            "Download Prompt Data",
            prompt_logs.to_csv().encode("utf-8"),
            file_name="prompt_logs.csv",
        )

with col2:
    st.write("## Per Response")
    st.write("Responses are messages returned in the Chat interface to the user. Every Response may have more than one Prompt Completion.")

    st.write("### Prompt Completions")
    st.metric(label="Average", value="{:.0f}".format(response_logs["MODEL_REQUEST_ID"].mean()))
    fig = px.histogram(response_logs, x="MODEL_REQUEST_ID", nbins=50)
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, height=800, use_container_width=True)

    st.write("### Cost")
    st.metric(label="Average", value="${:.4f}".format(response_logs["COST"].mean()))
    fig = px.histogram(response_logs, x="COST", nbins=50)
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, height=800, use_container_width=True)

    with st.expander("View Logs"):
        st.dataframe(response_logs)
        st.download_button(
            "Download Response Data",
            response_logs.to_csv().encode("utf-8"),
            file_name="response_logs.csv",
        )

st.write("---")
st.write("## Prompt Completion Task Type")
st.write("### Tokens")
fig = px.strip(
    prompt_logs,
    x="PROMPT_NAME",
    y="TOTAL_TOKENS",
)
# Sort by task ascending
fig.update_layout(
    xaxis={"categoryorder": "array", "categoryarray": prompt_logs["PROMPT_NAME"].unique()}
)

st.plotly_chart(fig, height=800, use_container_width=True)
st.write("### Average Cost")
temp = prompt_logs.groupby("PROMPT_NAME")["COST"].mean().reset_index()
fig = px.bar(
    temp,
    x="PROMPT_NAME",
    y="COST",
    height=800,
)
fig.update_layout(
    xaxis={"categoryorder": "array", "categoryarray": prompt_logs["PROMPT_NAME"].unique()}
)
st.plotly_chart(fig, height=800, use_container_width=True)

st.write("---")
st.write("## Bot Configurations")
st.write("### Tokens per Response")
temp = response_logs.groupby(["CHAT_BOT_NAME", "CHAT_RESPONSE_ID"])["TOTAL_TOKENS"].mean().reset_index()
temp = temp.groupby("CHAT_BOT_NAME")["TOTAL_TOKENS"].mean().reset_index()
temp.sort_values("TOTAL_TOKENS", inplace=True)
fig = px.bar(
    temp,
    x="CHAT_BOT_NAME",
    y="TOTAL_TOKENS",
    height=800,
)
st.plotly_chart(fig, height=800, use_container_width=True)

st.write("### Cost per Response")
temp = response_logs.groupby(["CHAT_BOT_NAME", "CHAT_RESPONSE_ID"])["COST"].mean().reset_index()
temp = temp.groupby("CHAT_BOT_NAME")["COST"].mean().reset_index()
temp.sort_values("COST", inplace=True)
fig = px.bar(
    temp,
    x="CHAT_BOT_NAME",
    y="COST",
    height=800,
)
st.plotly_chart(fig, height=800, use_container_width=True)

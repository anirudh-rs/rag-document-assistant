import streamlit as st
import random
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PL 24/25 Assist",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

:root {
    --pl-purple:  #38003c;
    --pl-pink:    #e90052;
    --pl-cyan:    #04f5ff;
    --pl-white:   #ffffff;
    --pl-muted:   #9b8fa0;
}

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: var(--pl-purple) !important;
    color: var(--pl-white) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

.block-container {
    padding: 2rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* Sticky header not supported reliably in Streamlit — removed */

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #2a0030 !important;
    border-right: 1px solid rgba(233,0,82,0.2) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

.sidebar-header {
    padding: 1.2rem 1rem 0.8rem 1rem;
    border-bottom: 1px solid rgba(233,0,82,0.2);
    text-align: center;
}
.sidebar-header .sidebar-emoji { font-size: 2rem; line-height: 1; }
.sidebar-header h1 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em;
    color: var(--pl-white) !important;
    margin: 0.3rem 0 0.1rem 0 !important;
    line-height: 1.1 !important;
}
.sidebar-header span {
    font-size: 0.65rem;
    color: var(--pl-pink);
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.sidebar-note {
    margin: 0.9rem 0.8rem 0 0.8rem;
    padding: 0.7rem 0.9rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-left: 3px solid rgba(255,255,255,0.35);
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #c8b8cc;
    line-height: 1.6;
}
.sidebar-note strong {
    display: block;
    font-size: 0.86rem;
    color: var(--pl-white);
    margin-bottom: 0.25rem;
    font-weight: 600;
}

.teams-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--pl-muted);
    padding: 0.9rem 1rem 0.4rem 1rem;
}

/* Sidebar team row buttons — invisible overlay */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: transparent !important;
    font-size: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    height: 36px !important;
    min-height: 0 !important;
    width: 100% !important;
    box-shadow: none !important;
    cursor: pointer !important;
    position: relative !important;
    z-index: 2 !important;
    margin-top: -36px !important;
    display: block !important;
}
section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] .stButton > button:focus {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 0 !important;
    height: 0 !important;
}

/* ── Main header ── */
.pl-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(233,0,82,0.2);
}
.pl-header-emoji { font-size: 2.4rem; line-height: 1; flex-shrink: 0; }
.pl-header-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    color: var(--pl-white);
    line-height: 1;
}
.pl-header-title span { color: var(--pl-pink); }
.pl-header-sub {
    font-size: 0.7rem;
    color: var(--pl-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Example cycling text */
.examples-bar {
    padding-bottom: 0.9rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 1.2rem;
}

/* Chat */
[data-testid="stChatMessage"] { padding: 0.5rem 0 !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent {
    background: rgba(233,0,82,0.1) !important;
    border: 1px solid rgba(233,0,82,0.2) !important;
    border-radius: 12px 12px 2px 12px !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stChatMessageContent {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 2px 12px 12px 12px !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    margin-top: 6px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.74rem !important;
    color: var(--pl-muted) !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stChatInput"] {
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.04) !important;
}
[data-testid="stChatInput"]:focus-within { border-color: rgba(233,0,82,0.4) !important; }
[data-testid="stChatInput"] textarea {
    color: var(--pl-white) !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--pl-muted) !important;
    font-style: italic !important;
}
[data-testid="stChatInputSubmitButton"] { background: var(--pl-pink) !important; border-radius: 8px !important; }

.source-pill {
    display: inline-block;
    background: rgba(4,245,255,0.07);
    border: 1px solid rgba(4,245,255,0.18);
    border-radius: 4px;
    padding: 3px 9px;
    font-size: 0.72rem;
    color: var(--pl-cyan);
    margin: 2px 4px 2px 0;
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 0.04em;
}

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: rgba(233,0,82,0.25); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
TEAMS = [
    {"name": "Arsenal",        "emoji": "🔴"},
    {"name": "Aston Villa",    "emoji": "🟣"},
    {"name": "Bournemouth",    "emoji": "🍒"},
    {"name": "Brentford",      "emoji": "🐝"},
    {"name": "Brighton",       "emoji": "🦅"},
    {"name": "Chelsea",        "emoji": "🔵"},
    {"name": "Crystal Palace", "emoji": "🦅"},
    {"name": "Everton",        "emoji": "💙"},
    {"name": "Fulham",         "emoji": "⚫"},
    {"name": "Ipswich",        "emoji": "🔵"},
    {"name": "Leicester",      "emoji": "🦊"},
    {"name": "Liverpool",      "emoji": "❤️"},
    {"name": "Man City",       "emoji": "🩵"},
    {"name": "Man United",     "emoji": "😈"},
    {"name": "Newcastle",      "emoji": "⚫"},
    {"name": "Nott'm Forest",  "emoji": "🌳"},
    {"name": "Southampton",    "emoji": "⚪"},
    {"name": "Tottenham",      "emoji": "🐓"},
    {"name": "West Ham",       "emoji": "⚒️"},
    {"name": "Wolves",         "emoji": "🐺"},
]

TEAM_EXAMPLES = {
    "Arsenal":        ["When was Arsenal founded?", "Who manages Arsenal?", "What is the Emirates capacity?", "How many PL titles have Arsenal won?", "What is Arsenal's nickname?", "Who are Arsenal's biggest rivals?", "What colour do Arsenal play in?", "Has Arsenal ever won the Champions League?"],
    "Aston Villa":    ["When was Aston Villa founded?", "Who manages Aston Villa?", "What is Villa Park's capacity?", "Have Villa ever won the European Cup?", "What is Aston Villa's nickname?", "What colours do Villa play in?", "Who are Aston Villa's rivals?", "How many FA Cups have Villa won?"],
    "Bournemouth":    ["When was Bournemouth founded?", "What is Bournemouth's stadium called?", "Who manages Bournemouth?", "What is Bournemouth's nickname?", "Where is Bournemouth based?", "What colours does Bournemouth play in?", "What league were Bournemouth in before the PL?", "How big is the Vitality Stadium?"],
    "Brentford":      ["When was Brentford founded?", "What is Brentford's stadium called?", "Who manages Brentford?", "What is Brentford's nickname?", "Where do Brentford play?", "What colours does Brentford play in?", "Who are Brentford's rivals?", "What is the capacity of Brentford's ground?"],
    "Brighton":       ["When was Brighton founded?", "What is Brighton's stadium called?", "Who manages Brighton?", "What is Brighton's nickname?", "What colours does Brighton play in?", "Where is Brighton based?", "Has Brighton ever won the FA Cup?", "What is the Amex Stadium's capacity?"],
    "Ipswich":        ["When was Ipswich founded?", "What is Portman Road's capacity?", "Who manages Ipswich?", "What is Ipswich's nickname?", "What colours does Ipswich play in?", "When did Ipswich last win the league?", "Who are Ipswich's biggest rivals?", "Has Ipswich ever won a European trophy?"],
    "Leicester":      ["When was Leicester founded?", "What is the King Power Stadium's capacity?", "Who manages Leicester?", "What is Leicester's nickname?", "When did Leicester win the Premier League?", "What colours does Leicester play in?", "Who are Leicester's rivals?", "Has Leicester won any European trophies?"],
    "Chelsea":        ["How many PL titles have Chelsea won?", "Who manages Chelsea?", "What is Stamford Bridge's capacity?", "When was Chelsea founded?", "What is Chelsea's nickname?", "Has Chelsea won the Champions League?", "What colours does Chelsea play in?", "Who are Chelsea's biggest rivals?"],
    "Crystal Palace": ["When was Crystal Palace founded?", "What is Selhurst Park's capacity?", "Who manages Crystal Palace?", "What is Palace's nickname?", "What colours does Crystal Palace play in?", "Where is Crystal Palace based?", "Who are Palace's rivals?", "Has Crystal Palace ever won the FA Cup?"],
    "Everton":        ["When was Everton founded?", "What is Everton's new stadium called?", "Who manages Everton?", "How many league titles have Everton won?", "What is Everton's nickname?", "What colours does Everton play in?", "Who are Everton's rivals?", "Where did Everton play before the new stadium?"],
    "Fulham":         ["When was Fulham founded?", "What is Craven Cottage's capacity?", "Who manages Fulham?", "What is Fulham's nickname?", "What colours does Fulham play in?", "Where is Fulham based?", "Who are Fulham's rivals?", "Has Fulham ever won a major trophy?"],

    "Liverpool":      ["How many PL titles have Liverpool won?", "Who manages Liverpool?", "What is Anfield's capacity?", "When was Liverpool founded?", "What is Liverpool's nickname?", "Has Liverpool won the Champions League?", "What colours does Liverpool play in?", "Who are Liverpool's biggest rivals?"],
    "Man City":       ["How many PL titles have Man City won?", "Who manages Man City?", "What is the Etihad's capacity?", "When was Man City founded?", "What is Man City's nickname?", "Has Man City won the Champions League?", "What colours does Man City play in?", "Who are Man City's biggest rivals?"],
    "Man United":     ["How many PL titles have Man United won?", "Who manages Man United?", "What is Old Trafford's capacity?", "When was Man United founded?", "What is Man United's nickname?", "Has Man United won the Champions League?", "What colours does Man United play in?", "Who are Man United's biggest rivals?"],
    "Newcastle":      ["When was Newcastle founded?", "What is St James' Park's capacity?", "Who manages Newcastle?", "What is Newcastle's nickname?", "What colours does Newcastle play in?", "Have Newcastle ever won the league?", "Where is Newcastle based?", "Who are Newcastle's rivals?"],
    "Nott'm Forest":  ["When was Nottingham Forest founded?", "Who manages Forest?", "Have Forest won the European Cup?", "What is Forest's home ground?", "What is Forest's nickname?", "What colours does Nottingham Forest play in?", "What is the City Ground's capacity?", "Who are Forest's biggest rivals?"],
    "Southampton":    ["When was Southampton founded?", "What is St Mary's Stadium's capacity?", "Who manages Southampton?", "What is Southampton's nickname?", "What colours does Southampton play in?", "Has Southampton ever won the league?", "Who are Southampton's biggest rivals?", "What is Southampton's record PL finish?"],
    "Tottenham":      ["When was Tottenham founded?", "What is the Spurs Stadium's capacity?", "Who manages Spurs?", "How many league titles have Spurs won?", "What is Spurs' nickname?", "What colours does Tottenham play in?", "Has Tottenham ever won the Champions League?", "Who are Spurs' biggest rivals?"],
    "West Ham":       ["When was West Ham founded?", "What is the London Stadium's capacity?", "Who manages West Ham?", "What is West Ham's nickname?", "What colours does West Ham play in?", "Has West Ham ever won the FA Cup?", "Who are West Ham's rivals?", "Where did West Ham play before the London Stadium?"],
    "Wolves":         ["When was Wolves founded?", "What is Molineux's capacity?", "Who manages Wolves?", "How many league titles have Wolves won?", "What is Wolves' nickname?", "What colours does Wolves play in?", "Who are Wolves' biggest rivals?", "Has Wolves ever played in European competition?"],
}

GENERAL_EXAMPLES = [
    "Who won the 2024/25 Premier League?",
    "Who were the top scorers in 2024/25?",
    "Which clubs were relegated in 2024/25?",
    "Which clubs were promoted in 2024/25?",
    "Which London clubs are in the PL?",
    "Which club has the most PL titles?",
    "Who finished second in 2024/25?",
    "Which PL clubs have won the European Cup?",
    "What was the biggest result in 2024/25?",
    "Which manager was sacked first in 2024/25?",
    "Compare Man City and Liverpool trophies",
    "What is the biggest stadium in the PL?",
]

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_team" not in st.session_state:
    st.session_state.selected_team = None
if "examples_for" not in st.session_state:
    st.session_state.examples_for = None
if "examples" not in st.session_state:
    st.session_state.examples = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    selected = st.session_state.selected_team
    team_data = next((t for t in TEAMS if t["name"] == selected), None) if selected else None

    header_emoji = team_data["emoji"] if team_data else "⚽"
    header_title = f"{selected.upper()}<br>ASSIST" if selected else "PL 24/25<br>ASSIST"

    st.markdown(f"""
    <div class="sidebar-header">
        <div class="sidebar-emoji">{header_emoji}</div>
        <h1>{header_title}</h1>
        <span>Premier League Intelligence</span>
    </div>
    <div class="sidebar-note">
        <strong>📋 Note</strong>
        Records cover the 2024/25 Premier League season only. Information may not reflect current squads, managers or standings.
    </div>
    <div class="teams-label">2024/25 Clubs</div>
    """, unsafe_allow_html=True)

    for team in TEAMS:
        is_selected = selected == team["name"]
        is_dimmed = selected is not None and not is_selected

        opacity = "1" if is_selected else ("0.28" if is_dimmed else "0.85")
        bg = "rgba(233,0,82,0.13)" if is_selected else "transparent"
        border_left = "3px solid #e90052" if is_selected else "3px solid transparent"
        name_color = "#ffffff" if is_selected else "#c8b8cc"
        name_weight = "600" if is_selected else "400"

        radio = (
            '<svg width="13" height="13" viewBox="0 0 13 13">'
            '<circle cx="6.5" cy="6.5" r="5.5" fill="none" stroke="#e90052" stroke-width="1.5"/>'
            '<circle cx="6.5" cy="6.5" r="3" fill="#e90052"/>'
            '</svg>'
        ) if is_selected else (
            '<svg width="13" height="13" viewBox="0 0 13 13">'
            '<circle cx="6.5" cy="6.5" r="5.5" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="1.5"/>'
            '</svg>'
        )

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;
                    padding:5px 0.9rem;height:36px;
                    opacity:{opacity};background:{bg};
                    border-left:{border_left};box-sizing:border-box;">
            <span style="font-size:1.15rem;line-height:1;width:24px;text-align:center;">{team['emoji']}</span>
            <span style="flex:1;font-size:0.9rem;font-weight:{name_weight};
                         color:{name_color};">{team['name']}</span>
            {radio}
        </div>
        """, unsafe_allow_html=True)

        if st.button("x", key=f"btn_{team['name']}", use_container_width=True):
            st.session_state.selected_team = None if is_selected else team["name"]
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.examples = []
            st.session_state.examples_for = None
            st.rerun()

    st.markdown("""
    <div style="padding:1rem;border-top:1px solid rgba(255,255,255,0.07);
                font-size:0.6rem;color:#9b8fa0;text-align:center;line-height:1.6;margin-top:0.5rem;">
        Powered by RAG + GPT-4o<br>Sources: Wikipedia · 20 clubs
    </div>
    """, unsafe_allow_html=True)

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    st.session_state.debug_mode = st.toggle(
        "🔍 Debug sources", value=st.session_state.debug_mode
    )

# ── RAG chain ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)

db = load_db()
llm = load_llm()

def get_context(question, selected_team=None):
    """
    Retrieve docs with higher k, then reorder so priority sources appear first.
    Always injects the standings table for season/position/comparison questions.
    """
    if selected_team:
        expanded = f"{selected_team} 2024/25 Premier League season: {question}"
    else:
        expanded = f"2024/25 Premier League season: {question}"

    docs = db.similarity_search(expanded, k=14)

    if not docs:
        return []

    if selected_team:
        team_file_map = {
            "Arsenal": "arsenal", "Aston Villa": "aston_villa",
            "Bournemouth": "bournemouth", "Brentford": "brentford",
            "Brighton": "brighton", "Chelsea": "chelsea",
            "Crystal Palace": "crystal_palace", "Everton": "everton",
            "Fulham": "fulham", "Ipswich": "ipswich",
            "Leicester": "leicester", "Liverpool": "liverpool",
            "Man City": "manchester_city", "Man United": "manchester_united",
            "Newcastle": "newcastle", "Nott'm Forest": "nottingham_forest",
            "Southampton": "southampton", "Tottenham": "tottenham",
            "West Ham": "west_ham", "Wolves": "wolves",
        }
        priority_key = team_file_map.get(selected_team, "").lower()
    else:
        priority_key = "pl_24_25"

    priority = [d for d in docs if priority_key in d.metadata.get("source", "").lower()]
    others   = [d for d in docs if priority_key not in d.metadata.get("source", "").lower()]
    reordered = (priority + others)[:10]

    # For position/comparison/season questions, always inject the standings table
    position_keywords = ["finish", "position", "place", "season", "better", "compare",
                         "table", "standing", "points", "relegated", "qualified"]
    if any(kw in question.lower() for kw in position_keywords):
        standings_docs = db.similarity_search(
            "2024/25 Premier League final standings table positions points", k=3
        )
        standings = [d for d in standings_docs
                     if "pl_24_25" in d.metadata.get("source", "").lower()]
        # Prepend standings chunks, avoiding duplicates
        existing_content = {d.page_content[:80] for d in reordered}
        for d in standings:
            if d.page_content[:80] not in existing_content:
                reordered.insert(0, d)
                existing_content.add(d.page_content[:80])

    return reordered

def make_prompt(team=None):
    if team:
        system = f"""You are PL Assist, a Premier League football intelligence assistant covering the 2024/25 season.
The user is focused on {team}.

RULES — follow these exactly:
1. Answer ONLY using facts explicitly stated in the context below. No inference, no guessing.
2. The context contains two types of chunks: (a) historical club wiki pages, (b) the 2024/25 season Wikipedia article. Treat them separately.
3. For questions about the 2024/25 season — use ONLY the 2024/25 season chunks. Do NOT use historical club wiki chunks to fill gaps.
4. For questions about league position or finishing place — use the standings table in the 2024/25 chunks. The table lists position by number. Read it carefully and state the exact position.
5. Do NOT combine historical facts with 2024/25 facts to construct an answer.
6. If the 2024/25 context does not explicitly answer the question, say: "I don't have enough detail about this in my 2024/25 records." Do not guess."""
    else:
        system = """You are PL Assist, a Premier League football intelligence assistant covering the 2024/25 season.

RULES — follow these exactly:
1. Answer ONLY using facts explicitly stated in the context below. No inference, no guessing.
2. The context contains two types of chunks: (a) historical club wiki pages, (b) the 2024/25 season Wikipedia article. Treat them separately.
3. For questions about the 2024/25 season — use ONLY the 2024/25 season chunks. Do NOT use historical club wiki chunks to fill gaps.
4. For questions about league position or finishing place — use the standings table in the 2024/25 chunks. The table lists position by number. Read it carefully and state the exact position.
5. Do NOT combine historical facts with 2024/25 facts to construct an answer.
6. If the 2024/25 context does not explicitly answer the question, say: "I don't have enough detail about this in my 2024/25 records." Do not guess."""

    return ChatPromptTemplate.from_messages([
        ("system", system + "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

def format_docs(docs):
    chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if "pl_24_25" in source.lower() or "PL_24_25" in source:
            label = "[2024/25 Season Data]"
        else:
            label = "[Club History]"
        chunks.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)

def get_sources(docs):
    name_map = {
        "pl_24_25": "Premier League 2024/25",
        "arsenal": "Arsenal", "aston_villa": "Aston Villa",
        "bournemouth": "Bournemouth", "brentford": "Brentford",
        "brighton": "Brighton", "chelsea": "Chelsea",
        "crystal_palace": "Crystal Palace", "everton": "Everton",
        "fulham": "Fulham", "ipswich": "Ipswich",
        "leicester": "Leicester", "liverpool": "Liverpool",
        "manchester_city": "Man City", "manchester_united": "Man United",
        "newcastle": "Newcastle", "nottingham_forest": "Nott'm Forest",
        "southampton": "Southampton", "tottenham": "Tottenham",
        "west_ham": "West Ham", "wolves": "Wolves",
    }
    seen = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        source = doc.metadata.get("source", "unknown")
        key = source.split("/")[-1].replace(".pdf", "").lower()
        filename = name_map.get(key, key.replace("_", " ").title())
        label = f"{filename} — p.{page}"
        if label not in seen:
            seen.append(label)
    return seen

def run_chain(question):
    selected_team = st.session_state.selected_team
    prompt = make_prompt(selected_team)
    history = st.session_state.chat_history[-6:]

    # Smart context retrieval
    docs = get_context(question, selected_team)
    context = format_docs(docs)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
        "chat_history": history,
    })
    return {"answer": answer, "docs": docs}

# ── Main UI ───────────────────────────────────────────────────────────────────
selected = st.session_state.selected_team
team_data = next((t for t in TEAMS if t["name"] == selected), None) if selected else None
header_emoji = team_data["emoji"] if team_data else "⚽"

if selected and team_data:
    header_html = f"""
    <div class="pl-header">
        <div class="pl-header-emoji">{header_emoji}</div>
        <div>
            <div class="pl-header-title"><span>{selected}</span> ASSIST</div>
            <div class="pl-header-sub">Premier League 2024/25 · Club Focus</div>
        </div>
    </div>"""
else:
    header_html = f"""
    <div class="pl-header">
        <div class="pl-header-emoji">{header_emoji}</div>
        <div>
            <div class="pl-header-title">PL 24/25 <span>ASSIST</span></div>
            <div class="pl-header-sub">Premier League 2024/25 · All 20 Clubs</div>
        </div>
    </div>"""

# ── Question handler ──────────────────────────────────────────────────────────
def handle_question(question):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner(""):
            result = run_chain(question)
            answer = result["answer"]
            sources = get_sources(result["docs"])
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                st.markdown("".join(f'<span class="source-pill">{s}</span>'
                                    for s in sources), unsafe_allow_html=True)
        if st.session_state.get("debug_mode"):
            with st.expander("🔍 Debug — retrieved chunks"):
                for i, doc in enumerate(result["docs"]):
                    src = doc.metadata.get("source", "?")
                    st.caption(f"**Chunk {i+1}** — `{src}`")
                    st.text(doc.page_content[:300])
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.chat_history = st.session_state.chat_history[-6:]

# Dynamic example questions — CSS fade cycle
if st.session_state.examples_for != selected or not st.session_state.examples:
    pool = TEAM_EXAMPLES.get(selected, GENERAL_EXAMPLES) if selected else GENERAL_EXAMPLES
    st.session_state.examples = random.sample(pool, min(6, len(pool)))
    st.session_state.examples_for = selected

examples = st.session_state.examples
styles = ""
questions_html = ""

if examples:
    n = len(examples)
    slot = 4
    cycle = n * slot
    pct_in   = (slot * 0.2  / cycle) * 100
    pct_hold = (slot * 0.75 / cycle) * 100
    pct_out  = (slot * 0.95 / cycle) * 100

    for idx, q in enumerate(examples):
        styles += f"""
@keyframes fadeq{idx} {{
    0%            {{ opacity:0; transform:translateY(5px); }}
    {pct_in:.2f}%   {{ opacity:1; transform:translateY(0); }}
    {pct_hold:.2f}% {{ opacity:1; transform:translateY(0); }}
    {pct_out:.2f}%  {{ opacity:0; transform:translateY(-5px); }}
    100%          {{ opacity:0; }}
}}"""
        neg_delay = -(idx * slot)
        questions_html += (
            f'<div style="position:absolute;left:0;right:0;opacity:0;'
            f'font-style:italic;font-size:0.82rem;color:var(--pl-muted);'
            f'line-height:24px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'animation:fadeq{idx} {cycle}s linear {neg_delay}s infinite;">'
            f'Ask &ldquo;{q}&rdquo;</div>'
        )

examples_html = f'<div style="position:relative;height:24px;">{questions_html}</div>' if questions_html else ""

# Inject keyframe animations
if styles:
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)

# Header
st.markdown(header_html, unsafe_allow_html=True)

# Examples
if examples_html:
    st.markdown(f'<div style="margin-bottom:1rem;">{examples_html}</div>', unsafe_allow_html=True)

# Message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                st.markdown("".join(f'<span class="source-pill">{s}</span>'
                                    for s in msg["sources"]), unsafe_allow_html=True)

if selected:
    placeholder = f'Ask about {selected} — e.g. "Who managed {selected} in 2024/25?"'
else:
    placeholder = 'Ask in full sentences — e.g. "Who won the 2024/25 Premier League?"'
if question := st.chat_input(placeholder):
    handle_question(question)
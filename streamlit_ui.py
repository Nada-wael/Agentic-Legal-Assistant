"""
streamlit_ui.py
===============
واجهة المستخدم للمساعد القانوني الذكي
تدعم وضعين:
  1. Agentic Workflow (LangGraph) — الـ Agent يقرر الأدوات بنفسه
  2. RAG Direct (LCEL)            — بحث مباشر في قاعدة المعرفة بدون Agent
"""

import uuid
from operator import itemgetter

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver

from agentic_workflow import (
    create_workflow,
    DEMO_CONTRACT,
    get_retriever,
    llm,
    is_api_key_valid,
)

# ─── إعداد الصفحة ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="المساعد القانوني الذكي",
    page_icon="⚖️",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════════
# Cached Resources — تُبنى مرة واحدة فقط وتُخزَّن طوال عمر التطبيق
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_workflow_app():
    """
    يبني الـ LangGraph app مرة واحدة ويحفظه في cache.
    MemorySaver يخزّن الـ checkpoints في RAM — لا قاعدة بيانات.
    """
    workflow = create_workflow()
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


@st.cache_resource
def get_rag_chain():
    """
    يبني سلسلة RAG مباشرة (LCEL) بدون Agent.

    السلسلة:
      السؤال
        → retriever (يجلب أقرب 3 chunks)
        → format_docs (يدمج النصوص)
        → PromptTemplate (يضع السياق والسؤال)
        → LLM (يولّد الإجابة)
        → StrOutputParser (يستخرج النص من رد الـ LLM)
    """
    if not is_api_key_valid:
        return None

    try:
        retriever = get_retriever()

        def format_docs(docs):
            """يدمج محتوى الـ chunks في نص واحد مفصول بسطر فارغ."""
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = PromptTemplate.from_template(
            "أنت مساعد قانوني ذكي متخصص في قانون العمل المصري.\n"
            "أجب على السؤال التالي بناءً على السياق القانوني المرفق فقط.\n"
            "إذا لم تجد الإجابة في السياق، قل ذلك صراحةً.\n\n"
            "السياق القانوني:\n{context}\n\n"
            "السؤال: {question}\n\n"
            "الإجابة:"
        )

        # LCEL pipe — يربط كل خطوة بالتالية باستخدام |
        rag_chain = (
            {
                # itemgetter("question") يسحب السؤال من القاموس المدخل
                # ثم يمرره للـ retriever ثم format_docs
                "context": itemgetter("question") | retriever | format_docs,
                "question": itemgetter("question"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain
    except Exception as e:
        # تسجيل الخطأ والعودة بـ None
        print(f"Error initializing RAG chain: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Session State — يحفظ البيانات بين كل rerun في Streamlit
# ═══════════════════════════════════════════════════════════════════════════
if "thread_id" not in st.session_state:
    # كل جلسة لها thread_id فريد → ذاكرة مستقلة في MemorySaver
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "contract_text" not in st.session_state:
    st.session_state.contract_text = ""

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚖️ المساعد القانوني الذكي")
    st.caption("AI Legal Assistant — Powered by Gemini + LangGraph")

    st.markdown("---")

    # ─── اختيار الوضع ────────────────────────────────────────────────────
    st.markdown("### ⚙️ وضع التشغيل")
    mode = st.radio(
        "كيف يعالج المساعد سؤالك؟",
        [
            "🤖 Agentic Workflow (LangGraph)",
            "📚 RAG Direct (LCEL)",
        ],
        help=(
            "Agentic: الـ LLM يقرر أي أدوات يستخدم بنفسه.\n"
            "RAG Direct: يبحث مباشرة في قاعدة المعرفة القانونية."
        ),
    )

    st.markdown("---")

    # ─── هيكل الـ Graph ───────────────────────────────────────────────────
    st.markdown("### 🗺️ هيكل الـ Graph")
    st.code(
        "START\n"
        "  ↓\n"
        "[agent_node]\n"
        "  ↓\n"
        "should_continue()\n"
        "  ├─ tools → [ToolNode] ─┐\n"
        "  │                      │\n"
        "  │         ┌────────────┘\n"
        "  │         ↓\n"
        "  │     [agent_node]\n"
        "  └─ END",
        language="text",
    )

    st.markdown("---")

    # ─── الأدوات المتاحة ──────────────────────────────────────────────────
    st.markdown("### 🔧 الأدوات المتاحة")
    st.markdown("""
1. `pdf_parser` — هيكل العقد وعدد المواد
2. `clause_detector` — اكتشاف البنود الأساسية
3. `risk_analyzer` — تحليل المخاطر (🔴🟡🟢)
4. `legal_knowledge_retriever` — RAG: قاعدة المعرفة
5. `web_search` — بحث في الإنترنت
    """)

    st.markdown("---")

    # ─── التقنيات المستخدمة ───────────────────────────────────────────────
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
- **LangChain** — Tools, Prompts, LCEL
- **LangGraph** — StateGraph, ToolNode, MemorySaver
- **RAG** — TextLoader, FAISS, Embeddings
- **Gemini 2.5 Flash** — LLM
- **Streamlit** — UI
    """)

    st.markdown("---")

    # ─── مسح المحادثة ────────────────────────────────────────────────────
    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.contract_text = ""
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# الصفحة الرئيسية
# ═══════════════════════════════════════════════════════════════════════════
st.title("🤖 المساعد القانوني الذكي")
st.caption("تحليل العقود العربية، اكتشاف المخاطر، والاستشارات القانونية")

if not is_api_key_valid:
    st.warning(
        "⚠️ **مفتاح API غير صالح أو مفقود!** يرجى تحديث ملف `.env` ووضع مفتاح API الخاص بـ Gemini في حقل `GOOGLE_API_KEY` لتشغيل المساعد الذكي.\n\n"
        "⚠️ **Invalid or missing API key!** Please update the `.env` file and set the `GOOGLE_API_KEY` variable with your Gemini API key to run the legal assistant."
    )

# ─── أزرار سريعة ─────────────────────────────────────────────────────────
st.markdown("### ⚡ إجراءات سريعة")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📄 تحميل عقد تجريبي", use_container_width=True):
        st.session_state.contract_text = DEMO_CONTRACT
        st.rerun()

with col2:
    if st.button("🔍 تحليل شامل", use_container_width=True):
        st.session_state.quick_prompt = (
            "قم بتحليل العقد خطوة بخطوة: استخرج الهيكل، اكتشف البنود، وحلّل المخاطر."
        )

with col3:
    if st.button("⚠️ المخاطر فقط", use_container_width=True):
        st.session_state.quick_prompt = (
            "حلّل مخاطر العقد وصنّفها إلى (HIGH / MEDIUM / LOW)."
        )

with col4:
    if st.button("📋 استخراج البنود", use_container_width=True):
        st.session_state.quick_prompt = (
            "استخرج جميع البنود الأساسية من العقد."
        )

# ─── نص العقد ────────────────────────────────────────────────────────────
st.markdown("### 📝 نص العقد")
contract_text = st.text_area(
    "أدخل نص العقد هنا (أو استخدم الزر أعلاه لتحميل عقد تجريبي):",
    value=st.session_state.contract_text,
    height=180,
    placeholder="اكتب أو الصق نص العقد هنا...",
)
st.session_state.contract_text = contract_text

st.markdown("---")

# ─── عرض سجل المحادثة ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            st.caption(f"🛠️ الأدوات المستخدمة: `{'`, `'.join(msg['tools_used'])}`")

# ─── مربع الدردشة ────────────────────────────────────────────────────────
prompt = st.chat_input("اكتب سؤالك القانوني أو اطلب تحليل العقد...", disabled=not is_api_key_valid)

# التعامل مع الأزرار السريعة
if "quick_prompt" in st.session_state:
    prompt = st.session_state.pop("quick_prompt")

# ═══════════════════════════════════════════════════════════════════════════
# معالجة السؤال
# ═══════════════════════════════════════════════════════════════════════════
if prompt:
    # عرض رسالة المستخدم
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # دمج نص العقد مع السؤال إذا وجد
    full_query = prompt
    if contract_text.strip():
        full_query = f"{prompt}\n\nنص العقد:\n{contract_text.strip()}"

    with st.chat_message("assistant"):
        with st.spinner("المساعد القانوني يفكر..."):

            # ─── وضع 1: Agentic Workflow ──────────────────────────────────
            if "Agentic" in mode:
                app = get_workflow_app()
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                used_tools = []
                final_answer = ""

                for event in app.stream(
                    {"messages": [HumanMessage(content=full_query)]},
                    config,
                    stream_mode="values",
                ):
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        if last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                used_tools.append(tc["name"])
                        elif last_msg.content:
                            final_answer = last_msg.content

                st.markdown(final_answer)
                used_tools = list(dict.fromkeys(used_tools))  # إزالة التكرار مع الحفاظ على الترتيب
                if used_tools:
                    st.caption(f"🛠️ الأدوات المستخدمة: `{'`, `'.join(used_tools)}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "tools_used": used_tools,
                })

            # ─── وضع 2: RAG Direct (LCEL) ─────────────────────────────────
            else:
                rag_chain = get_rag_chain()
                if rag_chain is None:
                    response = "⚠️ لا يمكن تشغيل وضع RAG المباشر بدون مفتاح API صالح. يرجى تعديل ملف `.env` ووضع مفتاح API الخاص بـ Gemini في حقل `GOOGLE_API_KEY`."
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "tools_used": [],
                    })
                else:
                    response = rag_chain.invoke({"question": full_query})
                    st.markdown(response)
                    st.caption("🛠️ الأداة المستخدمة: `legal_knowledge_retriever` (RAG Direct)")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "tools_used": ["legal_knowledge_retriever"],
                    })

"""
agentic_workflow.py
===================
المساعد القانوني الذكي — الملف الرئيسي
يحتوي على:
  1. LangChain Tools  (@tool decorator)
  2. RAG Setup        (TextLoader → Splitter → FAISS → Retriever → Tool)
  3. LLM + bind_tools (Function Calling)
  4. AgentState       (TypedDict + Annotated)
  5. LangGraph Nodes & Edges
  6. MemorySaver      (in-RAM memory)
"""

import operator
import os
import sys
from typing import Annotated, Sequence, TypedDict

# تهيئة الترميز للغة العربية والإيموجي على ويندوز
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
# from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ─── تحميل متغيرات البيئة (GOOGLE_API_KEY) ───────────────────────────────
load_dotenv()

# التحقق من صحة مفتاح API لتفادي كراش gRPC عند وجود نص عربي كافتراضي
api_key = os.environ.get("GOOGLE_API_KEY", "")
is_api_key_valid = True

if not api_key or api_key.strip() == "":
    is_api_key_valid = False
else:
    # إزالة علامات الاقتباس إن وجدت ومقارنتها بالقيم الافتراضية الشائعة
    clean_key = api_key.strip().strip("'\"")
    placeholders = ["ضع_مفتاح_API_هنا", "YOUR_API_KEY", "YOUR_API_KEY_HERE", "your_api_key", "your_api_key_here", "ضع API Key في .env"]
    if clean_key in placeholders:
        is_api_key_valid = False

# ═══════════════════════════════════════════════════════════════════════════
# 1. LLM  —  نموذج اللغة الأساسي
# ═══════════════════════════════════════════════════════════════════════════
# نستخدم Gemini-2.5-flash لأنه سريع وكافي لمهام تحليل العقود
if is_api_key_valid:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
else:
    # نستخدم مفتاحاً مؤقتاً بالإنجليزية لتجنب كراش gRPC بسبب الحروف العربية
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="MISSING_OR_INVALID_KEY")



# ═══════════════════════════════════════════════════════════════════════════
# 2. RAG  —  قاعدة المعرفة القانونية
# ═══════════════════════════════════════════════════════════════════════════
def get_retriever():
    """
    يبني retriever من ملف النصوص القانونية.

    الخطوات:
      1. TextLoader  → يحمّل النص من الملف
      2. RecursiveCharacterTextSplitter → يقسّم النص إلى chunks
         chunk_size=1000: كل chunk لا يتجاوز 1000 حرف
         chunk_overlap=200: تداخل 200 حرف بين الـ chunks لضمان عدم فقدان السياق
      3. GoogleGenerativeAIEmbeddings → يحوّل كل chunk لـ vector
      4. FAISS.from_documents → يخزّن الـ vectors في RAM (لا قاعدة بيانات)
      5. as_retriever(k=3) → عند البحث يجلب أقرب 3 chunks
    """
    if not is_api_key_valid:
        raise ValueError("مفتاح Google API مفقود أو غير صالح. يرجى تعديل ملف .env ووضع مفتاح صالح.")

    # تحديد مسار الملف بجانب هذا السكريبت
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base_dir, "mediumblog1.txt")

    if os.path.exists(txt_path):
        loader = TextLoader(txt_path, encoding="utf-8")
        docs = loader.load()
    else:
        # نص احتياطي إذا لم يوجد الملف
        from langchain_core.documents import Document
        docs = [Document(
            page_content=(
                "مدة عدم المنافسة لا تتجاوز سنتين وفق قانون العمل المصري. "
                "الإنهاء بدون إشعار يُلزم بتعويض. "
                "الغرامات المفرطة تُعدّ باطلة. "
                "شرط السرية مدى الحياة مبالغ فيه."
            )
        )]

    # تقسيم النص إلى chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # تحويل الـ chunks إلى vectors وتخزينها في FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # k=3 → يجلب أقرب 3 chunks عند البحث
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def setup_rag():
    """
    يحوّل الـ retriever إلى Tool يمكن للـ Agent استخدامه.
    create_retriever_tool() تعطي الأداة اسماً ووصفاً يفهمه الـ LLM.
    """
    retriever = get_retriever()
    return create_retriever_tool(
        retriever,
        name="legal_knowledge_retriever",
        description=(
            "يبحث في قاعدة المعرفة القانونية المصرية عن نصوص قانونية وأحكام "
            "وتعريفات. استخدمه عند الحاجة للرجوع لقانون العمل أو معرفة الحدود القانونية."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. LangChain Tools  —  الأدوات المخصصة للمساعد القانوني
# ═══════════════════════════════════════════════════════════════════════════

@tool
def pdf_parser(contract_text: str) -> str:
    """
    يحلّل هيكل نص العقد ويستخرج معلومات أساسية:
      - عدد المواد (كل جملة تبدأ بـ'المادة' تُحسب)
      - معاينة أول 150 حرف
      - إجمالي طول النص
    استخدمه أولاً لفهم بنية العقد قبل التحليل التفصيلي.
    """
    sections_count = contract_text.count("المادة")
    total_chars = len(contract_text)
    preview = contract_text[:150].strip()
    return (
        f"📄 تحليل هيكل العقد:\n"
        f"  • عدد المواد: {sections_count}\n"
        f"  • إجمالي الأحرف: {total_chars}\n"
        f"  • معاينة: {preview}..."
    )


@tool
def clause_detector(contract_text: str) -> str:
    """
    يكتشف البنود الأساسية الموجودة في العقد:
      - شروط الدفع (راتب، جنيه، دولار)
      - شروط الإنهاء
      - عدم المنافسة
      - السرية
      - تسوية المنازعات / التحكيم
    يعيد قائمة بالبنود المكتشفة أو رسالة إذا لم يجد شيئاً.
    """
    detected = []

    if any(kw in contract_text for kw in ["راتب", "جنيه", "دولار", "أجر", "مكافأة"]):
        detected.append("💰 شروط الدفع والأجر")

    if any(kw in contract_text for kw in ["إنهاء", "الإنهاء", "فسخ"]):
        detected.append("🔚 شروط الإنهاء")

    if any(kw in contract_text for kw in ["عدم المنافسة", "منافسة", "منافس"]):
        detected.append("🚫 شرط عدم المنافسة")

    if any(kw in contract_text for kw in ["السرية", "سرية", "سري", "خاص"]):
        detected.append("🔒 شرط السرية")

    if any(kw in contract_text for kw in ["تحكيم", "التحكيم", "محكمة", "قضاء", "نزاع"]):
        detected.append("⚖️ تسوية المنازعات")

    if any(kw in contract_text for kw in ["إجازة", "مرضية", "سنوية"]):
        detected.append("🏖️ الإجازات")

    if not detected:
        return "⚠️ لم يتم اكتشاف بنود رئيسية واضحة — تأكد من وجود نص العقد."

    return "البنود المكتشفة في العقد:\n" + "\n".join(f"  {b}" for b in detected)


@tool
def risk_analyzer(contract_text: str) -> str:
    """
    يحلّل العقد ويكتشف المخاطر القانونية مع تصنيفها:
      🔴 HIGH   — مخاطر عالية تحتاج تدخل فوري
      🟡 MEDIUM — مخاطر متوسطة تستحق المراجعة
      🟢 LOW    — لا مخاطر واضحة
    يبحث عن: عدم منافسة مفرط، إنهاء بدون إشعار، غرامات مبالغ فيها،
              سرية مدى الحياة، اختصاص قضائي أجنبي.
    """
    risks = []

    # فحص عدم المنافسة المفرط
    if "أي مكان في العالم" in contract_text:
        risks.append("🔴 HIGH: شرط عدم منافسة بنطاق جغرافي مفتوح (أي مكان في العالم) — باطل قانوناً.")
    if "5 سنوات" in contract_text or "خمس سنوات" in contract_text:
        risks.append("🔴 HIGH: مدة عدم منافسة تتجاوز السنتين — يتجاوز الحد القانوني المسموح.")

    # فحص الإنهاء بدون إشعار
    if any(kw in contract_text for kw in ["دون إشعار", "بدون إشعار", "في أي وقت دون"]):
        risks.append("🔴 HIGH: إنهاء العقد في أي وقت دون إشعار أو تعويض — مخالف لقانون العمل.")

    # فحص الغرامات المفرطة
    if any(kw in contract_text for kw in ["500,000", "مليون", "غرامات"]):
        risks.append("🔴 HIGH: غرامات مالية مفرطة — قد تُعدّ تعسفية وتقبل التخفيض قضائياً.")

    # فحص السرية المبالغ فيها
    if "مدى الحياة" in contract_text:
        risks.append("🟡 MEDIUM: شرط سرية مدى الحياة — مبالغ فيه وقابل للطعن.")

    # فحص الاختصاص القضائي الأجنبي
    if any(kw in contract_text for kw in ["قانون إنجلترا", "لندن", "نيويورك", "أجنبي"]):
        risks.append("🟡 MEDIUM: اختصاص قضائي أجنبي — يعني تكاليف قانونية مرتفعة للعامل.")

    if not risks:
        return "🟢 LOW: لم يتم اكتشاف مخاطر قانونية واضحة في النص."

    summary = f"تم اكتشاف {len(risks)} مخاطر:\n"
    summary += "\n".join(f"  {r}" for r in risks)
    return summary


# ─── أداة البحث على الإنترنت ─────────────────────────────────────────────
# DuckDuckGoSearchRun: تبحث في الإنترنت بدون API Key
web_search = DuckDuckGoSearchRun()
web_search.name = "web_search"
web_search.description = (
    "يبحث في الإنترنت عن معلومات قانونية حديثة، أحكام قضائية، أو أي معلومة "
    "غير موجودة في قاعدة المعرفة. استخدمه للأسئلة التي تحتاج معلومات متجددة."
)

# ─── تجميع كل الأدوات ───────────────────────────────────────────────────
try:
    retriever_tool = setup_rag()
    tools = [pdf_parser, clause_detector, risk_analyzer, retriever_tool, web_search]
    print("✅ RAG جاهز — جميع الأدوات محمّلة.")
except Exception as e:
    print(f"⚠️ خطأ في RAG: {e} — سيعمل بدون retriever_tool.")
    tools = [pdf_parser, clause_detector, risk_analyzer, web_search]


# ═══════════════════════════════════════════════════════════════════════════
# 4. LLM + Tools  —  ربط الأدوات بالنموذج (Function Calling)
# ═══════════════════════════════════════════════════════════════════════════
# bind_tools() تُعلّم الـ LLM بوجود الأدوات وتجعله يقرر متى يستخدمها
llm_with_tools = llm.bind_tools(tools)


# ═══════════════════════════════════════════════════════════════════════════
# 5. AgentState  —  حالة الـ Agent (الذاكرة قصيرة المدى)
# ═══════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    """
    TypedDict يحدد شكل الـ state الذي يتشاركه كل nodes في الـ graph.

    messages:
      - Annotated[Sequence[BaseMessage], operator.add]
      - operator.add يعني: عند كل خطوة تُضاف الرسائل الجديدة فوق القديمة
        (لا تُستبدل) — هكذا يحتفظ الـ Agent بكامل تاريخ المحادثة.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ═══════════════════════════════════════════════════════════════════════════
# 6. LangGraph Nodes
# ═══════════════════════════════════════════════════════════════════════════

def agent_node(state: AgentState) -> dict:
    """
    Node الرئيسي — يستدعي الـ LLM مع كامل تاريخ الرسائل.

    المدخل:  state["messages"] — كل الرسائل السابقة
    المخرج:  {"messages": [رد الـ LLM]}

    الـ LLM يقرر:
      • إذا يريد استخدام tool → يُرجع AIMessage مع tool_calls
      • إذا لديه إجابة نهائية → يُرجع AIMessage مع content فقط
    """
    if not is_api_key_valid:
        return {"messages": [AIMessage(content="⚠️ لا يمكن تشغيل المساعد الذكي بدون مفتاح API صالح. يرجى تعديل ملف `.env` ووضع مفتاح API الخاص بـ Gemini في حقل `GOOGLE_API_KEY` ثم إعادة تشغيل التطبيق.")]}
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Conditional Edge — يقرر الخطوة التالية بعد agent_node.

    المنطق:
      • إذا آخر رسالة فيها tool_calls → اذهب لـ "tools" node لتنفيذها
      • إذا لا → اذهب لـ END (المحادثة انتهت)

    هذا هو جوهر الـ Agentic Loop: Agent → Tools → Agent → ... → END
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# ═══════════════════════════════════════════════════════════════════════════
# 7. LangGraph Workflow  —  بناء الـ Graph
# ═══════════════════════════════════════════════════════════════════════════

def create_workflow() -> StateGraph:
    """
    يبني StateGraph الخاص بالـ Agent:

    الهيكل:
      START
        ↓
      [agent_node]  ← ─────────────────┐
        ↓                               │
      should_continue()                 │
        ├─ "tools" → [ToolNode] ────────┘  (loop)
        └─ END
    """
    graph = StateGraph(AgentState)

    # إضافة الـ Nodes
    graph.add_node("agent", agent_node)
    # ToolNode من LangGraph prebuilt — ينفّذ الـ tool المطلوب تلقائياً
    graph.add_node("tools", ToolNode(tools))

    # الـ Edges الثابتة
    graph.add_edge(START, "agent")          # البداية دائماً من agent
    graph.add_edge("tools", "agent")        # بعد الأداة ارجع للـ agent

    # الـ Edge الشرطية — تحدد: tools أو END
    graph.add_conditional_edges(
        "agent",
        should_continue,
        ["tools", END],
    )

    return graph


# ═══════════════════════════════════════════════════════════════════════════
# عقد تجريبي للاختبار
# ═══════════════════════════════════════════════════════════════════════════
DEMO_CONTRACT = """عقد عمل بين شركة التقنيات المتقدمة والسيد أحمد محمد:
المادة الأولى - المدة: يناير 2025 إلى ديسمبر 2025.
المادة الثانية - الراتب: 15,000 جنيه شهرياً.
المادة الثالثة - عدم المنافسة: 5 سنوات في أي مكان في العالم.
المادة الرابعة - الإنهاء: في أي وقت دون إشعار أو تعويض.
المادة الخامسة - السرية: مدى الحياة.
المادة السادسة - التحكيم: لندن وفق قانون إنجلترا.
المادة السابعة - الغرامات: 500,000 جنيه عند أي مخالفة."""


# ═══════════════════════════════════════════════════════════════════════════
# تشغيل من الـ Terminal مباشرة
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    workflow = create_workflow()

    # MemorySaver: يخزّن الـ checkpoints في RAM فقط — لا قاعدة بيانات
    # كل thread_id = محادثة مستقلة بذاكرتها الخاصة
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # thread_id يُحدد جلسة المحادثة — نفس الـ ID = نفس الذاكرة
    config = {"configurable": {"thread_id": "terminal_session_1"}}

    print("\n" + "=" * 60)
    print("العقد التجريبي:")
    print(DEMO_CONTRACT)
    print("=" * 60 + "\n")

    user_input = (
        "قم بتحليل العقد التالي خطوة بخطوة: "
        "أولاً استخرج هيكله، ثم اكتشف البنود، ثم حلّل المخاطر، "
        "وأخيراً قارن النتائج بالقانون المصري:\n\n" + DEMO_CONTRACT
    )

    print("🚀 بدء تنفيذ الـ Agentic Workflow...\n")

    for event in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values",
    ):
        msg = event["messages"][-1]
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                names = [tc["name"] for tc in msg.tool_calls]
                print(f"🛠️  الـ Agent يستخدم الأدوات: {names}")
            elif msg.content:
                print("🤖 الرد النهائي:")
                print(msg.content)
                print("-" * 40)

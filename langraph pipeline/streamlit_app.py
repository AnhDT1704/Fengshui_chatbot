"""
streamlit_app.py – Simple chat UI for testing the Vạn An chatbot.

Usage:
    cd "langraph pipeline"
    streamlit run streamlit_app.py

Features:
  - Text chat
  - Image upload (sent as base64 to chat_with_image)
  - Markdown render (so product image URLs render inline)
  - Sidebar: session_id picker, agent + tools used per turn
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import base64
import uuid
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

import graph as chat_graph

load_dotenv()


# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Vạn An Group – Fengshui Chatbot",
    page_icon="🪷",
    layout="wide",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = f"sl-{uuid.uuid4().hex[:8]}"
if "history" not in st.session_state:
    st.session_state.history = []   # [{"role": ..., "content": ..., "meta": ...}]


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🪷 Vạn An Chatbot")
    st.caption("Multi-agent · LangGraph · Gemini 2.5")

    st.text_input(
        "Session ID",
        key="session_id",
        help="Lịch sử hội thoại được lưu theo session_id này.",
    )

    if st.button("🔄 New session", use_container_width=True):
        st.session_state.session_id = f"sl-{uuid.uuid4().hex[:8]}"
        st.session_state.history = []
        st.rerun()

    if st.button("🗑️ Clear UI history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("📷 Đính kèm ảnh")
    MAX_IMAGES = 5
    uploaded_files = st.file_uploader(
        f"Upload tối đa {MAX_IMAGES} ảnh để gửi kèm câu tiếp theo",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    uploaded_files = uploaded_files or []
    if len(uploaded_files) > MAX_IMAGES:
        st.warning(f"Chỉ gửi {MAX_IMAGES} ảnh đầu (bạn chọn {len(uploaded_files)}).")
        uploaded_files = uploaded_files[:MAX_IMAGES]
    if uploaded_files:
        st.image(uploaded_files, caption=[f"Ảnh {i+1}" for i in range(len(uploaded_files))], width=120)


# ═══════════════════════════════════════════════════════════════════
#  MAIN CHAT
# ═══════════════════════════════════════════════════════════════════

st.title("💬 Tư vấn sản phẩm phong thủy")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            badge = f"🤖 `{meta.get('agent_used','')}`"
            tools = meta.get("tools_called") or []
            if tools:
                badge += "  ·  🛠️ " + ", ".join(f"`{t}`" for t in tools)
            st.caption(badge)


user_input = st.chat_input("Nhập câu hỏi cho shop...")

if user_input:
    # Echo user message
    with st.chat_message("user"):
        st.markdown(user_input)
        if uploaded_files:
            st.image(uploaded_files, width=120)
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            try:
                if uploaded_files:
                    images = [
                        {
                            "base64": base64.b64encode(f.getvalue()).decode("ascii"),
                            "mime":   f.type or "image/jpeg",
                        }
                        for f in uploaded_files
                    ]
                    out = chat_graph.chat_with_image(
                        user_message=user_input,
                        images=images,
                        session_id=st.session_state.session_id,
                    )
                else:
                    out = chat_graph.chat(
                        user_input,
                        session_id=st.session_state.session_id,
                    )
            except Exception as e:
                err = f"❌ Lỗi: `{type(e).__name__}: {e}`"
                st.error(err)
                st.session_state.history.append({
                    "role": "assistant",
                    "content": err,
                    "meta": {"agent_used": "error", "tools_called": []},
                })
                st.stop()

        response_md = out["response"]
        st.markdown(response_md)
        agent = out["agent_used"]
        tools = out["tools_called"]
        badge = f"🤖 `{agent}`"
        if tools:
            badge += "  ·  🛠️ " + ", ".join(f"`{t}`" for t in tools)
        st.caption(badge)

        st.session_state.history.append({
            "role":    "assistant",
            "content": response_md,
            "meta":    {"agent_used": agent, "tools_called": tools},
        })

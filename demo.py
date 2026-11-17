import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

from urbancart.config import ROOT

load_dotenv(ROOT / ".env")
API = os.getenv("URBANCART_API_URL", "http://127.0.0.1:5000").rstrip("/")
st.set_page_config(page_title="Ask UrbanCart", page_icon="🛒")
st.title("Ask UrbanCart")
st.caption("Product and policy answers from the supplied catalog. Orders and support tickets are simulations.")

if "customer_chats" not in st.session_state:
    st.session_state.customer_chats = {}
    previous = st.session_state.get("active_customer")
    if previous:
        st.session_state.customer_chats[previous] = {
            "messages": st.session_state.get("messages", []),
            "conversation": st.session_state.get("conversation"),
        }

with st.sidebar:
    st.subheader("Demo settings")
    customer = st.selectbox("Mock customer", ["DEMO-01", "DEMO-02"])
    st.session_state.active_customer = customer
    chat = st.session_state.customer_chats.setdefault(customer, {"messages": [], "conversation": None})
    st.caption("Switch customers to resume their chat during this session.")
    locale = st.selectbox("Query language", ["en-IN", "hi-IN", "hinglish"])
    if st.button("Clear this chat", disabled=not chat["messages"],
                 help="Clear this customer's messages and conversation context. The other customer's chat is kept."):
        st.session_state.customer_chats[customer] = {"messages": [], "conversation": None}
        st.rerun()
    st.caption("Clears only the selected customer's chat and context.")
    try:
        health = requests.get(f"{API}/healthz", timeout=2).json()
        st.caption("OpenRouter mode" if health.get("mode") == "openrouter" else "Local retrieval mode")
    except (requests.RequestException, ValueError):
        st.warning("Start the Flask API to send messages.")


def render(message):
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("mode"):
            st.caption(f"{message['mode']} · LLM calls: {message.get('llm_calls', 0)} · "
                       f"Embedding calls: {message.get('embedding_calls', 0)} · "
                       f"{message.get('elapsed_ms', 0)/1000:.2f}s")
        if message.get("confidence") is not None:
            st.caption(f"Model-reported confidence: {message['confidence']:.0%} (uncalibrated)")
        if message.get("mock"):
            st.caption("Catalog snapshot or simulated tool result")
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['id']}** · {source['title']}")
                    st.write(source["text"])


for message in chat["messages"]:
    render(message)

if not chat["messages"]:
    st.markdown(
        "Try **“Can I return electronics?”**, **“COD ka refund kaise milega?”**, "
        "or **“Where is order UC882140?”** with demo customer DEMO-01."
    )

if prompt := st.chat_input("Ask about a product, policy, or demo order", max_chars=1000):
    entry = {"role": "user", "text": prompt}
    chat["messages"].append(entry)
    render(entry)
    with st.spinner("Checking the available evidence…"):
        try:
            response = requests.post(f"{API}/ask", json={
                "query": prompt, "locale": locale, "customer_id": customer,
                "session_id": chat["conversation"], "request_id": uuid.uuid4().hex,
            }, timeout=(3, 150))
            reply = response.json()
            chat["conversation"] = reply.get("session_id", chat["conversation"])
            message = {"role": "assistant", "text": reply.get("answer", "The API could not answer this request."),
                       "sources": reply.get("sources", []), "mock": reply.get("mock", False),
                       **{key: reply.get(key) for key in ("mode", "llm_calls", "embedding_calls", "elapsed_ms", "confidence")}}
        except (requests.RequestException, ValueError):
            message = {"role": "assistant", "text": "The API did not respond. Check the Flask terminal and try again."}
    chat["messages"].append(message)
    # Refresh controls rendered before this new turn, including the disabled clear button.
    st.rerun()

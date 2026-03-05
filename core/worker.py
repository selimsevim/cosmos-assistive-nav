import threading
from typing import Any, Dict
from datetime import datetime

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from core.frame_sampler import sample_frames
from core.cosmos_client import analyze_frames


def worker_thread(frame_a, frame_b, call_meta=None):
    """
    Executes frame processing and API call in the background.
    """
    st.session_state.api_is_busy = True

    try:
        frame_a_b64, frame_b_b64 = sample_frames(frame_a, frame_b)

        current_state: Dict[str, Any] = {
            "guidance": st.session_state.get("latest_guidance", "Waiting for analysis..."),
        }
        result = analyze_frames(frame_a_b64, frame_b_b64, current_state, call_meta=call_meta)

        guidance = str(result.get("guidance", "SAFE - path status unavailable; continue carefully.")).strip()
        if not guidance:
            guidance = "SAFE - path status unavailable; continue carefully."

        st.session_state.latest_guidance = guidance

        if "reasoning_history" not in st.session_state:
            st.session_state.reasoning_history = []
        if "reasoning_seq" not in st.session_state:
            st.session_state.reasoning_seq = 0

        st.session_state.reasoning_seq += 1
        history_item = {
            "id": int(st.session_state.reasoning_seq),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "call_index": (call_meta or {}).get("call_index"),
            "frame_prev_idx": (call_meta or {}).get("frame_prev_idx"),
            "frame_curr_idx": (call_meta or {}).get("frame_curr_idx"),
            "guidance": guidance,
        }
        st.session_state.reasoning_history.append(history_item)

    except Exception as exc:
        st.session_state.latest_guidance = f"SAFE - Worker error: {str(exc)}"

    finally:
        st.session_state.api_is_busy = False


def trigger_ai_worker(frame_a, frame_b, call_meta=None):
    """
    Starts the worker in a background thread if it is not already running.
    """
    if not st.session_state.get("api_is_busy", False):
        # Lock immediately to avoid race before thread starts.
        st.session_state.api_is_busy = True
        try:
            thread = threading.Thread(
                target=worker_thread,
                args=(frame_a.copy(), frame_b.copy(), call_meta),
            )
            add_script_run_ctx(thread)
            thread.start()
        except Exception:
            st.session_state.api_is_busy = False
            raise

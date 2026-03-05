import collections
import os
import sys
import time
from datetime import datetime

import cv2
import streamlit as st

# Ensure root directory is accessible for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.worker import trigger_ai_worker


def main():
    st.set_page_config(page_title="PathSense: Vision Reasoning for Assistive Navigation", layout="wide")

    if "latest_guidance" not in st.session_state:
        st.session_state.latest_guidance = "Waiting for analysis..."
    if "api_is_busy" not in st.session_state:
        st.session_state.api_is_busy = False
    if "active_demo" not in st.session_state:
        st.session_state.active_demo = None
    if "trace_root" not in st.session_state:
        st.session_state.trace_root = ""
    if "reasoning_history" not in st.session_state:
        st.session_state.reasoning_history = []
    if "reasoning_seq" not in st.session_state:
        st.session_state.reasoning_seq = 0

    st.title("PathSense: Vision Reasoning for Assistive Navigation")

    demos = {
        "Video1": {"file": "Video1.mp4", "title": "Video 1 - Approaching Cyclist on Shared Path"},
        "Video2": {"file": "Video2.mp4", "title": "Video 2 - Path Split and Downward Slope"},
        "Video3": {"file": "Video3.mp4", "title": "Video 3 - Walking Close to Curb Edge"},
    }

    for demo_name, demo_cfg in demos.items():
        video_file = demo_cfg["file"]
        demo_title = demo_cfg["title"]
        st.header(demo_title)
        is_active = st.session_state.active_demo == demo_name

        if not is_active:
            if st.button(f"Run {demo_name} Analysis", key=f"start_{demo_name}"):
                st.session_state.active_demo = demo_name
                st.session_state.latest_guidance = "Waiting for analysis..."
                st.session_state.api_is_busy = False
                st.session_state.reasoning_history = []
                st.session_state.reasoning_seq = 0

                trace_root = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "reports",
                    f"project_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                os.makedirs(trace_root, exist_ok=True)
                st.session_state.trace_root = trace_root
                st.rerun()
        else:
            if st.button(f"Stop {demo_name} Analysis", key=f"stop_{demo_name}"):
                st.session_state.active_demo = None
                st.rerun()

            col_video, col_info = st.columns([1.5, 1])

            with col_video:
                video_placeholder = st.empty()

            with col_info:
                st.subheader("Live Guidance")
                guidance_placeholder = st.empty()
                st.markdown("<br>", unsafe_allow_html=True)
                history_placeholder = st.empty()
                if st.session_state.trace_root:
                    st.caption(f"Trace: {st.session_state.trace_root}")

            video_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "videos",
                video_file,
            )
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0

            window_frames = max(1, int(round(fps * 3.0)))
            step_frames = max(1, int(round(fps * 1.0)))

            frame_buffer = collections.deque(maxlen=window_frames + 2)
            stream_started = False
            frame_idx = -1
            call_index = 0
            last_call_frame_idx = -step_frames
            video_ended = False

            def try_trigger_call() -> bool:
                nonlocal call_index, last_call_frame_idx
                if st.session_state.api_is_busy:
                    return False
                if frame_idx < window_frames:
                    return False
                if (frame_idx - last_call_frame_idx) < step_frames:
                    return False

                prev_idx = frame_idx - window_frames
                history_map = {idx: buffered_frame for idx, buffered_frame in frame_buffer}
                if prev_idx not in history_map or frame_idx not in history_map:
                    return False

                prev_frame = history_map[prev_idx]
                curr_frame = history_map[frame_idx]
                call_index += 1
                call_meta = {
                    "trace_root": st.session_state.trace_root,
                    "demo_name": demo_name,
                    "video_file": video_file,
                    "call_index": call_index,
                    "frame_prev_idx": prev_idx,
                    "frame_curr_idx": frame_idx,
                    "time_prev_sec": round(prev_idx / fps, 3),
                    "time_curr_sec": round(frame_idx / fps, 3),
                    "window_seconds": 3.0,
                    "step_seconds": 1.0,
                }
                trigger_ai_worker(prev_frame, curr_frame, call_meta=call_meta)
                last_call_frame_idx = frame_idx
                return True

            while cap.isOpened() and st.session_state.active_demo == demo_name:
                ret, frame = cap.read()
                if not ret:
                    video_ended = True
                    st.session_state.active_demo = None
                    st.session_state.api_is_busy = False
                    break

                frame_idx += 1
                frame_buffer.append((frame_idx, frame.copy()))

                if not stream_started:
                    guidance_placeholder.markdown(
                        "<b>Guidance:</b> Buffering first 3-second window...",
                        unsafe_allow_html=True,
                    )
                    if try_trigger_call():
                        stream_started = True
                    else:
                        time.sleep(1 / 30.0)
                        continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

                guidance_placeholder.markdown(
                    f"<b>Guidance:</b> {st.session_state.latest_guidance}",
                    unsafe_allow_html=True,
                )

                with history_placeholder.container():
                    st.markdown("**Guidance History**")
                    history = st.session_state.get("reasoning_history", [])
                    if not history:
                        st.caption("No model responses yet.")
                    else:
                        for item in reversed(history):
                            call_label = item.get("call_index")
                            frame_prev = item.get("frame_prev_idx")
                            frame_curr = item.get("frame_curr_idx")
                            title = (
                                f"#{item.get('id')} | {item.get('timestamp')} | "
                                f"Call {call_label}"
                            )
                            with st.expander(title, expanded=False):
                                st.markdown(f"**Guidance:** {item.get('guidance', '')}")
                                st.caption(f"Frames: {frame_prev} -> {frame_curr}")

                try_trigger_call()
                time.sleep(1 / 30.0)

            cap.release()
            if video_ended:
                st.info(f"{demo_name} finished. Click Run {demo_name} Analysis to play again.")

        st.markdown("---")


if __name__ == "__main__":
    main()

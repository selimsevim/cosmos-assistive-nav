import base64
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict

import requests


SYSTEM_PROMPT = """You are a safety guidance engine for a blind pedestrian.

You receive two images:
Image 1 = earlier
Image 2 = current

Decide the safest immediate guidance.

Use these actions only:
SAFE, SLOW, STOP

Return exactly one line in this format:
ACTION - short guidance sentence

Examples:
SLOW - curb and grassy edge are close on the right side; keep left.
STOP - obstacle directly ahead blocks the path.
SAFE - path is clear; continue forward.

Rules:
- If a curb, edge, drop-off, or obstacle is close to the path: prefer SLOW.
- If immediate collision risk exists: use STOP.
- Keep the sentence short and concrete.
- No JSON. No markdown. One line only.
""".strip()


def _extract_guidance(content: Any) -> str:
    if isinstance(content, dict):
        for key in ("guidance", "reason", "output", "message"):
            if key in content and str(content[key]).strip():
                return str(content[key]).strip()

    text = str(content).strip()
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()

    # If response is JSON, try to recover common fields.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("guidance", "reason", "output", "message"):
                if key in parsed and str(parsed[key]).strip():
                    return str(parsed[key]).strip()
    except Exception:
        pass

    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _normalize_guidance(line: str) -> str:
    value = (line or "").strip()
    if not value:
        return "SAFE - path status unavailable; continue carefully."

    value = re.sub(r"\s+", " ", value)
    upper = value.upper()
    if upper.startswith("SAFE") or upper.startswith("SLOW") or upper.startswith("STOP"):
        return value

    lower = value.lower()
    if any(token in lower for token in ["stop", "collision", "blocked", "immediate"]):
        return f"STOP - {value}"
    if any(token in lower for token in ["curb", "edge", "drop", "obstacle", "caution", "risk", "close"]):
        return f"SLOW - {value}"
    return f"SAFE - {value}"


def _persist_call_trace(
    call_meta: Dict[str, Any],
    frame_a_b64: str,
    frame_b_b64: str,
    payload: Dict[str, Any],
    raw_content: str,
    response_json: Dict[str, Any],
    parsed: Dict[str, Any],
    url: str,
    model: str,
    timeout_seconds: int,
    error_info: str = "",
) -> None:
    if not call_meta:
        return

    trace_root = str(call_meta.get("trace_root", "")).strip()
    if not trace_root:
        return

    demo_name = str(call_meta.get("demo_name", "video")).strip() or "video"
    demo_dir = os.path.join(trace_root, demo_name.replace(" ", "_"))

    call_index = int(call_meta.get("call_index", 0) or 0)
    frame_prev_idx = call_meta.get("frame_prev_idx")
    frame_curr_idx = call_meta.get("frame_curr_idx")

    if frame_prev_idx is not None and frame_curr_idx is not None:
        call_name = f"call_{call_index:05d}_f{frame_prev_idx}_to_f{frame_curr_idx}"
    else:
        call_name = f"call_{call_index:05d}"

    call_dir = os.path.join(demo_dir, call_name)
    os.makedirs(call_dir, exist_ok=True)

    if frame_a_b64:
        with open(os.path.join(call_dir, "frame_01_sent.jpg"), "wb") as handle:
            handle.write(base64.b64decode(frame_a_b64))
    if frame_b_b64:
        with open(os.path.join(call_dir, "frame_02_sent.jpg"), "wb") as handle:
            handle.write(base64.b64decode(frame_b_b64))

    request_meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "url": url,
        "model": model,
        "timeout_seconds": timeout_seconds,
        "headers_sent": {
            "Content-Type": "application/json",
        },
        "frame_a_base64_kb": round(len(frame_a_b64) / 1024, 3),
        "frame_b_base64_kb": round(len(frame_b_b64) / 1024, 3),
        "call_meta": call_meta,
    }

    with open(os.path.join(call_dir, "request_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(request_meta, handle, indent=2, ensure_ascii=False)

    with open(os.path.join(call_dir, "request_payload_full.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with open(os.path.join(call_dir, "response_raw.txt"), "w", encoding="utf-8") as handle:
        handle.write(raw_content)

    with open(os.path.join(call_dir, "response_parsed.json"), "w", encoding="utf-8") as handle:
        json.dump(parsed, handle, indent=2, ensure_ascii=False)

    with open(os.path.join(call_dir, "response_full.json"), "w", encoding="utf-8") as handle:
        json.dump(response_json, handle, indent=2, ensure_ascii=False)

    if error_info:
        with open(os.path.join(call_dir, "error.txt"), "w", encoding="utf-8") as handle:
            handle.write(error_info)

    print(f"[PathSense] Trace saved: {call_dir}")


def analyze_frames(
    frame_a_b64: str,
    frame_b_b64: str,
    current_state: Dict[str, Any],
    call_meta: Dict[str, Any] = None,
) -> Dict[str, Any]:
    _ = current_state
    url = os.environ.get("COSMOS_ENDPOINT", "http://<IP>/v1/chat/completions")
    model = os.environ.get("COSMOS_MODEL", "nvidia/Cosmos-Reason2-8B")
    timeout_seconds = 15

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_a_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b_b64}"}},
                ],
            },
        ],
        "max_tokens": 120,
    }

    print(
        f"\n[PathSense] Sending Request "
        f"(Images: {len(frame_a_b64)/1024:.1f}KB, {len(frame_b_b64)/1024:.1f}KB)"
    )
    start_time = time.time()

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        response.raise_for_status()

        latency = time.time() - start_time
        print(f"[PathSense] Inference Completed in {latency:.2f}s")

        response_json = response.json()
        result_text = str(response_json["choices"][0]["message"]["content"])
        print(f"[PathSense] Response: {result_text}")

        guidance = _normalize_guidance(_extract_guidance(result_text))
        parsed = {"guidance": guidance}

        _persist_call_trace(
            call_meta=call_meta or {},
            frame_a_b64=frame_a_b64,
            frame_b_b64=frame_b_b64,
            payload=payload,
            raw_content=result_text,
            response_json=response_json,
            parsed=parsed,
            url=url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        return parsed

    except Exception as exc:
        latency = time.time() - start_time
        print(f"[PathSense] Failed after {latency:.2f}s: {str(exc)}")
        fallback = {"guidance": f"SAFE - API error: {str(exc)}"}

        _persist_call_trace(
            call_meta=call_meta or {},
            frame_a_b64=frame_a_b64,
            frame_b_b64=frame_b_b64,
            payload=payload,
            raw_content=str(exc),
            response_json={"error": str(exc)},
            parsed=fallback,
            url=url,
            model=model,
            timeout_seconds=timeout_seconds,
            error_info=str(exc),
        )
        return fallback

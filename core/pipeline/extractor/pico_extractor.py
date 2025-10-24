from typing import Tuple, Dict, Any, Optional
import os
import re
import orjson
from json_repair import repair_json
from pydantic import ValidationError

# Use your lowercase path as you showed
from agents.factory import build_vertex_agent
from schemas.pico import PICO, QuoteBacks


def _agent():
    PROJECT_ID  = os.getenv("PROJECT_ID",  "evidence-synthesis-gemma")
    LOCATION    = os.getenv("LOCATION",    "us-east1")
    ENDPOINT_ID = os.getenv("ENDPOINT_ID", "586173838023196672")
    DEDICATED   = os.getenv(
        "DEDICATED_DNS",
        "586173838023196672.us-east1-409780034045.prediction.vertexai.goog",
    )
    return build_vertex_agent(
        project_id=PROJECT_ID,
        location=LOCATION,
        endpoint_id=ENDPOINT_ID,
        dedicated_dns_or_predict_url=DEDICATED,
        temperature=0.1,
        max_tokens=800,
    )


# ---------- parsing helpers ----------

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)

def _strip_code_fences(s: str) -> str:
    m = _CODE_FENCE_RE.match(s.strip())
    return m.group(1) if m else s

def _loads_or_repair(text: str) -> Dict:
    """
    Parse JSON str; if it fails, repair and parse again.
    """
    s = text.strip()
    try:
        return orjson.loads(s)
    except Exception:
        fixed = repair_json(s)
        return orjson.loads(fixed)

def _extract_vertex_content(obj: Any) -> Optional[str]:
    """
    Accepts either:
      - stringified JSON of Vertex response (your case),
      - dict Vertex response,
      - or already-plain text.
    Returns the assistant text (could be fenced).
    """
    # If it's already plain text, return it
    if isinstance(obj, str):
        # If it looks like a JSON object for Vertex, try to decode then drill in
        trimmed = obj.strip()
        if trimmed.startswith("{") and ("\"predictions\"" in trimmed or "'predictions'" in trimmed):
            try:
                obj = _loads_or_repair(trimmed)
            except Exception:
                # It's plain text, just return it
                return obj
        else:
            return obj

    if isinstance(obj, dict):
        # Your response shape
        preds = obj.get("predictions")
        if isinstance(preds, dict):
            choices = preds.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content

        # Alternate Vertex shape (candidates / parts)
        cands = obj.get("candidates") or []
        if cands and isinstance(cands[0], dict):
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
            merged = "\n".join([t for t in texts if t.strip()])
            if merged.strip():
                return merged

    # Fallback: nothing found
    return None


def _parse_pico_from_text(text: str) -> Dict:
    """
    Strip code fences, then parse/repair into a dict.
    """
    inner = _strip_code_fences(text).strip()
    return _loads_or_repair(inner)


# ---------- main entry ----------

def extract_pico(question: str, prompt_template: str) -> Tuple[PICO, QuoteBacks, Dict]:
    """
    Single LLM call -> unwrap Vertex -> strip fences -> parse/repair -> validate.
    Returns (pico_valid, quotes, parsed_pico_dict).
    """
    agent = _agent()
    agent.set_system("You are concise and helpful.")

    prompt = prompt_template.replace("{{QUESTION}}", question)
    resp = agent.say(prompt)  # may be a raw dict OR a JSON string of the dict

    # 1) Get the assistant text out of the Vertex envelope
    content = _extract_vertex_content(resp)
    if not isinstance(content, str) or not content.strip():
        # Last resort: treat the whole thing as text and try to parse
        content = str(resp)

    # 2) Now parse the actual PICO JSON that was inside the content
    parsed = _parse_pico_from_text(content)

    # 3) Validate/coerce to clean PICO model
    try:
        pico = PICO.model_validate({
            "Population":   parsed.get("Population"),
            "Intervention": parsed.get("Intervention"),
            "Comparator":   parsed.get("Comparator"),
            "Outcomes":     parsed.get("Outcomes", []),
        })
    except ValidationError:
        pop = str(parsed.get("Population", "")).strip() or "unknown"
        itv = str(parsed.get("Intervention", "")).strip() or "unknown"
        comp_raw = parsed.get("Comparator")
        comp = None if comp_raw in [None, ""] else (str(comp_raw).strip() or None)
        outs = parsed.get("Outcomes") or []
        outs = [o for o in outs if isinstance(o, str) and o.strip()]
        pico = PICO(Population=pop, Intervention=itv, Comparator=comp, Outcomes=outs)

    quotes = QuoteBacks.model_validate(parsed.get("_quote_backs", {}))

    # Return the parsed inner JSON (not the outer Vertex payload)
    return pico, quotes, parsed

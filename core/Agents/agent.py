from __future__ import annotations
from google.cloud import aiplatform
from typing import List, Dict, Optional, Sequence, Any
import json

def _build_resource(project_id: str, location: str, endpoint_id: str) -> str:
    if endpoint_id.startswith(f"projects/{project_id}/locations/{location}/endpoints/"):
        return endpoint_id
    return f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"

def _format_chat(system: Optional[str], history: List[Dict[str,str]], user_text: Optional[str]) -> str:
    lines = []
    if system:
        lines += [f"[SYSTEM]\n{system}\n"]
    for m in history:
        role = m.get("role","user").lower()
        content = m.get("content","")
        tag = "ASSISTANT" if role == "assistant" else "USER"
        lines += [f"[{tag}]\n{content}\n"]
    if user_text is not None:
        lines += [f"[USER]\n{user_text}\n", "[ASSISTANT]\n"]
    return "\n".join(lines).strip()

def _extract_text(pred0: Any) -> str:
    if isinstance(pred0, str):
        return pred0
    if isinstance(pred0, dict):
        for k in ("content","text","generated_text","output","response"):
            v = pred0.get(k)
            if isinstance(v, str):
                return v
        for k in ("candidates","outputs","messages","choices"):
            seq = pred0.get(k)
            if isinstance(seq, list) and seq:
                item = seq[0]
                if isinstance(item, str):
                    return item
                if isinstance(item, dict):
                    for kk in ("content","text","generated_text"):
                        vv = item.get(kk)
                        if isinstance(vv, str):
                            return vv
                    msg = item.get("message")
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        if isinstance(c, str):
                            return c
    return json.dumps(pred0, ensure_ascii=False)

class Agent:
    def __init__(
        self,
        project_id: str,
        location: str,
        endpoint_id: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ):
        aiplatform.init(project=project_id, location=location)
        self.endpoint = aiplatform.Endpoint(
            endpoint_name=_build_resource(project_id, location, endpoint_id)
        )
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.top_p = float(top_p) if top_p is not None else None
        self.top_k = int(top_k) if top_k is not None else None
        self.stop = list(stop) if stop else None
        self.system: Optional[str] = None
        self.history: List[Dict[str,str]] = []

    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        params = {
            "max_tokens": int(self.max_new_tokens if max_new_tokens is None else max_new_tokens),
            "temperature": float(self.temperature if temperature is None else temperature),
        }
        if top_p is not None or self.top_p is not None:
            params["top_p"] = float(self.top_p if top_p is None else top_p)
        if top_k is not None or self.top_k is not None:
            params["top_k"] = int(self.top_k if top_k is None else top_k)
        s = stop if stop is not None else self.stop
        if s:
            params["stop"] = list(s)

        resp = self.endpoint.predict(instances=[{"prompt": prompt}], parameters=params)
        preds = getattr(resp, "predictions", None)
        if isinstance(preds, list) and preds:
            return _extract_text(preds[0]).strip()
        return str(resp)

    def say(self, user_text: str, **gen) -> str:
        prompt = _format_chat(self.system, self.history, user_text)
        reply = self.complete(prompt, **gen)
        self.history.append({"role":"user","content":user_text})
        self.history.append({"role":"assistant","content":reply})
        return reply

    def set_system(self, text: str) -> None:
        self.system = text

    def reset(self) -> None:
        self.history.clear()
        self.system = None

def agent_factory(
    project_id: str,
    location: str,
    model_name: str,  # kept for signature compatibility; unused here
    *,
    endpoint_id: str,
    **agent_kwargs,
) -> Agent:
    if not endpoint_id:
        raise ValueError("endpoint_id is required since we are not using a registry.")
    return Agent(project_id, location, endpoint_id, **agent_kwargs)

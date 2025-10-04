from google.cloud import aiplatform
import json, re
from typing import List, Dict, Optional

# optional: map model names → endpoint IDs you deployed
MODEL_REGISTRY: Dict[str, str] = {
    # "gemma-3-4b-it": "mg-endpoint-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
}

def _build_resource(project_id: str, location: str, endpoint_id: str) -> str:
    if endpoint_id.startswith(f"projects/{project_id}/locations/{location}/endpoints/"):
        return endpoint_id
    return f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"

def _format_chat(system: Optional[str], history: List[Dict[str,str]], user_text: Optional[str]) -> str:
    lines = []
    if system: lines += [f"[SYSTEM]\n{system}\n"]
    for m in history:
        role = m.get("role","user").lower()
        content = m.get("content","")
        tag = "ASSISTANT" if role == "assistant" else "USER"
        lines += [f"[{tag}]\n{content}\n"]
    if user_text is not None:
        lines += [f"[USER]\n{user_text}\n", "[ASSISTANT]\n"]
    return "\n".join(lines).strip()

class Agent:
    def __init__(self, project_id: str, location: str, endpoint_id: str, *, temperature: float = 0.0, max_new_tokens: int = 512):
        aiplatform.init(project=project_id, location=location)
        self.endpoint = aiplatform.Endpoint(endpoint_name=_build_resource(project_id, location, endpoint_id))
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system: Optional[str] = None
        self.history: List[Dict[str,str]] = []

    # single-shot completion (ignores history)
    def complete(self, prompt: str, *, temperature: Optional[float] = None, max_new_tokens: Optional[int] = None) -> str:
        """
        Vertex AI Model Garden (Gemma-3 Publisher Model, vLLM-backed) predict call.

        Contract for this endpoint:
        - instances: [{"prompt": "<text>"}]
        - parameters: use vLLM-style names; most importantly: "max_tokens"
            (NOT maxOutputTokens / max_new_tokens).
        """
        mnt = int(self.max_new_tokens if max_new_tokens is None else max_new_tokens)
        temp = float(self.temperature if temperature is None else temperature)

        params = {
            "temperature": temp,
            "max_tokens": mnt,     # <-- vLLM expects this exact key
            # Optional vLLM knobs you may add later:
            # "top_p": 0.95,
            # "top_k": 40,
            # "stop": [],          # list of strings
        }

        def _extract_text(pred0):
            # Normalize common shapes returned by MG publishers / vLLM bridges.
            if isinstance(pred0, str):
                return pred0
            if isinstance(pred0, dict):
                # common fields
                for k in ("content", "text", "generated_text", "output", "response"):
                    v = pred0.get(k)
                    if isinstance(v, str):
                        return v
                # candidates/outputs lists
                for k in ("candidates", "outputs", "messages"):
                    seq = pred0.get(k)
                    if isinstance(seq, list) and seq:
                        item = seq[0]
                        if isinstance(item, str):
                            return item
                        if isinstance(item, dict):
                            for kk in ("content", "text", "generated_text"):
                                vv = item.get(kk)
                                if isinstance(vv, str):
                                    return vv
            return json.dumps(pred0, ensure_ascii=False)

        # Primary path: 'prompt' + vLLM-style parameters
        resp = self.endpoint.predict(
            instances=[{"prompt": prompt}],
            parameters=params,
        )
        preds = getattr(resp, "predictions", None)
        if isinstance(preds, list) and preds:
            return _extract_text(preds[0]).strip()
        return str(resp)





    # chat turn: uses history + system; appends assistant reply
    def say(self, user_text: str, **gen) -> str:
        prompt = _format_chat(self.system, self.history, user_text)
        reply = self.complete(prompt, **gen)
        self.history.append({"role":"user","content":user_text})
        self.history.append({"role":"assistant","content":reply})
        return reply
    # convenience
    def set_system(self, text: str): self.system = text
    def reset(self): self.history.clear(); self.system = None

def agent_factory(project_id: str, location: str, model_name: str, *, endpoint_id: Optional[str]=None, registry: Optional[Dict[str,str]]=None, **agent_kwargs) -> Agent:
    reg = registry or MODEL_REGISTRY
    eid = endpoint_id or reg.get(model_name)
    if not eid:
        raise ValueError(f"No endpoint_id for model '{model_name}'. Pass endpoint_id=... or add to MODEL_REGISTRY.")
    return Agent(project_id, location, eid, **agent_kwargs)

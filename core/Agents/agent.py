# agent.py  (predict + chatCompletions)
from __future__ import annotations

from typing import List, Dict, Optional, Sequence, Any
import json
import requests
import google.auth
import google.auth.transport.requests as gar
from urllib.parse import urlparse
import socket
import re

_END_MARKERS = [r"<\s*end_of_turn\s*>", r"<\|end_of_turn\|>", r"<\|eot_id\|>"]
_END_RX = re.compile("|".join(_END_MARKERS), flags=re.IGNORECASE)
_END_TAIL_RX = re.compile(r"(?:\s*(?:%s)\s*)+$" % "|".join(_END_MARKERS), re.IGNORECASE)

def clean_reply(text: str) -> str:
    if not isinstance(text, str):
      return text
    # cut at first end-of-turn marker if present
    m = _END_RX.search(text)
    if m:
        text = text[:m.start()]
    # strip any trailing markers / junk
    text = _END_TAIL_RX.sub("", text)
    # tidy whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------- auth ----------

def _google_access_token() -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    req = gar.Request()
    creds.refresh(req)
    return creds.token

# ---------- endpoint + URL helpers ----------

def _endpoint_resource(project_id: str, location: str, endpoint_id: str) -> str:
    return f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"

def _host_from_any(s: str) -> str:
    """
    Accepts ANY of:
      - raw dedicated DNS:   "5861....us-east1-<projnum>.prediction.vertexai.goog"
      - full predict URL:    "https://...prediction.vertexai.goog/v1/projects/...:predict"
      - accidental 'endpoints.vertexai.goog' (we fix it)
    Returns ONLY the host:   "...prediction.vertexai.goog"
    """
    s = s.strip()
    if s.startswith("http://") or s.startswith("https://"):
        host = (urlparse(s).netloc or "").strip()
    else:
        host = s

    if ".endpoints.vertexai.goog" in host:
        host = host.replace(".endpoints.vertexai.goog", ".prediction.vertexai.goog")

    if not host or ".prediction.vertexai.goog" not in host:
        raise ValueError(
            f"Invalid dedicated DNS/URL: '{s}'. Expected something ending with '.prediction.vertexai.goog'."
        )
    return host

def _make_predict_url(dns_or_url: str, project_id: str, location: str, endpoint_id: str) -> str:
    """
    Build the classic Predict URL:
      https://<DEDICATED_DNS>/v1/projects/<PROJECT_ID>/locations/<LOCATION>/endpoints/<ENDPOINT_ID>:predict

    If a full predict URL is provided, use it as-is (after sanity check).
    """
    s = dns_or_url.strip()
    if s.startswith("http://") or s.startswith("https://"):
        # assume caller pasted the full predict URL that already works
        parsed = urlparse(s)
        host = parsed.netloc
        if ".prediction.vertexai.goog" not in host or not s.endswith(":predict"):
            raise ValueError(
                f"Looks like a URL, but not a valid predict URL: '{s}'. "
                f"Expected host '*.prediction.vertexai.goog' and path ending with ':predict'."
            )
        predict_url = s
    else:
        host = _host_from_any(s)
        predict_url = (
            f"https://{host}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        )

    # Preflight DNS to avoid opaque HTTP errors
    try:
        socket.getaddrinfo(host, 443)
    except socket.gaierror as e:
        raise RuntimeError(
            f"DNS failed for '{host}'. Copy the Dedicated endpoint DNS from Vertex Console."
        ) from e

    return predict_url

# ---------- response parsing ----------

def _extract_text(pred0: Any) -> str:
    """
    Be liberal: handle common shapes returned by MG/vLLM & chatCompletions bridge.
    """
    if isinstance(pred0, str):
        return pred0
    if isinstance(pred0, dict):
        # chat-completions-ish
        # e.g., { "candidates": [ { "message": { "content": "..." } } ] }
        cand = pred0.get("candidates")
        if isinstance(cand, list) and cand:
            first = cand[0]
            if isinstance(first, dict):
                msg = first.get("message") or {}
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str):
                        return c
                for k in ("content", "text", "generated_text", "output", "response"):
                    v = first.get(k)
                    if isinstance(v, str):
                        return v
        # flat content
        for k in ("content", "text", "generated_text", "output", "response"):
            v = pred0.get(k)
            if isinstance(v, str):
                return v
        # openai-like
        ch = pred0.get("choices")
        if isinstance(ch, list) and ch:
            item = ch[0]
            if isinstance(item, dict):
                msg = item.get("message", {})
                if isinstance(msg, dict):
                    v = msg.get("content")
                    if isinstance(v, str):
                        return v
    return json.dumps(pred0, ensure_ascii=False)

# ---------- Agent ----------

class Agent:
    """
    Minimal Gemma-3 agent using the *Predict* API with:
      @requestFormat = "chatCompletions"
      messages = [{role, content}, ...]
    This mirrors your working cURL.
    """

    def __init__(
        self,
        *,
        project_id: str,
        location: str,                # e.g., "us-east1"
        endpoint_id: str,             # numeric endpoint id
        dedicated_dns_or_predict_url: str,  # DNS host OR the full :predict URL
        model: str = "google/gemma3@gemma-3-1b-it",  # informative; server may ignore
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        timeout: int = 300,
    ):
        self._predict_url = _make_predict_url(
            dedicated_dns_or_predict_url, project_id, location, endpoint_id
        )
        self._endpoint_resource = _endpoint_resource(project_id, location, endpoint_id)
        self._model = model  # not strictly required by predict+chatCompletions; kept for future
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._top_p = float(top_p) if top_p is not None else None
        self._top_k = int(top_k) if top_k is not None else None
        self._stop = list(stop) if stop else ["<end_of_turn>"]
        self._timeout = int(timeout)
        self.system: Optional[str] = None
        self.history: List[Dict[str, str]] = []

    def _headers(self) -> Dict[str, str]:
        # Predict does NOT need X-Vertex-AI-Endpoint; it routes via the URL
        return {
            "Authorization": f"Bearer {_google_access_token()}",
            "Content-Type": "application/json",
        }

    def _predict_chat(self, messages: List[Dict[str, str]], **overrides) -> str:
        """
        calls :predict with the chatCompletions instance shape (your working curl)
        """
        body_inst: Dict[str, Any] = {
            "@requestFormat": "chatCompletions",
            "messages": messages,
            "max_tokens": int(overrides.get("max_tokens", self._max_tokens)),
        }
        # optional knobs (harmless if unsupported; server may ignore unknowns)
        if "temperature" in overrides or self._temperature is not None:
            body_inst["temperature"] = float(overrides.get("temperature", self._temperature))
        if (tp := overrides.get("top_p", self._top_p)) is not None:
            body_inst["top_p"] = float(tp)
        if (tk := overrides.get("top_k", self._top_k)) is not None:
            body_inst["top_k"] = int(tk)
        if (st := overrides.get("stop", self._stop)):
            body_inst["stop"] = list(st)

        payload = {"instances": [body_inst]}
        r = requests.post(self._predict_url, headers=self._headers(),
                          data=json.dumps(payload), timeout=self._timeout)
        if r.status_code != 200:
            try:
                details = r.json()
            except Exception:
                details = r.text
            raise RuntimeError(f"Predict call failed: {r.status_code} {details}")

        data = r.json()
        preds = data.get("predictions")
        if isinstance(preds, list) and preds:
            return _extract_text(preds[0]).strip()
        # fallback
        return json.dumps(data, ensure_ascii=False)

    # ---------- public API ----------

    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        raw_reply = self._predict_chat(
            messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            top_p=top_p if top_p is not None else self._top_p,
            top_k=top_k if top_k is not None else self._top_k,
            stop=stop if stop is not None else self._stop,
        )
        return clean_reply(raw_reply)
        
    def say(self, user_text: str, **gen) -> str:
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        reply = self._predict_chat(messages, **gen)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def set_system(self, text: str) -> None:
        self.system = text

    def reset(self) -> None:
        self.history.clear()
        self.system = None

# ---------- factory ----------

def agent_factory(
    *,
    project_id: str,
    location: str,
    endpoint_id: str,
    dedicated_dns_or_predict_url: str,  # DNS host OR the full :predict URL
    model: str = "google/gemma3@gemma-3-1b-it",
    **kwargs,
) -> Agent:
    return Agent(
        project_id=project_id,
        location=location,
        endpoint_id=endpoint_id,
        dedicated_dns_or_predict_url=dedicated_dns_or_predict_url,
        model=model,
        **kwargs,
    )

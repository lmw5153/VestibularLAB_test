# utils/llm.py — LLM 기반 이상응답 추론 (모순/불일치 전용)
# - 키 노출 방지: secrets/env 자동 탐지 + 에러 메시지 마스킹
# - SDK 호환: openai v1(OpenAI) 우선, 미탑재/오류 시 레거시(openai.ChatCompletion) 폴백
# - 견고성: 재시도(지수 백오프), response_format JSON 강제 → 실패 시 관대한 파싱
# - 유연성: 모델명 맵핑, BASE_URL/organization 지원(프록시/조직용)
from __future__ import annotations
from typing import Dict, Any, Optional
import os, json, time, math

# ─────────────────────────────────────────────────────────────
# 선택 임포트 (Streamlit이 없어도 동작하도록)
# ─────────────────────────────────────────────────────────────
try:
    import streamlit as st
except Exception:
    st = None

# ─────────────────────────────────────────────────────────────
# OpenAI SDKs (신형 우선, 레거시 폴백)
# ─────────────────────────────────────────────────────────────
OpenAI = None
try:
    from openai import OpenAI as _OpenAI
    OpenAI = _OpenAI
except Exception:
    OpenAI = None

LEGACY = None
try:
    import openai as _legacy_openai
    LEGACY = _legacy_openai
except Exception:
    LEGACY = None


# ─────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────
def _mask(s: Optional[str], show: int = 4) -> str:
    if not s:
        return ""
    if len(s) <= show * 2:
        return "*" * len(s)
    return s[:show] + "•" * 8 + s[-show:]


def _safe_err(msg: str) -> str:
    """키가 우연히 포함돼도 마스킹."""
    try:
        k_env = os.getenv("OPENAI_API_KEY", "")
        if k_env:
            msg = msg.replace(k_env, "****")
        if st is not None and hasattr(st, "secrets"):
            try:
                if "openai_api_key" in st.secrets and st.secrets["openai_api_key"]:
                    msg = msg.replace(st.secrets["openai_api_key"], "****")
                if "general" in st.secrets:
                    gen = st.secrets["general"]
                    if isinstance(gen, dict) and gen.get("openai_api_key"):
                        msg = msg.replace(gen["openai_api_key"], "****")
            except Exception:
                pass
    except Exception:
        pass
    return msg


def _get_api_key(user_api_key: Optional[str] = None) -> Optional[str]:
    """우선순위: 전달값 → st.secrets → 환경변수"""
    if user_api_key:
        return user_api_key
    # Streamlit secrets
    if st is not None and hasattr(st, "secrets"):
        try:
            if "openai_api_key" in st.secrets and st.secrets["openai_api_key"]:
                return st.secrets["openai_api_key"]
            if "general" in st.secrets:
                gen = st.secrets["general"]
                if isinstance(gen, dict) and gen.get("openai_api_key"):
                    return gen["openai_api_key"]
        except Exception:
            pass
    # Env
    return os.getenv("OPENAI_API_KEY")


def _get_base_and_org() -> Dict[str, Optional[str]]:
    """
    프록시/조직 지원: OPENAI_BASE_URL, OPENAI_ORG (또는 secrets.general.*)
    """
    base = os.getenv("OPENAI_BASE_URL")
    org = os.getenv("OPENAI_ORG")
    if st is not None and hasattr(st, "secrets"):
        try:
            if not base:
                base = st.secrets.get("openai_base_url", None)
            if not org:
                org = st.secrets.get("openai_org", None)
            if "general" in st.secrets:
                gen = st.secrets["general"]
                if isinstance(gen, dict):
                    base = base or gen.get("openai_base_url")
                    org = org or gen.get("openai_org")
        except Exception:
            pass
    return {"base_url": base, "organization": org}


def _compact_answers(per_survey_raw: Dict[str, list], max_items: int = 80) -> str:
    parts = []
    for key, items in per_survey_raw.items():
        seg = [key]
        for a in items[:max_items]:
            no = a.get("no", "?")
            label = a.get("label", "")
            score = a.get("score", "")
            seg.append(f"Q{no}={label}({score})")
        parts.append(" ".join(seg))
    txt = " | ".join(parts)
    return txt[:6000]  # 과도한 길이 방지


def _build_prompt(payload: Dict[str, Any], answers_compact: str) -> str:
    return f"""
당신은 임상 설문 응답의 '모순/불일치 가능성'만 점검하는 보조자입니다.
- 진단/치료/예후 언급 금지.
- 응답 간 상호모순, 기능수준-자각도 괴리, 비현실적 패턴(동일점수 연속 등)만 지적.
- 반드시 JSON만 출력하세요. 추가 설명 금지.

[요약된 응답]
{answers_compact}

[정량 요약 payload]
{json.dumps(payload, ensure_ascii=False)}

[출력 JSON 스키마]
{{
  "flags": [
    {{"id":"L1_PATTERN","severity":"low|medium|high","reason":"한줄","evidence":["DHI Q12=...","VADL Q15=..."]}}
  ],
  "triage":"low|medium|high",
  "summary_kor":"3~5문장(진단금지, 모순 가능성만)",
  "followups":["재확인 질문1","재확인 질문2","재확인 질문3"]
}}
    """.strip()


def _skeleton(msg: str = "") -> Dict[str, Any]:
    return {"flags": [], "triage": "low", "summary_kor": msg, "followups": []}


def _choose_model(model: str) -> str:
    """
    모델명 유연 처리: 미지원이면 안전 폴백.
    필요 시 사내 프록시 모델명 매핑도 여기서.
    """
    prefer = model or "gpt-4o-mini"
    allowed = {"gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"}
    if prefer not in allowed:
        return "gpt-4o"
    return prefer


def _parse_json_strict(text: str) -> Dict[str, Any]:
    t = text.strip()
    # 코드블록 제거
    if t.startswith("```"):
        # ```json ... ```
        if t.lower().startswith("```json"):
            t = t[7:]
        t = t.strip("`").strip()
    return json.loads(t)


def _retry_sleep(attempt: int, base: float = 0.7, cap: float = 6.0) -> None:
    """지수 백오프: 0.7, 1.4, 2.8 ... (최대 cap 초) + 소량 지터."""
    dur = min(cap, base * (2 ** attempt))
    # 간단 지터
    dur += (0.05 * (attempt + 1))
    time.sleep(dur)


# ─────────────────────────────────────────────────────────────
# 공개 함수
# ─────────────────────────────────────────────────────────────
def run_llm_inference(
    per_survey_raw: Dict[str, list],
    payload: Dict[str, Any],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    반환: {"flags":[], "triage":"low|medium|high", "summary_kor":"...", "followups":[...]}
    - 키/SDK/네트워크 오류는 내부에서 처리하고 summary_kor에 요약
    - 절대 키를 그대로 포함한 에러문을 반환하지 않음(마스킹)
    """
    api_key = _get_api_key(api_key)
    if not api_key:
        return _skeleton("OPENAI_API_KEY 미설정으로 LLM 건너뜀.")

    answers_compact = _compact_answers(per_survey_raw, max_items=120)
    prompt = _build_prompt(payload, answers_compact)
    model_use = _choose_model(model)
    conn = _get_base_and_org()  # base_url/organization

    # ── 1) 신형 SDK 경로 ───────────────────────────────────────────
    last_err = ""
    if OpenAI is not None:
        for attempt in range(max_retries):
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url=conn["base_url"] or None,
                    organization=conn["organization"] or None,
                )
                # response_format 우선 사용 → 실패 시 폴백
                try:
                    resp = client.chat.completions.create(
                        model=model_use,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": "당신은 임상 설문 모순 탐지 보조자입니다. 진단 금지."},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=800,
                    )
                except Exception:
                    resp = client.chat.completions.create(
                        model=model_use,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": "당신은 임상 설문 모순 탐지 보조자입니다. 진단 금지."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=800,
                    )
                txt = resp.choices[0].message.content
                try:
                    data = _parse_json_strict(txt)
                except Exception:
                    return _skeleton(f"LLM 응답 파싱 실패(비JSON): {txt[:160]}...")

                data.setdefault("flags", [])
                data.setdefault("triage", "low")
                data.setdefault("summary_kor", "")
                data.setdefault("followups", [])
                # evidence 길이 제한
                for f in data["flags"]:
                    if isinstance(f.get("evidence"), list) and len(f["evidence"]) > 6:
                        f["evidence"] = f["evidence"][:6]
                return data

            except Exception as e:
                last_err = _safe_err(str(e))
                if attempt < max_retries - 1:
                    _retry_sleep(attempt)
                else:
                    break
    else:
        last_err = "openai>=1.x 클라이언트 미탑재"

    # ── 2) 레거시 SDK 폴백 ─────────────────────────────────────────
    if LEGACY is not None:
        for attempt in range(max_retries):
            try:
                LEGACY.api_key = api_key
                base_url = conn["base_url"]
                organization = conn["organization"]
                # 레거시 SDK는 base_url/organization 지원 방식이 다를 수 있어 조건부 적용
                if organization:
                    LEGACY.organization = organization
                if base_url:
                    try:
                        LEGACY.api_base = base_url
                    except Exception:
                        pass

                resp = LEGACY.ChatCompletion.create(
                    model=model_use,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "당신은 임상 설문 모순 탐지 보조자입니다. 진단 금지."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=800,
                )
                txt = resp["choices"][0]["message"]["content"]
                try:
                    data = _parse_json_strict(txt)
                except Exception:
                    return _skeleton(f"(레거시) LLM 응답 파싱 실패(비JSON): {txt[:160]}...")

                data.setdefault("flags", [])
                data.setdefault("triage", "low")
                data.setdefault("summary_kor", "")
                data.setdefault("followups", [])
                for f in data["flags"]:
                    if isinstance(f.get("evidence"), list) and len(f["evidence"]) > 6:
                        f["evidence"] = f["evidence"][:6]
                return data

            except Exception as e2:
                last_err = _safe_err(str(e2))
                if attempt < max_retries - 1:
                    _retry_sleep(attempt)
                else:
                    break

        return _skeleton(f"LLM 호출 오류(레거시): {last_err}")

    # ── 3) 둘 다 실패 ─────────────────────────────────────────────
    return _skeleton(f"LLM 호출 준비 실패: {last_err} / 레거시 SDK도 미탑재")

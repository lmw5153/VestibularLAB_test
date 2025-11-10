# utils/llm.py — LLM 기반 이상응답 추론 (간단/안정 버전)
# - 키는 반드시 인자로 받는다(app에서 st.secrets로 읽어서 전달)
# - OpenAI v1 SDK 우선, 실패 시 간단 예외 처리
from typing import Dict, Any, Optional
import json

# 신형 SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def _skeleton(msg: str = "") -> Dict[str, Any]:
    return {"flags": [], "triage": "low", "summary_kor": msg, "followups": []}

def _compact_answers(per_survey_raw: Dict[str, list], max_items: int = 80) -> str:
    parts = []
    for key, items in per_survey_raw.items():
        seg = [key]
        for a in items[:max_items]:
            no = a.get("no", "?"); label = a.get("label", ""); score = a.get("score", "")
            seg.append(f"Q{no}={label}({score})")
        parts.append(" ".join(seg))
    return (" | ".join(parts))[:6000]

def _build_prompt(payload: Dict[str, Any], answers_compact: str) -> str:
    return f"""
당신은 임상 설문 응답의 '모순/불일치 가능성'만 점검하는 보조자입니다.
- 진단/치료/예후 언급 금지.
- 응답 간 상호모순, 기능수준-자각도 괴리, 비현실적 패턴만 지적.
- 반드시 JSON만 출력.

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

def _parse_json_relaxed(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        if t.lower().startswith("```json"):
            t = t[7:]
        t = t.strip("`").strip()
    return json.loads(t)

def run_llm_inference(
    per_survey_raw: Dict[str, list],
    payload: Dict[str, Any],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """앱에서 secrets로 받은 api_key를 그대로 넣어 호출."""
    if not api_key:
        return _skeleton("OPENAI_API_KEY 없음(secrets에 등록 필요).")
    if OpenAI is None:
        return _skeleton("openai SDK(v1) 미설치. requirements.txt에 openai>=1.0.0 추가.")

    answers_compact = _compact_answers(per_survey_raw, max_items=120)
    prompt = _build_prompt(payload, answers_compact)

    try:
        client = OpenAI(api_key=api_key)
        # JSON 응답 강제 시도
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "임상 설문 모순 탐지 보조자. 진단 금지."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
            )
        except Exception:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "임상 설문 모순 탐지 보조자. 진단 금지."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
            )

        txt = resp.choices[0].message.content
        try:
            data = _parse_json_relaxed(txt)
        except Exception:
            return _skeleton(f"LLM 응답이 JSON 형식이 아님: {txt[:160]}...")

        # 스키마 최소 보정
        data.setdefault("flags", []); data.setdefault("triage", "low")
        data.setdefault("summary_kor", ""); data.setdefault("followups", [])
        # evidence 과다 방어
        for f in data["flags"]:
            if isinstance(f.get("evidence"), list) and len(f["evidence"]) > 6:
                f["evidence"] = f["evidence"][:6]
        return data

    except Exception as e:
        return _skeleton(f"LLM 호출 오류: {e}")

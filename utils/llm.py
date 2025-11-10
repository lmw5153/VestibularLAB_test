 # utils/llm.py — LLM 기반 이상응답 추론 (모순/불일치 전용)
# 양쪽 SDK 호환: openai v1 (OpenAI) + 레거시(openai.ChatCompletion)
from typing import Dict, Any
import os, json

# 1) 시도: 신형 SDK (openai>=1.x)
OpenAI = None
try:
    from openai import OpenAI as _OpenAI
    OpenAI = _OpenAI
except Exception:
    OpenAI = None

# 2) 레거시 SDK 대비
LEGACY = None
try:
    import openai as _legacy_openai
    LEGACY = _legacy_openai
except Exception:
    LEGACY = None


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
    모델명 유연 처리: 사용자가 'gpt-4o-mini' 미지원이면 'gpt-4o'로 폴백.
    필요하면 여기서 사내 프록시 모델명 매핑도 가능.
    """
    prefer = model or "gpt-4o-mini"
    allowed = {"gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"}
    if prefer not in allowed:
        return "gpt-4o"
    return prefer


def _parse_json_strict(text: str) -> Dict[str, Any]:
    # 혹시 모델이 코드블록으로 감싸 보낼 경우 제거
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        # ```json ... ```
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)


def run_llm_inference(
    per_survey_raw: Dict[str, list],
    payload: Dict[str, Any],
    model: str = "gpt-4o",
    api_key: str = None
) -> Dict[str, Any]:
    """
    반환: {"flags":[], "triage":"low|medium|high", "summary_kor":"...", "followups":[...]}
    - SDK/모델/키 문제 등 모든 예외는 여기서 흡수하여 summary_kor에 원인 표시
    """
    # 키 확인
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _skeleton("OPENAI_API_KEY 미설정으로 LLM 건너뜀.")

    answers_compact = _compact_answers(per_survey_raw, max_items=120)
    prompt = _build_prompt(payload, answers_compact)
    model_use = _choose_model(model)

    # 1) 신형 SDK 경로
    if OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            # 일부 환경에서 response_format 미지원일 수 있어 안전 폴백
            try:
                resp = client.chat.completions.create(
                    model=model_use,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "당신은 임상 설문 모순 탐지 보조자입니다. 진단 금지."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
            except Exception:
                # 폴백: response_format 없이 요청
                resp = client.chat.completions.create(
                    model=model_use,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "당신은 임상 설문 모순 탐지 보조자입니다. 진단 금지."},
                        {"role": "user", "content": prompt},
                    ],
                )
            txt = resp.choices[0].message.content
            try:
                data = _parse_json_strict(txt)
            except Exception:
                # 모델이 JSON 형식을 안지킨 경우 방어
                return _skeleton(f"LLM 응답 파싱 실패(비JSON): {txt[:160]}...")

            data.setdefault("flags", [])
            data.setdefault("triage", "low")
            data.setdefault("summary_kor", "")
            data.setdefault("followups", [])
            # evidence 과다시 제한
            for f in data["flags"]:
                if isinstance(f.get("evidence"), list) and len(f["evidence"]) > 6:
                    f["evidence"] = f["evidence"][:6]
            return data
        except Exception as e:
            # 신형 실패 시 레거시로 폴백 시도
            err_msg = str(e)

    else:
        err_msg = "openai>=1.x 클라이언트 미탑재"

    # 2) 레거시 SDK 경로 (가능하면 시도)
    if LEGACY is not None:
        try:
            LEGACY.api_key = api_key
            resp = LEGACY.ChatCompletion.create(
                model=model_use,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "당신은 임상 설문 모순 탐지 보조자입니다. 진단 금지."},
                    {"role": "user", "content": prompt},
                ],
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
            return _skeleton(f"LLM 호출 오류(레거시): {e2}")

    # 둘 다 실패
    return _skeleton(f"LLM 호출 준비 실패: {err_msg} / 레거시 SDK도 미탑재")

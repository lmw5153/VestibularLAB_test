# app.py — Multi Survey (DHI + VADL)
# - 참여자 입력(이름/생년월일/성별/기타사항) + CSV/Sheets 저장
# - 안전 보정(문항 no/domain/text 누락)
# - VADL '적용불능' 기본 미체크 (기존 응답 시에만 복원)
# - 마지막 문항 버튼 라벨: 제출/다음 설문/다음
# - 규칙 기반 이상탐지 + LLM 추론형 옵션 (키 자동 탐지)

import os
import json
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

# 내부 모듈 (프로젝트 구조 기준)
from utils.registry import list_surveys, load_survey
from utils.export import build_row, save_df_to_gsheet
from utils.consistency import make_payload, load_rulebook, eval_rules
from utils.llm import run_llm_inference
from scoring.dhi import DHIScorer
from scoring.vadl import VADLScorer

SCORERS = {
    "DHI": DHIScorer(),
    "VADL": VADLScorer(),
}

st.set_page_config(page_title="인지 설문 플랫폼 (멀티)", layout="wide")

# ─────────────────────────────────────────────────────────────
# 안전 보정: YAML에서 누락된 필드(no/domain/text) 자동 채움
# ─────────────────────────────────────────────────────────────
def _normalize_items(items):
    norm = []
    for idx, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            it = {"text": str(it)}
        no = it.get("no", idx)
        domain = it.get("domain", "")
        text = it.get("text", "")
        rest = {k: v for k, v in it.items() if k not in ("no", "domain", "text")}
        norm.append({"no": no, "domain": domain, "text": text, **rest})
    return norm

# ─────────────────────────────────────────────────────────────
# OPENAI API 키 안전 획득 (secrets.toml/환경변수/섹션 폴백 모두 지원)
# ─────────────────────────────────────────────────────────────
def _get_openai_key():
    key = os.getenv("OPENAI_API_KEY")  # 1) 환경변수
    try:
        # 2) secrets 최상위
        if "openai_api_key" in st.secrets and st.secrets["openai_api_key"]:
            return st.secrets["openai_api_key"]
        # 3) secrets 안의 [general] 섹션 폴백
        if "general" in st.secrets:
            gen = st.secrets["general"]
            if isinstance(gen, dict) and gen.get("openai_api_key"):
                return gen["openai_api_key"]
    except Exception:
        pass
    return key

# ─────────────────────────────────────────────────────────────
# 세션 상태
# ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        page=1,
        # 참여자 정보
        participant_id="",
        participant_name="",
        participant_birth=None,   # 'YYYY-MM-DD' 문자열
        participant_sex="",
        participant_notes="",
        # 설문 진행
        preset_name="",
        selected_keys=[],   # ['DHI','VADL', ...]
        queue=[],           # 진행 순서 복사본
        curr_idx=0,         # 현재 설문 index
        answers_map={},     # {key: [ {no,domain,text,label,score}, ... ]}
        summaries={},       # {key: {total,max,domains}}
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# 사이드바: Google Sheets (옵션)
st.sidebar.subheader("Google Sheets 연동(옵션)")
gs_enable = st.sidebar.checkbox("응답을 Google Sheets로 저장", value=False)
gs_url = st.sidebar.text_input("스프레드시트 URL", placeholder="https://docs.google.com/...", disabled=not gs_enable)
gs_ws = st.sidebar.text_input("워크시트 이름", value="responses", disabled=not gs_enable)

# ─────────────────────────────────────────────────────────────
# PAGE 1 — Main: 설문 선택/프리셋/참여자 입력/시작
# ─────────────────────────────────────────────────────────────
if st.session_state.page == 1:
    st.title("🧠 Vestibular LAB 설문 플랫폼")
    st.write("전북대 병원 신경과 Vestibular LAB")
    st.write("LLM 생성형 모델을 이용하여 이상 응답을 파악합니다.")

    st.write("여러 설문을 동시에 선택하고 프리셋으로 저장해 다음에 쉽게 불러올 수 있습니다.")

    metas = list_surveys()
    key_to_title = {m["key"]: m["title"] for m in metas}

    # 프리셋 저장/불러오기 (로컬 JSON)
    presets_path = Path("data/presets.json")
    if presets_path.exists():
        presets = json.load(open(presets_path, "r", encoding="utf-8"))
    else:
        presets = {}

    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("설문 선택")
        all_keys = [m["key"] for m in metas]
        sel = st.multiselect(
            "실시할 설문을 선택하세요",
            options=all_keys,
            format_func=lambda k: key_to_title.get(k, k),
            default=st.session_state.selected_keys,
        )
        st.session_state.selected_keys = sel

        with st.expander("프리셋 관리", expanded=False):
            preset_col1, preset_col2 = st.columns([3, 1])
            with preset_col1:
                preset_name = st.text_input("프리셋 이름", value=st.session_state.preset_name)
            with preset_col2:
                if st.button("저장"):
                    presets[preset_name] = sel
                    presets_path.parent.mkdir(parents=True, exist_ok=True)
                    json.dump(presets, open(presets_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                    st.success("프리셋 저장 완료")
                    st.session_state.preset_name = preset_name
            if presets:
                pick = st.selectbox("불러오기", options=["(선택)"] + list(presets.keys()))
                if pick != "(선택)" and st.button("프리셋 적용"):
                    st.session_state.selected_keys = presets[pick]
                    st.session_state.preset_name = pick
                    st.success(f"프리셋 '{pick}' 적용")

    with cols[1]:
        st.subheader("참여자/동의")

        # 이름
        name = st.text_input("이름", value=st.session_state.participant_name)

        # 생년월일 (값이 없으면 value 인자 없이 렌더)
        if st.session_state.participant_birth:
            _birth_date = pd.to_datetime(st.session_state.participant_birth).date()
            dob = st.date_input("생년월일", value=_birth_date, key="dob")
        else:
            dob = st.date_input("생년월일", key="dob")

        # 성별
        sex_options = ["", "남", "여", "기타"]
        try:
            sex_idx = sex_options.index(st.session_state.participant_sex or "")
        except ValueError:
            sex_idx = 0
        sex = st.selectbox("성별", options=sex_options, index=sex_idx)

        # 기타사항
        notes = st.text_area("기타사항", value=st.session_state.participant_notes, height=90,
                             placeholder="알레르기, 복용약, 주의사항 등 필요 시 기입")

        # 연구 ID (선택)
        pid = st.text_input("연구 ID (선택)", value=st.session_state.participant_id)

        agree = st.checkbox("개인정보 이용에 동의합니다.")
        start_disabled = (not agree) or (not name.strip()) or (len(st.session_state.selected_keys) == 0)

        if st.button("검사 시작", type="primary", disabled=start_disabled):
            st.session_state.participant_name = name.strip()
            st.session_state.participant_birth = (dob.isoformat() if dob else None)
            st.session_state.participant_sex = sex
            st.session_state.participant_notes = notes.strip()
            st.session_state.participant_id = pid.strip()

            st.session_state.queue = list(st.session_state.selected_keys)
            st.session_state.curr_idx = 0
            st.session_state.answers_map = {}
            st.session_state.summaries = {}
            st.session_state.page = 2
            st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGE 2 — 설문 진행(순차)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == 2:
    queue = st.session_state.queue
    idx = st.session_state.curr_idx

    if idx >= len(queue):
        st.session_state.page = 3
        st.rerun()

    key = queue[idx]
    meta = load_survey(key)  # {key,title,input_type,scoring,choices?,na_label?,items:[]}
    # 🔒 안전 보정: 문항 누락 필드 자동 채움
    meta["items"] = _normalize_items(meta.get("items", []))

    items = meta["items"]
    input_type = meta.get("input_type", "radio")

    st.title(meta["title"])
    st.caption(f"설문 {idx+1} / {len(queue)}")

    # 문항 상태 초기화
    answers = st.session_state.answers_map.get(key, [])
    if not answers:
        st.session_state.answers_map[key] = []
        answers = st.session_state.answers_map[key]

    # 현재 문항 인덱스
    if f"i_{key}" not in st.session_state:
        st.session_state[f"i_{key}"] = 0
    i = st.session_state[f"i_{key}"]

    n = len(items)
    st.progress((i + 0.0001) / max(n, 1))
    st.caption(f"문항 {i+1} / {n}")

    it = items[i]
    it_no = it.get("no", i + 1)
    it_domain = it.get("domain", "")
    it_text = it.get("text", "")
    st.subheader(f"({it_domain}) {it_text}")

    # 공통 버튼 라벨 로직
    is_last_item   = (i == n - 1)
    is_last_survey = (st.session_state.curr_idx == len(st.session_state.queue) - 1)
    btn_label = "제출" if (is_last_item and is_last_survey) else ("다음 설문" if is_last_item else "다음")

    # 이전 답변 복구
    prev = answers[i] if i < len(answers) else {}

    if input_type == "radio":
        labels = [c[0] for c in meta.get("choices", [])]
        if not labels:
            st.error("이 설문은 choices가 비어 있습니다.")
            st.stop()
        default_idx = 0
        if prev and prev.get("label") in labels:
            default_idx = labels.index(prev["label"])
        sel = st.radio("응답 선택", labels, index=default_idx, key=f"radio_{key}_{i}")
        score = dict(meta.get("choices", [])).get(sel, 0)

        c1, c2 = st.columns(2)
        if c1.button("이전", disabled=(i == 0)):
            ans = {"no": it_no, "domain": it_domain, "text": it_text, "label": sel, "score": score}
            if i < len(answers): answers[i] = ans
            else: answers.append(ans)
            st.session_state[f"i_{key}"] -= 1
            st.rerun()

        if c2.button(btn_label, type="primary"):
            ans = {"no": it_no, "domain": it_domain, "text": it_text, "label": sel, "score": score}
            if i < len(answers): answers[i] = ans
            else: answers.append(ans)

            if is_last_item:
                # 설문 채점
                scorer = SCORERS.get(key)
                summary = scorer.score(answers, meta) if scorer else {"total": None, "max": None, "domains": {}}
                st.session_state.summaries[key] = summary

                if is_last_survey:
                    # 모든 설문 완료 → 결과 페이지
                    st.session_state.curr_idx += 1
                    st.session_state.page = 3
                else:
                    # 다음 설문 이어서 진행
                    st.session_state.curr_idx += 1
                    next_key = st.session_state.queue[st.session_state.curr_idx]
                    st.session_state[f"i_{next_key}"] = 0
                    st.session_state.page = 2
            else:
                # 같은 설문 내 다음 문항
                st.session_state[f"i_{key}"] += 1

            st.rerun()

    elif input_type == "slider_1_10_na":
        na_label = meta.get("na_label", "적용불능")
        # 이전 상태 복원 — 키 존재까지 확인하여 기본은 미체크
        has_score_key = isinstance(prev, dict) and ("score" in prev)
        was_na = has_score_key and (prev["score"] is None)
        prev_val = prev["score"] if (has_score_key and isinstance(prev["score"], int)) else 1

        c1, c2 = st.columns([1, 2])
        with c1:
            na = st.checkbox(na_label, value=was_na, key=f"na_{key}_{i}")
        with c2:
            val = st.slider("점수 (1–10)", 1, 10, value=prev_val, step=1, disabled=na, key=f"slider_{key}_{i}")

        if not na:
            info_map = meta.get("score_info", {})
            cat, desc = info_map.get(str(val), ["", ""])
            st.info(f"**{val}점** · **{cat}** — {desc}")
        else:
            st.warning("이 문항은 적용불능으로 저장됩니다 (합계/최대점 제외)")

        c1, c2 = st.columns(2)
        if c1.button("이전", disabled=(i == 0)):
            ans = {
                "no": it_no,
                "domain": it_domain,
                "text": it_text,
                "label": na_label if na else str(val),
                "score": None if na else val,
            }
            if i < len(answers): answers[i] = ans
            else: answers.append(ans)
            st.session_state[f"i_{key}"] -= 1
            st.rerun()

        if c2.button(btn_label, type="primary"):
            ans = {
                "no": it_no,
                "domain": it_domain,
                "text": it_text,
                "label": na_label if na else str(val),
                "score": None if na else val,
            }
            if i < len(answers): answers[i] = ans
            else: answers.append(ans)

            if is_last_item:
                scorer = SCORERS.get(key)
                summary = scorer.score(answers, meta) if scorer else {"total": None, "max": None, "domains": {}}
                st.session_state.summaries[key] = summary

                if is_last_survey:
                    st.session_state.curr_idx += 1
                    st.session_state.page = 3
                else:
                    st.session_state.curr_idx += 1
                    next_key = st.session_state.queue[st.session_state.curr_idx]
                    st.session_state[f"i_{next_key}"] = 0
                    st.session_state.page = 2
            else:
                st.session_state[f"i_{key}"] += 1

            st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGE 3 — 결과/비교/다운로드/이상탐지 + LLM 옵션
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == 3:
    st.title("결과 요약 & 비교")
    pid = st.session_state.participant_id
    ts = datetime.now().isoformat(timespec="seconds")

    # 카드 렌더링
    cols = st.columns(len(st.session_state.summaries) or 1)
    for c, (k, s) in zip(cols, st.session_state.summaries.items()):
        with c:
            st.subheader(k)
            if s.get("max") is not None:
                st.metric("총점", s["total"], delta=f"/ {s['max']}")
            else:
                st.metric("총점", s["total"])
            for dkey, dval in s.get("domains", {}).items():
                st.caption(f"{dkey}: {dval}")

    # (옵션) 참여자 요약 정보
    with st.expander("참여자 정보", expanded=False):
        st.write(f"**이름**: {st.session_state.participant_name or '-'}")
        st.write(f"**생년월일**: {st.session_state.participant_birth or '-'}")
        st.write(f"**성별**: {st.session_state.participant_sex or '-'}")
        st.write(f"**기타사항**: {st.session_state.participant_notes or '-'}")
        st.write(f"**연구 ID**: {pid or '-'}")

    # 설문별 raw 응답표
    with st.expander("설문별 응답표"):
        for k, answers in st.session_state.answers_map.items():
            st.markdown(f"### {k}")
            df = pd.DataFrame(
                [
                    {
                        "no": a.get("no", idx + 1),
                        "domain": a.get("domain", ""),
                        "question": a.get("text", ""),
                        "response_label": a.get("label", ""),
                        "response_score": ("" if a.get("score") is None else a.get("score")),
                    }
                    for idx, a in enumerate(answers)
                ]
            )
            st.dataframe(df, use_container_width=True)

    # 통합 CSV 행 구성
    per_survey_summaries = st.session_state.summaries
    per_survey_raw = st.session_state.answers_map
    row = build_row(ts, pid, st.session_state.preset_name, per_survey_summaries, per_survey_raw)

    # ⬇️ 참여자 기본정보를 CSV에도 포함
    row.update({
        "name": st.session_state.participant_name,
        "birth": st.session_state.participant_birth or "",
        "sex": st.session_state.participant_sex or "",
        "notes": st.session_state.participant_notes or "",
    })

    df_out = pd.DataFrame([row])

    csv_buf = StringIO()
    df_out.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 통합 CSV 다운로드",
        data=csv_buf.getvalue().encode("utf-8-sig"),
        file_name=f"{ts.replace(':','-')}_summary.csv",
        mime="text/csv",
    )

    # (옵션) Google Sheets 저장
    if gs_enable and gs_url:
        try:
            save_df_to_gsheet(df_out, gs_url, gs_ws)
            st.success("Google Sheets 저장 완료")
        except Exception as e:
            st.error(f"Google Sheets 저장 실패: {e}")

    st.divider()

    # ── 규칙 기반 이상탐지(경량)
    st.subheader("이상 응답 탐지 (규칙 기반·경량)")
    payload = make_payload(per_survey_raw, per_survey_summaries)
    rulebook = load_rulebook(Path("rules/rulebook_v1.json"))
    flags = eval_rules(payload, rulebook)

    if not flags:
        st.success("모순 신호가 없습니다.")
        row["is_consistent"] = True
        row["flags_json"] = "[]"
    else:
        for f in flags:
            st.warning(f"**{f['id']}** · {f['reason']}  \n제안: {', '.join(f.get('suggestion', []))}")
        row["is_consistent"] = False
        row["flags_json"] = json.dumps(flags, ensure_ascii=False)

        # 갱신 저장/다운로드 갱신
        df_out = pd.DataFrame([row])
        csv_buf = StringIO()
        df_out.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            "📥 통합 CSV(플래그 포함) 재다운로드",
            data=csv_buf.getvalue().encode("utf-8-sig"),
            file_name=f"{ts.replace(':','-')}_summary_flags.csv",
            mime="text/csv",
        )
        if gs_enable and gs_url:
            try:
                save_df_to_gsheet(df_out, gs_url, gs_ws)
                st.success("Google Sheets 저장 완료 (플래그 포함)")
            except Exception as e:
                st.error(f"Google Sheets 저장 실패: {e}")

    st.divider()

    # === LLM 기반 이상응답 추론(옵션) ======================================
    st.subheader("LLM 기반 이상응답 추론 (모순 가능성 제시)")
    llm_on = st.checkbox("LLM 사용 (진단 아님, 모순 가능성만 요약)", value=False)
    if llm_on and not _get_openai_key():
        st.info("🔑 OPENAI_API_KEY가 설정되지 않았습니다. 환경변수 또는 .streamlit/secrets.toml에 키를 넣어주세요.")
    llm_model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0, disabled=not llm_on)

    if llm_on and st.button("LLM으로 모순 가능성 분석"):
        ai = run_llm_inference(
            per_survey_raw=per_survey_raw,
            payload=payload,
            model=llm_model,
            api_key=_get_openai_key()  # ← secrets.toml 없어도 안전
        )

        tri = ai.get("triage", "low")
        if tri == "high":
            st.error("전반 주의도: HIGH")
        elif tri == "medium":
            st.warning("전반 주의도: MEDIUM")
        else:
            st.info("전반 주의도: LOW")

        if ai.get("summary_kor"):
            st.markdown("**요약**")
            st.write(ai["summary_kor"])

        flags_ai = ai.get("flags", [])
        if flags_ai:
            st.markdown("**지적된 모순 가능성 (LLM)**")
            for f in flags_ai:
                rid = f.get("id", "Lx")
                sev = f.get("severity", "low")
                rsn = f.get("reason", "")
                evd = f.get("evidence", []) or []
                msg = f"**{rid}** · severity={sev} — {rsn}"
                if sev == "high": st.error(msg)
                elif sev == "medium": st.warning(msg)
                else: st.info(msg)
                if evd:
                    st.caption("근거: " + "; ".join(evd[:6]))

        fus = ai.get("followups", [])
        if fus:
            st.markdown("**재확인 질문 제안**")
            for q in fus[:5]:
                st.write("• " + q)

        # CSV/Sheets에 저장 컬럼 추가
        row["ai_triage"] = tri
        row["ai_summary_kor"] = ai.get("summary_kor", "")
        row["ai_flags_json"] = json.dumps(flags_ai, ensure_ascii=False)
        row["ai_followups_json"] = json.dumps(fus, ensure_ascii=False)

        df_out = pd.DataFrame([row])
        csv_buf = StringIO(); df_out.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            "📥 통합 CSV(LLM 결과 포함) 재다운로드",
            data=csv_buf.getvalue().encode("utf-8-sig"),
            file_name=f"{ts.replace(':','-')}_summary_llm.csv",
            mime="text/csv",
        )
        if gs_enable and gs_url:
            try:
                save_df_to_gsheet(df_out, gs_url, gs_ws)
                st.success("Google Sheets 저장 완료 (LLM 결과 포함)")
            except Exception as e:
                st.error(f"Google Sheets 저장 실패: {e}")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("처음으로"):
        st.session_state.page = 1
        st.rerun()
    if c2.button("다시 진행"):
        st.session_state.page = 2
        st.session_state.curr_idx = 0
        st.rerun()

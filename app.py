# app.py — Multi Survey Platform (Mouse-only / No keyboard hooks)
# - Sidebar collapsed
# - DOB yyyy.mm.dd 텍스트(선택), 개인정보 동의만 있으면 시작
# - Surveys: DHI, VADL, MIDAS, HIT-6, VAS-D, PHQ-9, GAD-7
# - VADL: 적용불능(NA) 지원
# - CSV 요약( *_max 컬럼 제거 )
# - Google Sheets(옵션)
# - LLM 분석: st.secrets["openai_api_key"]만 사용

import os, sys, time, json
from io import StringIO
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Project path
# ─────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────
# Internal modules
# ─────────────────────────────────────────────────────────────
from utils.registry import list_surveys, load_survey
from utils.export import build_row, save_df_to_gsheet
from utils.consistency import make_payload, load_rulebook, eval_rules
from utils.llm import run_llm_inference

from scoring.dhi import DHIScorer
from scoring.vadl import VADLScorer
from scoring.midas import MIDASScorer
from scoring.hit6 import HIT6Scorer
from scoring.vasd import VASDScorer
from scoring.phq9 import PHQ9Scorer
from scoring.gad7 import GAD7Scorer
from scoring.hads import HADSScorer
from scoring.kgds import KGDSScorer

SCORERS = {
    "DHI": DHIScorer(),
    "VADL": VADLScorer(),
    "MIDAS": MIDASScorer(),
    "HIT6": HIT6Scorer(),
    "VASD": VASDScorer(),
    "PHQ9": PHQ9Scorer(),
    "GAD7": GAD7Scorer(),
    "HADS": HADSScorer(),
    "KGDS": KGDSScorer(),
}

st.set_page_config(
    page_title="설문 플랫폼 Vestibular LAB",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# Secrets helpers
# ─────────────────────────────────────────────────────────────
def get_secret_openai_key() -> str:
    try:
        if "openai_api_key" in st.secrets and st.secrets["openai_api_key"]:
            return st.secrets["openai_api_key"]
        if "general" in st.secrets:
            gen = st.secrets["general"]
            if isinstance(gen, dict) and gen.get("openai_api_key"):
                return gen["openai_api_key"]
    except Exception:
        pass
    return ""

def mask_key(k: str, show: int = 4) -> str:
    if not k:
        return "(없음)"
    return k if len(k) <= show * 2 else k[:show] + "•" * 8 + k[-show:]

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
def normalize_items(items):
    """items에 no/domain/text가 없으면 보정."""
    out = []
    for idx, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            it = {"text": str(it)}
        out.append({
            "no": it.get("no", idx),
            "domain": it.get("domain", ""),
            "text": it.get("text", ""),
            **{k: v for k, v in it.items() if k not in ("no", "domain", "text")}
        })
    return out

def init_state():
    defaults = dict(
        page=1,
        # participant
        participant_id="", participant_name="",
        participant_birth="", participant_sex="", participant_notes="",
        # survey selection & progress
        preset_name="", selected_keys=[], queue=[], curr_idx=0,
        answers_map={}, summaries={},
        # UX
        loading_until=0.0,
        _pending_preset=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────
# Sidebar (collapsed by default)
# ─────────────────────────────────────────────────────────────
st.sidebar.subheader("Google Sheets 연동(옵션)")
gs_enable = st.sidebar.checkbox("응답을 Google Sheets로 저장", value=False)
gs_url = st.sidebar.text_input("스프레드시트 URL", placeholder="https://docs.google.com/...", disabled=not gs_enable)
gs_ws  = st.sidebar.text_input("워크시트 이름", value="responses", disabled=not gs_enable)

with st.sidebar.expander("🔐 LLM 키 상태(마스킹)"):
    api_key = get_secret_openai_key()
    st.write("OPENAI_API_KEY:", mask_key(api_key))
    st.caption("※ 키는 secrets에만 저장되며, 브라우저로 원문은 노출하지 않습니다.")

# ─────────────────────────────────────────────────────────────
# PAGE 1 — Main
# ─────────────────────────────────────────────────────────────
if st.session_state.page == 1:
    st.title("🧠 설문 검사 플랫폼 — Vestibular LAB")
    st.write("전북대 신경과 내 자체 개발한 설문 플랫폼입니다")
    st.write("설문지 종류는 현재 9개가 구현되었고 지속적으로 업데이트 예정입니다")
    st.write("생성형 AI를 이용한 응답에 대한 신뢰성을 확인할 수 있습니다")
    st.caption("Minwoo Lee")
    metas = list_surveys()
    key_to_title = {m["key"]: m["title"] for m in metas}
    all_keys = [m["key"] for m in metas]

    # sanitize selection
    st.session_state.selected_keys = [k for k in st.session_state.selected_keys if k in all_keys]

    # Presets
    presets_path = Path("data/presets.json")
    if presets_path.exists():
        try:
            presets = json.load(open(presets_path, "r", encoding="utf-8"))
        except Exception:
            presets = {}
    else:
        presets = {}

    # pending preset apply
    pending = st.session_state.get("_pending_preset", None)
    if pending:
        raw = presets.get(pending, [])
        if isinstance(raw, dict): raw = list(raw.keys())
        elif isinstance(raw, str): raw = [x.strip() for x in raw.split(",") if x.strip()]
        st.session_state.selected_keys = [k for k in raw if k in all_keys]
        st.session_state.preset_name = pending
        st.session_state.loading_until = time.time() + 2.0
        st.session_state._pending_preset = None
        st.rerun()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("설문 선택")

        def on_select_change():
            st.session_state.loading_until = time.time() + 2.0

        st.multiselect(
            "실시할 설문을 선택하세요",
            options=all_keys,
            format_func=lambda k: key_to_title.get(k, k),
            key="selected_keys",
            on_change=on_select_change,
        )

        remain = st.session_state.loading_until - time.time()
        if remain > 0:
            with st.spinner("설문 구성을 불러오는 중..."):
                time.sleep(min(remain, 2.0))
            st.session_state.loading_until = 0.0
            st.rerun()

        with st.expander("프리셋 관리", expanded=False):
            c1, c2 = st.columns([3, 1])
            with c1:
                preset_name = st.text_input("프리셋 이름", value=st.session_state.preset_name)
            with c2:
                if st.button("저장"):
                    if preset_name.strip():
                        presets[preset_name.strip()] = st.session_state.selected_keys
                        presets_path.parent.mkdir(parents=True, exist_ok=True)
                        json.dump(presets, open(presets_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                        st.success("프리셋 저장 완료")
                        st.session_state.preset_name = preset_name.strip()
                    else:
                        st.warning("프리셋 이름을 입력하세요.")

            pick = st.selectbox("불러오기", options=["(선택)"] + list(presets.keys()))
            if pick != "(선택)":
                if st.button("프리셋 적용", key="apply_preset_btn"):
                    st.session_state._pending_preset = pick
                    st.rerun()

    with right:
        st.subheader("참여자/동의")

        name = st.text_input("이름 (선택)", value=st.session_state.participant_name)
        dob_text = st.text_input(
            "생년월일 (yyyy.mm.dd, 선택)",
            value=(st.session_state.participant_birth or ""),
            placeholder="예) 1992.07.15"
        )
        sex = st.selectbox(
            "성별 (선택)", ["", "남", "여", "기타"],
            index=["","남","여","기타"].index(st.session_state.participant_sex or "")
        )
        notes = st.text_area(
            "기타사항 (선택)",
            value=st.session_state.participant_notes,
            height=90,
            placeholder="알레르기, 복용약, 주의사항 등"
        )
        pid = st.text_input("연구 ID (선택)", value=st.session_state.participant_id)

        agree = st.checkbox("개인정보 이용에 동의합니다.")
        start_disabled = not agree  # 동의만 하면 시작 가능

        if st.button("검사 시작", type="primary", disabled=start_disabled):
            # DOB parsing
            birth_iso = ""
            s = dob_text.strip()
            if s:
                for sep in [".", "-", "/"]:
                    if sep in s:
                        parts = s.split(sep)
                        if len(parts) == 3:
                            y, m, d = parts
                            try:
                                y, m, d = int(y), int(m), int(d)
                                birth_iso = f"{y:04d}-{m:02d}-{d:02d}"
                            except Exception:
                                birth_iso = ""
                        break

            st.session_state.participant_name = name.strip()
            st.session_state.participant_birth = birth_iso
            st.session_state.participant_sex = sex
            st.session_state.participant_notes = notes.strip()
            st.session_state.participant_id = pid.strip()

            st.session_state.queue = list(st.session_state.selected_keys)
            st.session_state.curr_idx = 0
            st.session_state.answers_map = {}
            st.session_state.summaries = {}
            st.session_state.page = 2
            st.session_state.loading_until = time.time() + 1.0
            st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGE 2 — Survey flow (mouse only)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == 2:
    queue = st.session_state.queue
    idx = st.session_state.curr_idx
    if idx >= len(queue):
        st.session_state.page = 3
        st.rerun()

    key = queue[idx]
    meta = load_survey(key)
    meta["items"] = normalize_items(meta.get("items", []))
    items = meta["items"]
    input_type = meta.get("input_type", "radio")

    st.title(meta["title"])
    st.caption(f"설문 {idx+1} / {len(queue)}")

    answers = st.session_state.answers_map.get(key, [])
    if not answers:
        st.session_state.answers_map[key] = []
        answers = st.session_state.answers_map[key]

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

    def _qtitle(no, domain, text):
        no_str = f"Q{no}" if no is not None else ""
        dom_str = f" ({domain})" if domain else ""
        return f"{no_str}{dom_str}. {text}".strip()

    st.subheader(_qtitle(it_no, it_domain, it_text))

    is_last_item = (i == n - 1)
    is_last_survey = (st.session_state.curr_idx == len(st.session_state.queue) - 1)
    btn_label = "제출" if (is_last_item and is_last_survey) else ("다음 설문" if is_last_item else "다음")

    # 기본값 주입(문항 진입 시 한 번만)
    if input_type == "radio":
        _choices = it.get("choices") or meta.get("choices", [])
        labels = [c[0] for c in _choices]
        if labels:
            ss_key = f"radio_{key}_{i}"
            if ss_key not in st.session_state:
                st.session_state[ss_key] = labels[0]
    elif input_type == "slider_0_10":
        ss_key = f"vas_{key}_{i}"
        if ss_key not in st.session_state:
            st.session_state[ss_key] = int(it.get("min", 0))
    elif input_type == "slider_1_10_na":
        ss_na  = f"na_{key}_{i}"
        ss_val = f"slider_{key}_{i}"
        if ss_na not in st.session_state:
            st.session_state[ss_na] = False
        if ss_val not in st.session_state:
            st.session_state[ss_val] = 1
    elif input_type == "number_int":
        ss_key = f"num_{key}_{i}"
        if ss_key not in st.session_state:
            st.session_state[ss_key] = int(it.get("min", 0))

    # 입력 렌더링 + 버튼으로만 네비게이션
    def _save_and_go_next():
        if input_type == "radio":
            sel = st.session_state.get(f"radio_{key}_{i}")
            _choices = it.get("choices") or meta.get("choices", [])
            score = dict(_choices).get(sel, 0)
            ans = {"no": it_no, "domain": it_domain, "text": it_text, "label": sel, "score": score}
        elif input_type == "number_int":
            val = int(st.session_state.get(f"num_{key}_{i}", int(it.get("min", 0))))
            ans = {"no": it_no, "domain": it_domain, "text": it_text, "label": str(val), "score": val}
        elif input_type == "slider_0_10":
            val = int(st.session_state.get(f"vas_{key}_{i}", int(it.get("min", 0))))
            ans = {"no": it_no, "domain": it_domain, "text": it_text, "label": str(val), "score": val}
        elif input_type == "slider_1_10_na":
            na  = st.session_state.get(f"na_{key}_{i}", False)
            val = st.session_state.get(f"slider_{key}_{i}", 1)
            if na:
                ans = {"no": it_no, "domain": it_domain, "text": it_text,
                       "label": meta.get("na_label","적용불능"), "score": None}
            else:
                ans = {"no": it_no, "domain": it_domain, "text": it_text,
                       "label": str(val), "score": int(val)}
        else:
            return

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

    # 위젯
    if input_type == "radio":
        _choices = it.get("choices") or meta.get("choices", [])
        labels = [c[0] for c in _choices]
        if not labels:
            st.error("이 문항은 choices가 비어 있습니다."); st.stop()
        curr = st.session_state.get(f"radio_{key}_{i}", labels[0])
        default_idx = labels.index(curr) if curr in labels else 0
        sel = st.radio("응답 선택", labels, index=default_idx, key=f"radio_{key}_{i}")
        score = dict(_choices).get(sel, 0)

        c1, c2 = st.columns(2)
        if c1.button("이전", disabled=(i == 0)):
            ans = {"no": it_no, "domain": it_domain, "text": it_text, "label": sel, "score": score}
            if i < len(answers): answers[i] = ans
            else: answers.append(ans)
            if i > 0:
                st.session_state[f"i_{key}"] -= 1
            st.rerun()

        if c2.button(btn_label, type="primary"):
            _save_and_go_next()
            st.rerun()

    elif input_type == "slider_1_10_na":
        na_label = meta.get("na_label", "적용불능")
        na  = st.session_state.get(f"na_{key}_{i}", False)
        val = int(st.session_state.get(f"slider_{key}_{i}", 1))

        c1, c2 = st.columns([1, 2])
        with c1:
            na = st.checkbox(na_label, value=na, key=f"na_{key}_{i}")
        with c2:
            val = st.slider("점수 (1–10)", 1, 10, value=val, step=1, disabled=na, key=f"slider_{key}_{i}")

        if not na:
            info_map = meta.get("score_info", {})
            cat, desc = info_map.get(str(val), ["", ""])
            st.info(f"**{val}점** · **{cat}** — {desc}")
        else:
            st.warning("이 문항은 적용불능으로 저장됩니다 (합계/최대점 제외)")

        c1, c2 = st.columns(2)
        if c1.button("이전", disabled=(i == 0)):
            ans = {"no": it_no, "domain": it_domain, "text": it_text,
                   "label": na_label if na else str(val), "score": None if na else val}
            if i < len(answers): answers[i] = ans
            else: answers.append(ans)
            if i > 0:
                st.session_state[f"i_{key}"] -= 1
            st.rerun()

        if c2.button(btn_label, type="primary"):
            _save_and_go_next()
            st.rerun()

    elif input_type == "number_int":
        val = int(st.session_state.get(f"num_{key}_{i}", int(it.get("min", 0))))
        val = st.number_input("정수 입력", min_value=int(it.get("min", 0)), max_value=int(it.get("max", 999)),
                              step=1, value=int(val), key=f"num_{key}_{i}")

        c1, c2 = st.columns(2)
        if c1.button("이전", disabled=(i == 0)):
            ans = {"no": it_no, "domain": it_domain, "text": it_text, "label": str(val), "score": int(val)}
            if i < len(answers): answers[i] = ans
            else: answers.append(ans)
            if i > 0:
                st.session_state[f"i_{key}"] -= 1
            st.rerun()

        if c2.button(btn_label, type="primary"):
            _save_and_go_next()
            st.rerun()

    elif input_type == "slider_0_10":
        val = int(st.session_state.get(f"vas_{key}_{i}", int(it.get("min", 0))))
        val = st.slider("점수 (0–10)", int(it.get("min", 0)), int(it.get("max", 10)),
                        value=val, step=1, key=f"vas_{key}_{i}")

        c1, c2 = st.columns(2)
        if c1.button("이전", disabled=(i == 0)):
            ans = {"no": it_no, "domain": it_domain, "text": it_text, "label": str(val), "score": int(val)}
            if i < len(answers): answers[i] = ans
            else: answers.append(ans)
            if i > 0:
                st.session_state[f"i_{key}"] -= 1
            st.rerun()

        if c2.button(btn_label, type="primary"):
            _save_and_go_next()
            st.rerun()

    else:
        st.error(f"지원하지 않는 input_type: {input_type}")

# ─────────────────────────────────────────────────────────────
# PAGE 3 — Results (simple, with hideable detail tables)
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == 3:
    st.title("결과 요약 & 비교")
    pid = st.session_state.participant_id
    ts = datetime.now().isoformat(timespec="seconds")

    per_summ = st.session_state.summaries
    per_raw  = st.session_state.answers_map

    # 1) 상단 요약(기존 단순 metric 방식 유지)
    cols = st.columns(len(per_summ) or 1)
    for c, (k, s) in zip(cols, per_summ.items()):
        with c:
            st.subheader(k)
            if s.get("max") is not None:
                st.metric("총점", s["total"], delta=f"/ {s['max']}")
            else:
                st.metric("총점", s["total"])
            if "severity" in s:
                st.caption(f"등급: {s['severity']}")
            for dkey, dval in s.get("domains", {}).items():
                st.caption(f"{dkey}: {dval}")

    # 2) 참여자 정보 (선택)
    with st.expander("참여자 정보", expanded=False):
        st.write(f"**이름**: {st.session_state.participant_name or '-'}")
        st.write(f"**생년월일**: {st.session_state.participant_birth or '-'}")
        st.write(f"**성별**: {st.session_state.participant_sex or '-'}")
        st.write(f"**기타사항**: {st.session_state.participant_notes or '-'}")
        st.write(f"**연구 ID**: {pid or '-'}")

    # 3) 설문별 응답표 — 숨기기 토글 추가
    st.markdown("### 설문별 응답표")
    hide_details = st.checkbox("설문별 응답표 숨기기", value=True)
    if not hide_details:
        # 각 설문을 접을 수 있도록 개별 expander 제공
        for k, answers in per_raw.items():
            with st.expander(f"{k} 응답표", expanded=True):
                df = pd.DataFrame([
                    {"no": a.get("no", i+1),
                     "domain": a.get("domain",""),
                     "question": a.get("text",""),
                     "response_label": a.get("label",""),
                     "response_score": ("" if a.get("score") is None else a.get("score"))}
                    for i, a in enumerate(answers)
                ])
                # 인쇄 레이아웃 제거 요청에 따라 단순 표로만 표시
                st.table(df)

    # 4) 통합 CSV 다운로드 ( *_max 제외)
    from utils.export import build_row, save_df_to_gsheet
    row = build_row(ts, pid, st.session_state.preset_name, per_summ, per_raw)
    row.update({
        "name":  st.session_state.participant_name,
        "birth": st.session_state.participant_birth or "",
        "sex":   st.session_state.participant_sex or "",
        "notes": st.session_state.participant_notes or "",
    })

    df_out = pd.DataFrame([row])
    drop_cols = [c for c in df_out.columns if c.endswith("_max")]
    if drop_cols:
        df_out = df_out.drop(columns=drop_cols, errors="ignore")

    from io import StringIO
    buf = StringIO(); df_out.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button("📥 통합 CSV 다운로드", data=buf.getvalue().encode("utf-8-sig"),
                       file_name=f"{ts.replace(':','-')}_summary.csv", mime="text/csv")

    # 5) Google Sheets 저장(옵션)
    try:
        if 'gs_enable' in globals() and gs_enable and gs_url:
            save_df_to_gsheet(df_out, gs_url, gs_ws)
            st.success("Google Sheets 저장 완료")
    except Exception as e:
        st.error(f"Google Sheets 저장 실패: {e}")

    # 6) 규칙 기반 이상응답 + LLM(기존 유지)
    st.divider()
    st.subheader("이상 응답 탐지 (규칙 기반·경량)")
    from utils.consistency import make_payload, load_rulebook, eval_rules
    from pathlib import Path
    payload = make_payload(per_raw, per_summ)
    rulebook = load_rulebook(Path("rules/rulebook_v1.json"))
    flags = eval_rules(payload, rulebook)

    if not flags:
        st.success("모순 신호가 없습니다.")
    else:
        for f in flags:
            st.warning(f"**{f['id']}** · {f['reason']}  \n제안: {', '.join(f.get('suggestion', []))}")

    st.divider()
    st.subheader("LLM 기반 이상응답 추론 (모순 가능성 제시)")
    llm_on = st.checkbox("LLM 사용", value=False)
    llm_model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=not llm_on)

    if llm_on and st.button("LLM으로 모순 가능성 분석"):
        # secrets에서 키 로딩
        def _get_secret_openai_key():
            try:
                if "openai_api_key" in st.secrets and st.secrets["openai_api_key"]:
                    return st.secrets["openai_api_key"]
                if "general" in st.secrets:
                    gen = st.secrets["general"]
                    if isinstance(gen, dict) and gen.get("openai_api_key"):
                        return gen["openai_api_key"]
            except Exception:
                pass
            return ""
        key_api = _get_secret_openai_key()

        if not key_api:
            st.info("🔑 Secrets에 openai_api_key가 없습니다. App Settings → Secrets에 등록하세요.")
        else:
            ai = run_llm_inference(per_survey_raw=per_raw, payload=payload, model=llm_model, api_key=key_api)
            tri = ai.get("triage", "low")
            if tri == "high": st.error("전반 주의도: HIGH")
            elif tri == "medium": st.warning("전반 주의도: MEDIUM")
            else: st.info("전반 주의도: LOW")

            if ai.get("summary_kor"):
                st.markdown("**요약**"); st.write(ai["summary_kor"])
            if ai.get("flags"):
                st.markdown("**지적된 모순 가능성**")
                for f in ai["flags"]:
                    st.write(f"- {f.get('id','Lx')}: {f.get('reason','')}")
                    ev = f.get("evidence") or []
                    if ev: st.caption("근거: " + "; ".join(ev[:6]))
            if ai.get("followups"):
                st.markdown("**재확인 질문 제안**")
                for q in ai["followups"][:5]:
                    st.write("• " + q)

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("처음으로"):
        st.session_state.page = 1; st.rerun()
    if c2.button("다시 진행"):
        st.session_state.page = 2; st.session_state.curr_idx = 0; st.rerun()

# app.py — Multi Survey (DHI + VADL) / 프리셋 안전 적용 + 지연 스피너 + LLM 키는 Secrets 전용
# - 설문 선택 단일 클릭 반영 + 2초 로딩 스피너(Cloud 플리커 완화)
# - 프리셋 적용: 같은 렌더 사이클에서 위젯 값 직접 변경 금지 → '대기 플래그' + rerun 훅
# - 참여자 정보(이름/생년월일/성별/메모/ID)
# - YAML 설문 로드(utils.registry)
# - DHI/VADL 채점, CSV/Google Sheets 저장
# - 규칙 기반 이상탐지 + LLM 기반 모순 가능성 요약
# - LLM API 키는 st.secrets["openai_api_key"] (또는 secrets.general.openai_api_key)만 사용

import os, sys, time, json
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

# --- 프로젝트 루트를 sys.path에 강제 주입(배포 경로 차이 방지) ---
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------------------

# 내부 모듈
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
SCORERS = {
    "DHI": DHIScorer(),
    "VADL": VADLScorer(),
    "MIDAS": MIDASScorer(),
    "HIT6": HIT6Scorer(),
    "VASD": VASDScorer(),   # ← 추가
    "PHQ9": PHQ9Scorer(),   # ← 추가
    "GAD7": GAD7Scorer(),   # ← 추가
}
#st.set_page_config(page_title="인지 설문 플랫폼 (멀티)", layout="wide")
st.set_page_config(
    page_title="인지 설문 플랫폼 (멀티)",
    layout="wide",
    initial_sidebar_state="collapsed"   # ← 사이드바 기본 접힘
)

# ─────────────────────────────────────────────────────────────
# 유틸: LLM 키는 '오직' Streamlit Secrets에서만 읽기
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
# 안전 보정: YAML items 필수키 보정
# ─────────────────────────────────────────────────────────────
def normalize_items(items):
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

def _qtitle(no, domain, text):
# 번호는 항상 보이게, 도메인은 있으면 괄호로
    no_str = f"Q{no}" if no is not None else ""
    dom_str = f" ({domain})" if domain else ""
    return f"{no_str}{dom_str}. {text}".strip()
# ─────────────────────────────────────────────────────────────
# 세션 초기화
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = dict(
        page=1,
        # 참여자
        participant_id="", participant_name="",
        participant_birth=None, participant_sex="", participant_notes="",
        # 진행
        preset_name="", selected_keys=[], queue=[], curr_idx=0,
        answers_map={}, summaries={},
        # UX
        loading_until=0.0,
        _pending_preset=None,  # 프리셋 적용 대기 플래그
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────
# 사이드바: Google Sheets + LLM 키 상태
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
# PAGE 1 — 메인(설문 선택/프리셋/참여자/시작)
# ─────────────────────────────────────────────────────────────
if st.session_state.page == 1:
    st.title("🧠 인지 설문 플랫폼 — Multi Survey")

    metas = list_surveys()
    key_to_title = {m["key"]: m["title"] for m in metas}
    all_keys = [m["key"] for m in metas]

    # 옵션에 없는 값 제거 (렌더 전 정리)
    st.session_state.selected_keys = [k for k in st.session_state.selected_keys if k in all_keys]

    # 프리셋 로드
    presets_path = Path("data/presets.json")
    if presets_path.exists():
        try:
            presets = json.load(open(presets_path, "r", encoding="utf-8"))
        except Exception:
            presets = {}
    else:
        presets = {}

    # ▼▼▼ 프리셋 '적용 대기' 훅: multiselect 렌더 전에만 동작 ▼▼▼
    pending = st.session_state.get("_pending_preset", None)
    if pending:
        raw = presets.get(pending, [])
        if isinstance(raw, dict):
            raw = list(raw.keys())
        elif isinstance(raw, str):
            raw = [x.strip() for x in raw.split(",") if x.strip()]

        st.session_state.selected_keys = [k for k in raw if k in all_keys]
        st.session_state.preset_name = pending
        st.session_state.loading_until = time.time() + 2.0
        st.session_state._pending_preset = None
        st.rerun()
    # ▲▲▲ 프리셋 '적용 대기' 훅 끝 ▲▲▲

    left, right = st.columns([2, 1])

    # ── 좌측: 설문 선택/프리셋
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

        # 선택 지연 스피너
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

    # ── 우측: 참여자/동의 (생년월일 텍스트 입력 + 동의만으로 시작 가능)
    with right:
        st.subheader("참여자/동의")

        # 선택 입력들
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
            "기타사항 (선택)", value=st.session_state.participant_notes, height=90,
            placeholder="알레르기, 복용약, 주의사항 등"
        )
        pid = st.text_input("연구 ID (선택)", value=st.session_state.participant_id)

        agree = st.checkbox("개인정보 이용에 동의합니다.")
        start_disabled = not agree  # 동의만 체크되면 시작 가능

        if st.button("검사 시작", type="primary", disabled=start_disabled):
            # 생년월일 텍스트 → YYYY-MM-DD 약식 파싱
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
# PAGE 2 — 설문 진행(순차)
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

    prev = answers[i] if i < len(answers) else {}

    # === 이하: input_type 분기 (radio / slider_1_10_na / number_int / slider_0_10) ===
    # ... (기존 코드 그대로 유지) ...

    # 키보드 사용 팁 (선택적)
    with st.expander("⌨️ 키보드 사용 팁", expanded=False):
        st.markdown(
            "- **Radio**: 포커스 상태에서 **↑ / ↓** 로 선택 이동\n"
            "- **Slider**: 포커스 상태에서 **← / →** 로 값 조절\n"
            "- **다음/제출** 버튼에 포커스 후 **Enter/Space** 로 진행\n"
            "- **이전** 버튼에 포커스 후 **Enter** 로 뒤로가기\n"
            "\n*전역 단축키는 브라우저 정책상 제한이 있어 기본 위젯 조작을 활용합니다.*"
        )


# ─────────────────────────────────────────────────────────────
# PAGE 3 — 결과/다운로드/이상탐지 + LLM
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == 3:
    st.title("결과 요약 & 비교")
    pid = st.session_state.participant_id
    ts = datetime.now().isoformat(timespec="seconds")

    cols = st.columns(len(st.session_state.summaries) or 1)
    for c, (k, s) in zip(cols, st.session_state.summaries.items()):
        with c:
            st.subheader(k)
            if s.get("max") is not None: st.metric("총점", s["total"], delta=f"/ {s['max']}")
            else: st.metric("총점", s["total"])
            for dkey, dval in s.get("domains", {}).items():
                st.caption(f"{dkey}: {dval}")

    with st.expander("참여자 정보", expanded=False):
        st.write(f"**이름**: {st.session_state.participant_name or '-'}")
        st.write(f"**생년월일**: {st.session_state.participant_birth or '-'}")
        st.write(f"**성별**: {st.session_state.participant_sex or '-'}")
        st.write(f"**기타사항**: {st.session_state.participant_notes or '-'}")
        st.write(f"**연구 ID**: {pid or '-'}")

    with st.expander("설문별 응답표"):
        for k, answers in st.session_state.answers_map.items():
            st.markdown(f"### {k}")
            df = pd.DataFrame([
                {"no": a.get("no", i+1), "domain": a.get("domain",""),
                 "question": a.get("text",""), "response_label": a.get("label",""),
                 "response_score": ("" if a.get("score") is None else a.get("score"))}
                for i, a in enumerate(answers)
            ])
            st.dataframe(df, use_container_width=True)

    per_summ = st.session_state.summaries
    per_raw  = st.session_state.answers_map
    row = build_row(ts, pid, st.session_state.preset_name, per_summ, per_raw)
    row.update({
        "name": st.session_state.participant_name,
        "birth": st.session_state.participant_birth or "",
        "sex": st.session_state.participant_sex or "",
        "notes": st.session_state.participant_notes or "",
    })

    df_out = pd.DataFrame([row])
    
    drop_cols = [c for c in df_out.columns if c.endswith("_max")]
    if drop_cols:
        df_out = df_out.drop(columns=drop_cols, errors="ignore")
    
    buf = StringIO(); df_out.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button("📥 통합 CSV 다운로드", data=buf.getvalue().encode("utf-8-sig"),
                       file_name=f"{ts.replace(':','-')}_summary.csv", mime="text/csv")

    if gs_enable and gs_url:
        try:
            save_df_to_gsheet(df_out, gs_url, gs_ws)
            st.success("Google Sheets 저장 완료")
        except Exception as e:
            st.error(f"Google Sheets 저장 실패: {e}")

    st.divider()

    st.subheader("이상 응답 탐지 (규칙 기반·경량)")
    payload = make_payload(per_raw, per_summ)
    rulebook = load_rulebook(Path("rules/rulebook_v1.json"))
    flags = eval_rules(payload, rulebook)

    if not flags:
        st.success("모순 신호가 없습니다.")
        row["is_consistent"] = True; row["flags_json"] = "[]"
    else:
        for f in flags:
            st.warning(f"**{f['id']}** · {f['reason']}  \n제안: {', '.join(f.get('suggestion', []))}")
        row["is_consistent"] = False; row["flags_json"] = json.dumps(flags, ensure_ascii=False)

    st.divider()

    # ── LLM 기반 모순 가능성 요약 (Secrets 키만 사용)
    st.subheader("LLM 기반 이상응답 추론 (모순 가능성 제시)")
    llm_on = st.checkbox("LLM 사용", value=False)
    llm_model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=not llm_on)

    if llm_on and st.button("LLM으로 모순 가능성 분석"):
        key = get_secret_openai_key()
        if not key:
            st.info("🔑 Secrets에 openai_api_key가 없습니다. App Settings → Secrets에 등록하세요.")
        else:
            ai = run_llm_inference(per_survey_raw=per_raw, payload=payload, model=llm_model, api_key=key)
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

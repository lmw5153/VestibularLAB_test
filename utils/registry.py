# utils/registry.py
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ 설문 목록/로드 (YAML 유지, 중복 키 제거, 친절한 오류)                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

# YAML 강제 사용 (요청사항 반영)
try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError(
        "PyYAML이 설치되어 있지 않습니다. "
        "requirements.txt에 'PyYAML' 추가 후 재배포하거나, 로컬/서버에서 'pip install PyYAML'을 실행하세요."
    ) from e

# Streamlit 경고/오류 출력 보조(없으면 프린트)
try:
    import streamlit as st
except Exception:
    st = None

SURVEYS_DIR = Path("surveys")


def _warn(msg: str) -> None:
    if st is not None:
        st.warning(msg)
    else:
        print("[WARN]", msg)


def _error(msg: str) -> None:
    if st is not None:
        st.error(msg)
    else:
        print("[ERROR]", msg)


def _safe_load_yaml(p: Path) -> Dict[str, Any]:
    try:
        with p.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)  # type: ignore
        if not isinstance(doc, dict):
            raise ValueError("YAML 루트가 dict가 아닙니다.")
        return doc
    except Exception as e:
        _error(f"YAML 파싱 실패: {p.name} → {e}")
        raise


def _infer_meta(doc: Dict[str, Any], fallback_key: str) -> Dict[str, Any]:
    return {
        "key": doc.get("key", fallback_key),
        "title": doc.get("title", fallback_key),
        "input_type": doc.get("input_type", "radio"),
        "domains": doc.get("domains", {}),
    }


def list_surveys() -> List[Dict[str, Any]]:
    """
    surveys/*.yaml, *.yml 스캔 → (key,title,...) 목록
    - 동일 key가 여러 파일에 존재하면 **첫 번째만**採用
    """
    if not SURVEYS_DIR.exists():
        _warn(f"{SURVEYS_DIR} 폴더가 없습니다. 설문 파일을 추가하세요.")
        return []

    files = sorted(SURVEYS_DIR.glob("*.yaml")) + sorted(SURVEYS_DIR.glob("*.yml"))
    if not files:
        _warn(f"{SURVEYS_DIR}에 YAML 설문 파일이 없습니다(.yaml/.yml).")
        return []

    metas: List[Dict[str, Any]] = []
    seen_keys = set()

    for p in files:
        try:
            doc = _safe_load_yaml(p)
            meta = _infer_meta(doc, p.stem)
            k = meta["key"]
            if k in seen_keys:
                # 동일 key 중복 방지 (깜빡임/재등장 방지)
                continue
            seen_keys.add(k)
            metas.append(meta)
        except Exception:
            # 이미 _error 출력됨. 계속 다음 파일 시도.
            continue

    if not metas:
        _error("읽을 수 있는 YAML 설문이 없습니다. 파일 내용/형식을 확인하세요.")
    return metas


def load_survey(key: str) -> Dict[str, Any]:
    """
    주어진 key의 설문 YAML 로드.
    우선순위: surveys/{key}.yaml → {key}.yml
    """
    candidates = [
        SURVEYS_DIR / f"{key}.yaml",
        SURVEYS_DIR / f"{key}.yml",
    ]
    for p in candidates:
        if p.exists():
            return _safe_load_yaml(p)

    _error(f"설문 파일을 찾지 못했습니다: {key} (surveys/{key}.yaml|yml)")
    raise FileNotFoundError(f"No survey file for key={key}")

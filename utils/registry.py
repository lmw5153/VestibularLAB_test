# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ utils/registry.py — 설문 목록/로드 (YAML optional, JSON 기본 지원)     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json

# YAML 의존성은 선택(Optional)
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# Streamlit 경고/오류 표시 (없으면 프린트)
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


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(p: Path) -> Dict[str, Any]:
    if not HAVE_YAML:
        raise RuntimeError(
            "YAML 파일을 읽으려면 PyYAML이 필요합니다. "
            "① requirements.txt에 PyYAML 추가 후 배포하거나 ② 같은 내용을 .json으로 저장하세요."
        )
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore


def _infer_meta(doc: Dict[str, Any], fallback_key: str) -> Dict[str, Any]:
    return {
        "key": doc.get("key", fallback_key),
        "title": doc.get("title", fallback_key),
        "input_type": doc.get("input_type", "radio"),
        "domains": doc.get("domains", {}),
    }


def list_surveys() -> List[Dict[str, Any]]:
    """
    surveys/ 폴더의 설문 메타를 나열.
    - JSON(.json) 우선
    - YAML(.yaml/.yml)은 PyYAML 있을 때만 읽음
    """
    if not SURVEYS_DIR.exists():
        _warn(f"{SURVEYS_DIR} 폴더가 없습니다. 설문 파일을 추가하세요.")
        return []

    files_json = sorted(SURVEYS_DIR.glob("*.json"))
    files_yaml = sorted(SURVEYS_DIR.glob("*.yaml")) + sorted(SURVEYS_DIR.glob("*.yml"))
    metas: List[Dict[str, Any]] = []

    # 1) JSON 먼저
    for p in files_json:
        try:
            doc = _load_json(p)
            metas.append(_infer_meta(doc or {}, p.stem))
        except Exception as e:
            _warn(f"설문 메타(JSON) 읽기 실패: {p.name} → {e}")

    # 2) YAML (가능한 경우에만)
    if files_yaml and not HAVE_YAML:
        _warn(
            "YAML 설문이 있으나 PyYAML이 없어 목록에 포함하지 못했습니다. "
            "requirements.txt에 PyYAML 추가 또는 JSON으로 변환하세요."
        )
    else:
        for p in files_yaml:
            try:
                doc = _load_yaml(p)
                metas.append(_infer_meta(doc or {}, p.stem))
            except Exception as e:
                _warn(f"설문 메타(YAML) 읽기 실패: {p.name} → {e}")

    if not metas and (files_yaml and not HAVE_YAML):
        _error("설문 파일이 YAML뿐인데 PyYAML이 없습니다. JSON으로 변환하거나 PyYAML을 설치하세요.")
    elif not metas:
        _warn("surveys/ 폴더에 읽을 수 있는 설문 파일(.json 또는 .yaml/.yml)이 없습니다.")

    return metas


def load_survey(key: str) -> Dict[str, Any]:
    """
    key에 해당하는 설문 원문 로드.
    우선순위: surveys/{key}.json → {key}.yaml → {key}.yml
    """
    candidates = [
        SURVEYS_DIR / f"{key}.json",
        SURVEYS_DIR / f"{key}.yaml",
        SURVEYS_DIR / f"{key}.yml",
    ]
    for p in candidates:
        if p.exists():
            try:
                if p.suffix.lower() == ".json":
                    return _load_json(p)
                else:
                    return _load_yaml(p)  # PyYAML 없으면 친절한 에러
            except Exception as e:
                _error(f"설문 로드 실패: {p.name} → {e}")
                raise

    _error(f"설문 파일을 찾지 못했습니다: {key} (surveys/{key}.json|yaml|yml)")
    raise FileNotFoundError(f"No survey file for key={key}")

    _error(f"설문 파일을 찾지 못했습니다: {key} (surveys/{key}.yaml|yml|json)")
    raise FileNotFoundError(f"No survey file for key={key}")

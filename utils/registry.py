# utils/registry.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# YAML 의존성은 선택(Optional)
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# Streamlit은 선택 임포트(레지스트리가 단독 실행될 수도 있어서)
try:
    import streamlit as st
except Exception:
    st = None  # 타입만 맞춰놓기


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


def _load_yaml(p: Path) -> Dict[str, Any]:
    """YAML 파일 로드 (PyYAML 필요). 없으면 친절한 가이드 출력."""
    if not HAVE_YAML:
        _error(
            "이 설문은 YAML 형식입니다만 PyYAML이 설치되어 있지 않습니다.\n"
            "설치 방법: `pip install PyYAML` 혹은 requirements.txt에 `PyYAML` 추가 후 재배포하세요."
        )
        raise RuntimeError("PyYAML 미설치")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_meta_from_doc(doc: Dict[str, Any], fallback_key: str) -> Dict[str, Any]:
    """설문 문서에서 리스트 카드용 메타 정보(키/제목) 추출."""
    return {
        "key": doc.get("key", fallback_key),
        "title": doc.get("title", fallback_key),
    }


def list_surveys() -> List[Dict[str, Any]]:
    """
    surveys/ 폴더의 설문 메타를 나열.
    - 지원 확장자: .yaml, .yml, .json
    """
    if not SURVEYS_DIR.exists():
        _warn(f"{SURVEYS_DIR} 폴더가 없습니다. 설문 파일을 추가하세요.")
        return []

    metas: List[Dict[str, Any]] = []
    files = list(SURVEYS_DIR.glob("*.yaml")) + list(SURVEYS_DIR.glob("*.yml")) + list(SURVEYS_DIR.glob("*.json"))
    files.sort()

    if not files:
        _warn(f"{SURVEYS_DIR} 폴더에 설문 파일(.yaml/.yml/.json)이 없습니다.")
        return []

    for p in files:
        try:
            if p.suffix.lower() in [".yaml", ".yml"]:
                doc = _load_yaml(p)
            else:
                doc = _load_json(p)
            meta = _infer_meta_from_doc(doc or {}, p.stem)
            metas.append(meta)
        except Exception as e:
            _warn(f"설문 메타 읽기 실패: {p.name} → {e}")

    return meta


def load_survey(key: str) -> Dict[str, Any]:
    """
    key에 해당하는 설문 원문 로드.
    파일 탐색 우선순위:
      surveys/{key}.yaml → {key}.yml → {key}.json
    """
    candidates = [
        SURVEYS_DIR / f"{key}.yaml",
        SURVEYS_DIR / f"{key}.yml",
        SURVEYS_DIR / f"{key}.json",
    ]

    for p in candidates:
        if p.exists():
            try:
                if p.suffix.lower() in [".yaml", ".yml"]:
                    return _load_yaml(p)
                else:
                    return _load_json(p)
            except Exception as e:
                _error(f"설문 로드 실패: {p.name} → {e}")
                raise

    _error(f"설문 파일을 찾지 못했습니다: {key} (surveys/{key}.yaml|yml|json)")
    raise FileNotFoundError(f"No survey file for key={key}")

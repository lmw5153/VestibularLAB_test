# scoring/hads.py
from typing import List, Dict, Any

class HADSScorer:
    """Hospital Anxiety and Depression Scale (HADS)

    - HADS-A: 1,3,5,7,9,11,13 (0-21)
    - HADS-D: 2,4,6,8,10,12,14 (0-21)

    본 프로젝트에서는 YAML 각 문항의 choices에 이미 0~3 점수가 포함되어 있다고 가정합니다.
    """

    def _band(self, v: int) -> str:
        # 전통적 분류: 0-7 정상, 8-10 경계, 11-21 의심/비정상
        if v <= 7:
            return "정상(0-7)"
        if v <= 10:
            return "경계(8-10)"
        return "의심(11-21)"

    def score(self, answers: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        domains = {"A": 0, "D": 0}

        for a in answers:
            dom = a.get("domain", "")
            s = a.get("score", None)
            if s is None:
                continue
            try:
                s = int(s)
            except Exception:
                continue
            if dom in domains:
                domains[dom] += s

        total = domains["A"] + domains["D"]

        out = {"total": total, "max": meta.get("max_score", 42), "domains": domains}
        out["severity"] = f"불안(A) {self._band(domains['A'])}, 우울(D) {self._band(domains['D'])}"
        return out

# scoring/kgds.py
from typing import List, Dict, Any

class KGDSScorer:
    """K-GDS (Korean Form of Geriatric Depression Scale, 30 items)

    본 프로젝트에서는 YAML 각 문항의 choices에 (우울 반응=1, 비우울 반응=0) 형태로 점수가
    이미 반영되어 있다고 가정합니다.
    """

    def score(self, answers: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        total = 0
        for a in answers:
            s = a.get("score", None)
            if s is None:
                continue
            try:
                total += int(s)
            except Exception:
                pass

        out = {"total": total, "max": meta.get("max_score", 30), "domains": {}}

        # 참고용 절단점(문헌에서 흔히 보고되는 cut-off 16 기반)
        cutoff = meta.get("cutoff", 16)
        out["severity"] = "우울 의심(≥{0})".format(cutoff) if total >= cutoff else "정상 범위(<{0})".format(cutoff)
        return out

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# scoring/phq9.py
from typing import List, Dict, Any

class PHQ9Scorer:
    def score(self, answers: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        total = 0
        for a in answers:
            s = a.get("score", 0)
            try:
                total += int(s)
            except Exception:
                pass

        # 선택: severity 라벨 추출
        sev = None
        for band in meta.get("severity_thresholds", []):
            lo, hi = band.get("range", [None, None])
            if lo is not None and hi is not None and lo <= total <= hi:
                sev = band.get("label")
                break

        out = {"total": total, "max": meta.get("max_score", 27), "domains": {}}
        if sev:
            out["severity"] = sev
        return out


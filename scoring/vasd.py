#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# scoring/vasd.py
from typing import List, Dict, Any

class VASDScorer:
    def score(self, answers: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        VAS-D: 단일 문항(0~10) 값 자체가 점수.
        """
        val = 0
        for a in answers:
            v = a.get("score")
            if isinstance(v, int):
                val = v
                break
        return {"total": val, "max": meta.get("max_score", 10), "domains": {}}


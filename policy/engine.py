from __future__ import annotations
import yaml
import pandas as pd

OPS = {
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

def load_policy(path: str = "policy/policy_table.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _eval_clause(row: pd.Series, clause: dict, thresholds: dict) -> bool:
    feat = clause.get("feature")
    op = clause.get("op")
    if "threshold_key" in clause:
        b = float(thresholds[clause["threshold_key"]])
    else:
        b = float(clause.get("value"))
    a = float(row.get(feat, 0.0))
    return OPS[op](a, b)

def apply_policy(feats: pd.DataFrame, policy: dict) -> pd.DataFrame:
    thresholds = policy.get("thresholds", {})
    rules = policy.get("rules", [])

                                 
    rule_fired = {r["id"]: [False]*len(feats) for r in rules}

    actions_out = []
    explain_out = []

                                                                              
                                           
    for pass_id in [1, 2]:
        for ri, r in enumerate(rules):
            rid = r["id"]
            cond = r.get("if", {})
            has_dep = "rule" in str(cond)                
            if pass_id == 1 and has_dep:
                continue
            if pass_id == 2 and not has_dep:
                continue

            fired_list = []
            for i, row in feats.iterrows():
                fired = True
                if "any" in cond:
                    fired = any(_eval_clause(row, c, thresholds) for c in cond["any"])
                elif "all" in cond:
                    fired = all(
                        (_eval_clause(row, c, thresholds) if "feature" in c else rule_fired[c["rule"]][i])
                        for c in cond["all"]
                    )
                else:
                    fired = False
                fired_list.append(bool(fired))
            rule_fired[rid] = fired_list

                                                      
    rules_sorted = sorted(rules, key=lambda r: int(r.get("priority", 0)))
    for i, row in feats.iterrows():
        fired_rules = [r for r in rules_sorted if rule_fired[r["id"]][i]]
                       
        acts = []
        expl = []
        for r in fired_rules:
            acts.extend(list(r.get("then", [])))
            expl.append({"rule": r["id"], "why": r.get("explain", "")})
                                   
        seen = set()
        acts_u = []
        for a in acts:
            if a not in seen:
                seen.add(a)
                acts_u.append(a)

        actions_out.append(acts_u)
        explain_out.append(expl)

    out = feats.copy()
    out["policy_actions"] = actions_out
    out["policy_explain"] = explain_out
    out["policy_fired"] = [len(a) > 0 for a in actions_out]
    return out

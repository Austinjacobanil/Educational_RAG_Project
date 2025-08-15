ALPHA = 0.35

def init_mastery(comps):
    return {c: 0.2 for c in comps}

def update_mastery(mastery, comp_scores):
    for c, s in comp_scores.items():
        prev = mastery.get(c, 0.2)
        mastery[c] = (1-ALPHA)*prev + ALPHA*(1 if s else 0)
    return mastery

def gaps(mastery, threshold=0.6):
    return [c for c, m in mastery.items() if m < threshold]
import networkx as nx

COMPETENCIES = {
    "time_complexity": {"name": "Time Complexity", "prereq": []},
    "asymptotics": {"name": "Asymptotics", "prereq": ["time_complexity"]},
    "software_engineering": {"name": "Software Engineering", "prereq": ["asymptotics"]},
    "recurrence": {"name": "Recurrences", "prereq": ["software_engineering"]},
    "master_theorem": {"name": "Master Theorem", "prereq": ["recurrence"]},
}

def build_graph():
    g = nx.DiGraph()
    for k, v in COMPETENCIES.items():
        g.add_node(k)
        for p in v["prereq"]:
            g.add_edge(p, k)
    return g

G = build_graph()
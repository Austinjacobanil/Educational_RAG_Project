import os
import time
import json
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from competencies import COMPETENCIES
from progress import init_mastery, update_mastery, gaps

import os
if not os.path.exists(".chroma"):
    import ingest
    ingest.run(catalog_path="sample_data/catalog.csv", persist_directory=".chroma")

#  Config 
load_dotenv()
PROFILE_PATH = Path("student_profiles.json")

USE_GROQ = bool(os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

if USE_GROQ:
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#  Global Resources 
EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
VS = Chroma(persist_directory=".chroma", embedding_function=EMB)
ALL_COMPS = list(COMPETENCIES.keys())

STYLE_TAG_MAP = {
    "visual": {"visual", "video", "diagram"},
    "auditory": {"audio", "podcast", "lecture"},
    "kinesthetic": {"hands_on", "exercise", "interactive"},
}

#  Helpers 
def get_llm_client():
    if USE_GROQ:
        return groq_client
    raise RuntimeError("No LLM client available — please set GROQ_API_KEY")

# Profile Persistence 
def load_profile():
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text())
        except:
            pass
    return {
        "name": "Student",
        "level": "beginner",
        "learning_style": "visual",
        "completed_topics": [],
        "mastery": init_mastery(ALL_COMPS),
    }

def save_profile(profile):
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))

#  Retrieval 
def personalized_retrieve(query: str, profile: dict, k: int = 6, gaps_only=False):
    t0 = time.time()

    if gaps_only:
        gap_comps = [c for c in profile["mastery"] if profile["mastery"][c] < 0.6]
        query = " ".join(gap_comps) if gap_comps else query

    docs = VS.similarity_search(query, k=k * 2)
    preferred = STYLE_TAG_MAP.get(profile.get("learning_style", "visual"), set())

    def score(d):
        tags = set((d.metadata.get("style_tags") or "").split(";"))
        diff = int(d.metadata.get("difficulty", 3))
        comp_list = (d.metadata.get("competencies") or "").split(";")

        if any(profile["mastery"].get(c, 0) < 0.4 for c in comp_list):
            diff_pref = abs(diff - 1) 
        elif any(profile["mastery"].get(c, 0) > 0.8 for c in comp_list):
            diff_pref = abs(diff - 5) 
        else:
            diff_pref = abs(diff - 3) 

        style_boost = 1 + 0.2 * len(preferred.intersection(tags))
        completed_penalty = 1.2 if any(c in profile.get("completed_topics", []) for c in comp_list) else 1.0
        return (diff_pref, completed_penalty), style_boost

    docs_scored = sorted(docs, key=lambda d: (score(d)[0], score(d)[1]))
    return docs_scored[:k], time.time() - t0

# Path Generation 
def generate_path(query, name, level, style, completed, gaps_only=False):
    profile = {
        "name": name,
        "level": level,
        "learning_style": style,
        "completed_topics": [t.strip() for t in completed.split(",") if t.strip()],
        "mastery": load_profile()["mastery"]
    }

    if gaps_only:
        retrieved_docs, _ = personalized_retrieve(query, profile, gaps_only=True)
    else:
        retrieved_docs, _ = personalized_retrieve(query, profile)

    topics = [d.page_content for d in retrieved_docs]
    objectives = query

    client = get_llm_client()
    prompt = f"""You are an educational path planner.
    Learner profile: {profile}
    Topics: {topics}
    Learning objectives: {objectives}
    Generate a step-by-step personalized learning path."""

    try:
        out = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        plan = out.choices[0].message.content.strip()
    except Exception as e:
        plan = f"Error generating plan: {e}"

    return plan

# Progress Update 
def update_progress(topic: str, correct: bool):
    profile = load_profile()
    profile["mastery"] = update_mastery(profile["mastery"], {topic: correct})
    save_profile(profile)
    return json.dumps(profile["mastery"], indent=2), ", ".join(gaps(profile["mastery"]))

# UI 
with gr.Blocks() as demo:
    gr.Markdown("# 🎓 EduRAG — Personalized Learning Path")

    with gr.Row():
        name = gr.Textbox(label="Name", value=load_profile()["name"])
        level = gr.Radio(["beginner", "intermediate", "advanced"], label="Level", value=load_profile()["level"])
        style = gr.Radio(["visual", "auditory", "kinesthetic"], label="Learning Style", value=load_profile()["learning_style"])

    completed = gr.Textbox(label="Completed Topics (comma-separated)", value=", ".join(load_profile()["completed_topics"]))
    query = gr.Textbox(label="What do you want to learn?", value="Master the Master Theorem")

    with gr.Row():
        gen_btn = gr.Button("Generate Learning Path")
        gap_btn = gr.Button("Remediate Knowledge Gaps")

    output = gr.Markdown()
    gen_btn.click(fn=generate_path, inputs=[query, name, level, style, completed], outputs=output)
    gap_btn.click(fn=lambda q, n, l, s, c: generate_path(q, n, l, s, c, gaps_only=True),
                  inputs=[query, name, level, style, completed], outputs=output)

    gr.Markdown("## 📈 Progress Tracker")
    topic = gr.Dropdown(choices=ALL_COMPS, label="Competency")
    correct = gr.Checkbox(label="Answered last quiz correctly?", value=True)
    upd_btn = gr.Button("Update Mastery")
    mastery_out = gr.Code(label="New Mastery JSON")
    gaps_out = gr.Textbox(label="Current Gaps")
    upd_btn.click(fn=update_progress, inputs=[topic, correct], outputs=[mastery_out, gaps_out])

if __name__ == "__main__":
    demo.launch()

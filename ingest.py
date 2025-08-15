from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_row(row):
    path = str(row["path"]).strip()
    modality = str(row["modality"]).strip().lower()

    if modality == "pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
    elif modality in ("text", "txt"):
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
    elif modality == "video":
        transcript_path = path.replace(".mp4", ".txt")  
        loader = TextLoader(transcript_path, encoding="utf-8")
        docs = loader.load()
    elif modality == "audio":
        transcript_path = path.replace(".mp3", ".txt") 
        loader = TextLoader(transcript_path, encoding="utf-8")
        docs = loader.load()
    elif modality == "image":
        caption_path = path.replace(".jpg", ".txt") 
        loader = TextLoader(caption_path, encoding="utf-8")
        docs = loader.load()
    else:
        raise ValueError(f"Unknown modality: {modality}")

    for d in docs:
        d.metadata.update({
            "doc_id": row["id"],
            "title": row["title"],
            "modality": modality,
            "competencies": row["competencies"],
            "difficulty": int(row["difficulty"]),
            "style_tags": row.get("style_tags", ""),
            "learning_objectives": row.get("learning_objectives", "")
        })
    return docs

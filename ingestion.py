# ingestion.py
import os
import time
import hashlib
import json
from pymongo import MongoClient, errors, ASCENDING
from bs4 import BeautifulSoup
from llama_cloud_services import LlamaParse
import pandas as pd

# Optional: import your PDF parser here (keep your existing usage)
# from llamaparse import LlamaParse
# If LlamaParse is a CLI you may need subprocess instead.

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "pdf_cache_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "parsed_files")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coll = db[COLLECTION_NAME]

# ensure unique index on file_hash to avoid duplicates
coll.create_index([("file_hash", ASCENDING)], unique=True)


# ----------------------
# Helpers: hashing & db
# ----------------------
def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cached_parse(file_hash: str):
    """Return cached doc or None."""
    return coll.find_one({"file_hash": file_hash})


def upsert_parsed_doc(file_path: str, file_hash: str, md: str, meta: dict = None):
    """Insert document into Mongo; handle race condition if duplicate inserted concurrently."""
    stat = os.stat(file_path)
    doc = {
        "file_hash": file_hash,
        "file_path": os.path.abspath(file_path),
        "file_size": stat.st_size,
        "file_mtime": stat.st_mtime,
        "md": md,
        "meta": meta or {},
        "cached_at": time.time(),
    }
    try:
        coll.insert_one(doc)
        return doc
    except errors.DuplicateKeyError:
        # Race - another process inserted the same hash. Return what is in DB.
        cached = get_cached_parse(file_hash)
        return cached if cached else doc


# ----------------------
# Parsers (file types)
# ----------------------
def parse_pdf_document(pdf_path: str, use_cache: bool = True):
    """
    Parse a PDF into markdown using LlamaParse (or your PDF parser).
    Returns (markdown_text, metadata_dict)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    file_hash = sha256_of_file(pdf_path)

    if use_cache:
        cached = get_cached_parse(file_hash)
        if cached:
            return cached["md"], {"cached": True, "file_hash": file_hash, "cached_at": cached.get("cached_at")}

    # ---------- run parser ----------
    # Keep your existing LlamaParse usage; adapt API if necessary
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        num_workers=4,
        verbose=False,
        language="en",
        result_type="json",
    )

    result_raw = parser.parse(pdf_path)  # same as your snippet
    # join page markdowns; if your API returns different structure adapt this
    result_md = "\n\n".join([getattr(page, "md", str(page)) for page in result_raw.pages])

    stored = upsert_parsed_doc(pdf_path, file_hash, result_md, meta={"type": "pdf", "pages": len(result_raw.pages)})
    return stored["md"], {"cached": False, "file_hash": file_hash, "stored_at": stored.get("cached_at")}


def parse_csv_document(csv_path: str, use_cache: bool = True, max_rows_preview: int = 500):
    """
    Parse CSV -> markdown preview (and JSON if needed).
    Returns (markdown_text, metadata_dict)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    file_hash = sha256_of_file(csv_path)
    if use_cache:
        cached = get_cached_parse(file_hash)
        if cached:
            return cached["md"], {"cached": True, "file_hash": file_hash, "cached_at": cached.get("cached_at")}

    # read CSV (pandas handles many edge cases). Limit memory usage by reading in chunks if necessary.
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        # fallback: small robust reader
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(100_000)
        md = f"```\n{text}\n```"
        stored = upsert_parsed_doc(csv_path, file_hash, md, meta={"type": "csv", "error": str(e)})
        return stored["md"], {"cached": False, "file_hash": file_hash}

    # create reasonable markdown: schema + preview rows
    schema_lines = [f"- **{col}**: {str(dtype)}" for col, dtype in zip(df.columns, df.dtypes)]
    preview = df.head(max_rows_preview)
    md = f"### Schema\n\n" + "\n".join(schema_lines) + "\n\n### Preview (first rows)\n\n"
    # pandas -> markdown (if large tables, this string can be long; it's OK for caching)
    try:
        md += preview.to_markdown(index=False)
    except Exception:
        md += preview.to_json(orient="records", indent=2)

    stored = upsert_parsed_doc(csv_path, file_hash, md, meta={"type": "csv", "rows": int(df.shape[0]), "cols": int(df.shape[1])})
    return stored["md"], {"cached": False, "file_hash": file_hash, "rows": int(df.shape[0])}


def parse_json_document(json_path: str, use_cache: bool = True, max_chars: int = 200_000):
    """
    Parse JSON -> pretty-printed markdown block. For very large JSONs we trim/preview.
    Returns (markdown_text, metadata_dict)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    file_hash = sha256_of_file(json_path)
    if use_cache:
        cached = get_cached_parse(file_hash)
        if cached:
            return cached["md"], {"cached": True, "file_hash": file_hash, "cached_at": cached.get("cached_at")}

    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            payload = json.load(f)
            pretty = json.dumps(payload, indent=2)
        except Exception as e:
            # if JSON is malformed, fallback to raw snippet
            f.seek(0)
            pretty = f.read(max_chars)
            pretty = pretty[:max_chars] + ("\n...truncated" if len(pretty) == max_chars else "")

    md = "```json\n" + pretty + "\n```"
    stored = upsert_parsed_doc(json_path, file_hash, md, meta={"type": "json"})
    return stored["md"], {"cached": False, "file_hash": file_hash}


def parse_html_document(html_path: str, use_cache: bool = True, max_chars: int = 200_000):
    """
    Parse HTML -> extracted visible text (markdown-like). Returns (markdown_text, metadata_dict)
    """
    if not os.path.exists(html_path):
        raise FileNotFoundError(html_path)

    file_hash = sha256_of_file(html_path)
    if use_cache:
        cached = get_cached_parse(file_hash)
        if cached:
            return cached["md"], {"cached": True, "file_hash": file_hash, "cached_at": cached.get("cached_at")}

    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read(max_chars)

    soup = BeautifulSoup(raw, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    text = soup.get_text("\n")
    # collapse multiple newlines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text_clean = "\n\n".join(lines)
    md = text_clean if len(text_clean) <= max_chars else text_clean[:max_chars] + "\n\n...truncated"

    stored = upsert_parsed_doc(html_path, file_hash, md, meta={"type": "html"})
    return stored["md"], {"cached": False, "file_hash": file_hash}


# ----------------------
# Example convenience wrapper
# ----------------------
def parse_file_by_type(path: str, use_cache: bool = True):
    """Detect type by extension and call appropriate parser. Returns (md, meta)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pdf"}:
        return parse_pdf_document(path, use_cache=use_cache)
    if ext in {".csv"}:
        return parse_csv_document(path, use_cache=use_cache)
    if ext in {".json"}:
        return parse_json_document(path, use_cache=use_cache)
    if ext in {".html", ".htm"}:
        return parse_html_document(path, use_cache=use_cache)
    # txt or fallback
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(200_000)
    file_hash = sha256_of_file(path)
    stored = upsert_parsed_doc(path, file_hash, text, meta={"type": "txt_or_other"})
    return stored["md"], {"cached": False, "file_hash": file_hash}
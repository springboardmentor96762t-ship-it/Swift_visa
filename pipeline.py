# Persistence helpers
# ----------------------------
def save_meta_jsonl(meta_list, out_path):
    with open(out_path, "w", encoding="utf8") as f:
        for m in meta_list:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def load_meta_jsonl(path):
    out = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

 # Main pipeline orchestration
# ----------------------------
def process_pdfs(data_dir, out_dir, model_name=MODEL_NAME, chunk_words=CHUNK_MAX_WORDS, stride=CHUNK_STRIDE_WORDS):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load models
    print("Loading NLP model (spaCy)...")
    nlp = spacy.load("en_core_web_sm", disable=["ner","tok2vec"])
    print("Loading embedding model:", model_name)
    embed_model = SentenceTransformer(model_name)

    all_chunks = []
    all_meta = []
    # Expect PDFs organized like data/{country}/{visa_type}/*.pdf OR data/*.pdf
    pdf_paths = list(data_dir.rglob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDF(s).")

    for pdf_path in tqdm(pdf_paths, desc="PDFs"):
        # infer metadata from path if possible
        parts = pdf_path.parts
        country = None
        visa_type = None
        # heuristic: if structure has .../country/visa/file.pdf
        if len(parts) >= 3:
            # e.g. data/Canada/Study/file.pdf
            try:
                idx = parts.index(data_dir.name)
                if idx + 1 < len(parts):
                    country = parts[idx+1]
                if idx + 2 < len(parts):
                    visa_type = parts[idx+2]
            except ValueError:
                pass

        pages = extract_text_from_pdf(pdf_path)
        pages = remove_repeated_headers_footers(pages)
        sent_objs = sentence_split_pages(pages, nlp)
        chunks = chunk_sentences(sent_objs, max_words=chunk_words, stride_words=stride)
        # attach metadata & global id
        base_name = pdf_path.name
        for i,c in enumerate(chunks):
            meta = {
                "source_file": str(pdf_path),
                "file_name": base_name,
                "country": country or "UNKNOWN",
                "visa_type": visa_type or "UNKNOWN",
                "chunk_id": len(all_meta),
                "pages": c.get("pages", []),
                "text": c["text"]
            }
            all_meta.append(meta)
            all_chunks.append(c["text"])

    if not all_chunks:
        print("No text/chunks produced. Exiting.")
        return

    print(f"Total chunks: {len(all_chunks)}. Computing embeddings...")
    embeddings = embed_texts(all_chunks, embed_model, batch_size=BATCH_SIZE)

     # Save embeddings array for backup
    np.save(out_dir / "embeddings.npy", embeddings)
    save_meta_jsonl(all_meta, out_dir / "chunks_meta.jsonl")

    # Build FAISS
    print("Building FAISS index (IVF+PQ). If dataset small, pass use_ivfpq=False to use HNSWFlat")
    index = build_faiss_index(embeddings, nlist=FAISS_NLIST, use_ivfpq=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    print("Saved faiss.index and metadata.")
    return str(out_dir / "faiss.index"), str(out_dir / "chunks_meta.jsonl"), embed_model

# Command-line entrypoint
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to folder containing PDF(s).")
    parser.add_argument("--out_dir", required=True, help="Where to write embeddings/index/meta.")
    args = parser.parse_args()

    faiss_idx_path, meta_path, embed_model = process_pdfs(args.data_dir, args.out_dir)

    # Example query session
    print("\nExample query run (type 'quit' to stop):")
    index = faiss.read_index(faiss_idx_path)
    while True:
        q = input("Enter query (or 'quit'): ").strip()
        if not q or q.lower() == "quit":
            break
        country = input("Filter country (press enter to skip): ").strip() or None
        visa = input("Filter visa_type (press enter to skip): ").strip() or None
        res = query_index(q, embed_model, index, meta_path, top_k=5, filter_country=country, filter_visa=visa)
        ctx = build_context(res)
        print("\n--- Top results ---")
        for i,r in enumerate(res,1):
            print(f"{i}. file:{r['file_name']} country:{r.get('country')} visa:{r.get('visa_type')} pages:{r.get('pages')}")
            print(repr(r['text'][:400]) + ("..." if len(r['text'])>400 else ""))
            print()
        print("Assembled context (first 1200 chars):\n", ctx[:1200], "\n---\n")

if __name__ == "__main__":
    main()
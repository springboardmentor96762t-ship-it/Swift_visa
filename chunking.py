import json
from pathlib import Path

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Simple fixed-size text chunking.
    """
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def save_chunks(chunks, output_path="raw/chunks.json"):
    Path("raw").mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

import pandas as pd
import textwrap

WRAP_WIDTH = 110   # adjust width as per your screen

# ----------------------------
# PRETTY PRINT HELPERS
# ----------------------------
def print_block(title, content):
    print("\n" + "=" * 120)
    print(f"📌 {title}")
    print("=" * 120 + "\n")
    wrapped = textwrap.fill(content, WRAP_WIDTH)
    print(wrapped)
    print("\n" + "-" * 120 + "\n")


# ----------------------------
# VISUALIZE SENTENCES
# ----------------------------
def visualize_sentences(sentences, limit=10):
    print("\n" + "#" * 120)
    print("                     🔎 PREVIEW OF SPLIT SENTENCES")
    print("#" * 120 + "\n")

    for i, s in enumerate(sentences[:limit]):
        block_title = f"Sentence {i+1} (Page {s['page']})"
        print_block(block_title, s['sentence'])


# ----------------------------
# VISUALIZE CHUNKS (TEXT BLOCK VIEW)
# ----------------------------
def visualize_chunks(chunks, limit=3, show_full_text=True):
    print("\n" + "#" * 120)
    print("                     📚 PREVIEW OF CHUNKS")
    print("#" * 120 + "\n")

    for i, ch in enumerate(chunks[:limit]):
        header = (
            f"CHUNK {i+1}\n"
            f"Pages: {ch['pages']}\n"
            f"Sentence Index Range: {ch['start_idx']} → {ch['end_idx']}"
        )
        print("=" * 120)
        print(header)
        print("=" * 120)

        if show_full_text:
            text = ch["text"]
        else:
            text = ch["text"][:600] + "..."

        wrapped = textwrap.fill(text, WRAP_WIDTH)
        print(wrapped)
        print("\n" + "-" * 120 + "\n")


# ----------------------------
# DATAFRAME VISUALIZATION
# ----------------------------
def visualize_dataframe(sentences, chunks):
    df_sent = pd.DataFrame(sentences)
    df_chunks = pd.DataFrame(chunks)

    print("\n" + "#" * 120)
    print("                     🗂 DATAFRAMES PREVIEW")
    print("#" * 120)

    print("\nSentence DataFrame (df_sent):")
    print(df_sent.head())

    print("\nChunk DataFrame (df_chunks):")
    print(df_chunks.head())

    print("\n(Use df_sent and df_chunks for deeper analysis.)")

    return df_sent, df_chunks

# Sentence splitting & chunking
# ----------------------------
import pandas as pd


def sentence_split_pages(page_texts, nlp):
    """
    Return list of dicts: {'page':int, 'sentence':str}
    """
    out = []
    for p_no, txt in enumerate(page_texts):
        if not txt or not txt.strip():
            continue
        txt = dehyphenate(txt)
        txt = normalize_text(txt)
        doc = nlp(txt)
        for s in doc.sents:
            st = s.text.strip()
            if st:
                out.append({"page": p_no, "sentence": st})
    return out


def chunk_sentences(sentences, max_words=CHUNK_MAX_WORDS, stride_words=CHUNK_STRIDE_WORDS):
    chunks = []
    current_chunk = []
    current_words = 0
    i = 0

    def wcount(s): 
        return len(s.split())

    while i < len(sentences):
        s = sentences[i]['sentence']
        ws = wcount(s)

        if current_words + ws > max_words and current_chunk:
            chunk_text = " ".join([x['sentence'] for x in current_chunk])

            chunk_meta = {
                "text": chunk_text,
                "start_idx": i - len(current_chunk),
                "end_idx": i - 1,
                "pages": sorted(list({x['page'] for x in current_chunk}))
            }
            chunks.append(chunk_meta)

            # overlap logic
            overlap_words = 0
            j = len(current_chunk) - 1

            while j >= 0 and overlap_words < stride_words:
                overlap_words += wcount(current_chunk[j]['sentence'])
                j -= 1

            current_chunk = current_chunk[j + 1:]
            current_words = sum(wcount(x['sentence']) for x in current_chunk)

        else:
            current_chunk.append(sentences[i])
            current_words += ws
            i += 1

    # last chunk
    if current_chunk:
        chunk_text = " ".join([x['sentence'] for x in current_chunk])
        chunks.append({
            "text": chunk_text,
            "start_idx": len(sentences) - len(current_chunk),
            "end_idx": len(sentences) - 1,
            "pages": sorted(list({x['page'] for x in current_chunk}))
        })

    return chunks


# ----------------------------
# OUTPUT SECTION
# ----------------------------

def visualize_sentences(sentences, limit=10):
    print("\n===== FIRST SENTENCES PREVIEW =====\n")
    for i, s in enumerate(sentences[:limit]):
        print(f"{i+1}. (Page {s['page']}) → {s['sentence']}")


def visualize_chunks(chunks, limit=3):
    print("\n\n===== FIRST CHUNKS PREVIEW =====\n")
    for i, ch in enumerate(chunks[:limit]):
        print(f"\n----- CHUNK {i+1} -----")
        print(f"Pages: {ch['pages']}")
        print(f"Start Index: {ch['start_idx']}, End Index: {ch['end_idx']}")
        print(f"Text Preview:\n{ch['text'][:500]}...") 


def visualize_dataframe(sentences, chunks):
    df_sent = pd.DataFrame(sentences)
    df_chunks = pd.DataFrame(chunks)

    print("\nSentence DataFrame Ready (df_sent)")
    print("Chunk DataFrame Ready (df_chunks)")
    return df_sent, df_chunks
import os
import json

def chunk_json_files():
    # Use the actual folder name where your JSON files are
    input_folder = "C:\\Users\\akhil\\Downloads\\AKHIL MENON BATCH 7 VISA"
    
    # Create output folder
    output_folder = os.path.join(input_folder, "chunked_output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each JSON file
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".json", "_chunks.txt"))
            
            # Read JSON file
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Convert to string and split into chunks
            json_str = json.dumps(data, indent=2)
            chunks = []
            current_chunk = ""
            
            for line in json_str.split('\n'):
                if len(current_chunk + line) > 500:  # ~500 chars per chunk
                    chunks.append(current_chunk)
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Save chunks to file
            with open(output_path, 'w') as f:
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"--- CHUNK {i} ---\n")
                    f.write(chunk)
                    f.write("\n\n")
            
            print(f"Created {output_path} with {len(chunks)} chunks")

# Run the function
chunk_json_files()

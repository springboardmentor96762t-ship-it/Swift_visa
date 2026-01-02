import faiss

def search(query, index, chunks, model, k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    
    distances, indices = index.search(query_vec, k)
    
    results = []
    top_score = 0.0
    
    if len(distances[0]) > 0:
        top_score = float(distances[0][0])
        
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])
            
    return results, top_score
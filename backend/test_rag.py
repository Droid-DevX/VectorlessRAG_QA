from pathlib import Path
from backend.rag_engine import *

p = Path("Resume.pdf")
index = PageIndex()

pages = extract_pages_from_pdf(p.read_bytes())
count = index.add_document(pages, p.name)

print("=" * 80)
print("INDEXING TEST")
print("=" * 80)
print("Pages:", len(pages))
print("Chunks indexed:", count)
print("Stats:", index.stats())

queries = [
    "What projects have I worked on?",
    "What is my education?",
    "What are my technical skills?",
    "What publications have I worked on?",
]

for q in queries:
    print("\n" + "=" * 80)
    print("QUERY:", q)
    print("=" * 80)

    results = index.search(q, top_k=3)

    if not results:
        print("NO RESULTS")
        continue

    for n, r in enumerate(results, 1):
        print(
            f"Rank {n} | "
            f"Section: {r['section']} | "
            f"Page: {r['page']} | "
            f"Chunk: {r['chunk']} | "
            f"Score: {r['score']:.4f}"
        )

        print(r["text"][:500])
        print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
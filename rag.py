import re
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    """
    Simple TF-IDF RAG over markdown sections.
    Improvements:
      - Title is concatenated (and upweighted) to the body when indexing.
      - Bigram support for better matching.
      - Helper to fetch a section by exact title.
    """
    def __init__(self, kb_paths: List[str]):
        self.docs: List[str] = []
        self.titles: List[str] = []
        self.title_to_doc = {}

        for p in kb_paths:
            with open(p, 'r', encoding='utf-8') as f:
                txt = f.read()

            # split by headings like: "\n# Title"
            chunks = re.split(r"\n# ", txt)
            for ch in chunks:
                ch = ch.strip()
                if not ch:
                    continue
                if not ch.startswith('# '):
                    ch = '# ' + ch
                title = ch.splitlines()[0].lstrip('# ').strip()
                body = '\n'.join(ch.splitlines()[1:]).strip()

                # Upweight title in the document text for better matching
                # (repeat title tokens to bias similarity)
                doc_text = ((title + ' ') * 3).strip() + '\n' + body

                self.titles.append(title)
                self.docs.append(doc_text)
                self.title_to_doc[title] = body

        # Use bigrams for better phrase matching; remove English stop words
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.mat = self.vectorizer.fit_transform(self.docs)

    def retrieve(self, query: str, topk: int = 3) -> List[Tuple[str, str, float]]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.mat).ravel()
        idxs = sims.argsort()[::-1][:topk]
        out = []
        for i in idxs:
            title = self.titles[i]
            body = self.title_to_doc.get(title, "")
            out.append((title, body, float(sims[i])))
        return out

    def get_by_title(self, title: str) -> Optional[str]:
        """Return the body text for an exact section title, or None."""
        return self.title_to_doc.get(title)

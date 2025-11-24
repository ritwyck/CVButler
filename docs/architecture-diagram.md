# Architecture Diagram

Create a PNG diagram based on this description using tools like Draw.io, Lucidchart, or PowerPoint.

## Diagram Description

**Components:**

- Streamlit UI (Web Interface)
- File Uploaders (JD and CVs)
- File Extractors (PDF/Word readers)
- Text Cleaner and Processor
- SBERT Embeddings Engine
- Similarity Calculator (Cosine)
- Explanations Generator
- Results Display

**Flow:**

1. User uploads files via Streamlit UI
2. Files are processed by extractors to get raw text
3. Text is cleaned and preprocessed
4. Clean text is encoded using Sentence-BERT
5. Similarities are computed between JD and each CV
6. Explanations are generated using matched skills and experience
7. Top 3 results are ranked and displayed with scores and explanations

**Visual Layout:**

- Horizontal flow from left to right
- Data flows downwards or rightwards
- Group related components (e.g., text processing in one box)
- Show icons for technologies (Python, Streamlit, SBERT)

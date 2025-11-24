# Demo Video Script

## Video Title: CVButler Demo - AI-Powered Candidate Matching in 2 Minutes

## Script (2-5 minutes total)

### Opening (0:00 - 0:20)

- Hi everyone! Today I'm demonstrating CVButler, an AI-powered web app that helps recruiters quickly match job descriptions with the best candidate CVs.
- Built with Python, Streamlit, and Sentence-BERT for semantic matching.

### Setup & Upload (0:20 - 1:00)

- First, I upload a job description for a Senior Python Developer. It highlights required skills like Python, Django, SQL, and ML experience.
- Then I upload three candidate CVs - John Doe, Jane Smith, and Bob Johnson - each with different experience levels and skill sets.

### Processing & Results (1:00 - 1:40)

- Click "Match Candidates" - the app processes the files, extracts text, generates AI embeddings, and computes similarities.
- Results are ranked by semantic similarity score.
- John Doe ranks first - high match for Python, Django, and 5 years experience.
- Bob Johnson second - strong data analysis skills matching the job.
- Jane Smith third - more web-focused, still relevant but lower overall match.

### Explanations (1:40 - 2:20)

- Each result includes a detailed explanation:
  - Similarity score (0.0-1.0)
  - Matched skills count
  - Years of experience
  - Clear reasoning for the ranking

### Tech Overview (2:20 - 3:00)

- Powered by Sentence-BERT for understanding document meaning beyond keyword matching.
- Processes PDF and DOCX files directly.
- Lightweight text processing with numpy for fast similarity calculations.
- Streamlit makes it easy to deploy this AI app with just a GitHub repo.

### Closing (3:00)

- That's CVButler - turning recruitment into an efficient AI-driven process!
- Check out the GitHub repo for setup instructions. Code is available under MIT license.

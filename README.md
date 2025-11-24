# CVButler: AI-Powered Candidate Matching

CVButler is a web application that leverages AI to help recruiters match job descriptions with candidate CVs. It uses Sentence-BERT embeddings to compute semantic similarity and ranks candidates, providing clear explanations for matching decisions.

## Features

- **File Upload**: Supports PDF, DOCX, and HTML formats for job descriptions and CVs
- **AI-Powered Matching**: Uses Sentence-BERT (all-mpnet-base-v2) for semantic similarity
- **Intelligent Explanations**: Provides detailed explanations based on skill matches and experience
- **Top 3 Ranking**: Displays the best matching candidates with scores and reasons
- **Easy Deployment**: Deployable on Streamlit Community Cloud

## Technology Stack

- **Frontend/Backend**: Streamlit
- **AI/ML**: Sentence Transformers, NumPy
- **Text Processing**: PDF Plumber, Python-Docx
- **Language**: Python 3.8+

## Installation and Local Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repo/CVButler.git
   cd CVButler
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Note: Downloading the SBERT model (~90MB) may take some time on first run.

3. **Run the application**:
   ```bash
   streamlit run app/app.py
   ```
   The app will open in your default browser at `http://localhost:8501`.

## Usage

1. **Upload Job Description**: Select a PDF or DOCX file containing the job requirements.
2. **Upload CVs**: Select multiple CV files (PDF or DOCX) for candidates.
3. **Match Candidates**: Click the "Match Candidates" button to process and rank.
4. **Review Results**: View the top 3 candidates with similarity scores and explanations.

## Demo Data

Fictional demo data is provided in the `demo_data/` folder:

- `JD.pdf` / `JD.docx`: Sample job description
- `*_CV.*`: Sample candidate CVs (John Doe, Jane Smith, Bob Johnson)

Use these for testing and demonstrations.

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push your code to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select the repo.
4. Set main file path: `app/app.py`.
5. Deploy!

### Other Options

- **Docker**: Create a Dockerfile with Python image, install requirements, and run `streamlit run app/app.py`.
- **Render**: Similar to Docker, use their Python-based setups.

## Architecture Overview

### Components

- **Streamlit App** (`app/app.py`): Main UI and processing logic
- **File Reader** (`app/utils/file_reader.py`): Extracts text from PDFs/DOCX
- **Text Processing** (`app/utils/text_processing.py`): Cleans text, extracts skills/experience
- **Matcher** (`app/utils/matcher.py`): SBERT embeddings, similarity computation, ranking

### Data Flow

1. Upload files → Text extraction → Text cleaning
2. SBERT embedding → Cosine similarity → Ranking
3. Explanations generation → Results display

### SBERT Integration

- Model: sentence-transformers/all-mpnet-base-v2
- Uses cleaned text for embeddings
- Cosine similarity for matching
- Model cached for performance

## Privacy Considerations

- **No Data Storage**: Files are processed in-memory only
- **No Data Persistence**: No uploaded data is saved or stored
- **Local Processing**: All AI processing happens locally
- **Demo Only**: Not intended for production use without additional security measures

## Troubleshooting

- **Model Download Issues**: Ensure stable internet for first-time SBERT download
- **Memory Issues**: Process fewer CVs if encountering OOM errors
- **File Extraction Errors**: Verify PDFs/DOCX are not password-protected or corrupted

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test locally
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

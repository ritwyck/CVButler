import streamlit as st
from utils.file_reader import extract_text
from utils.matcher import rank_candidates

st.title("CVButler: AI-Powered Candidate Matching")
st.markdown(
    "Upload a job description and candidate CVs to rank the top candidates based on semantic similarity.")

# Matching method selection
method = st.selectbox(
    "Select Matching Method",
    ["LLM (DialoGPT)"],
    help="Advanced AI evaluation with detailed analysis using DialoGPT model."
)

# File uploaders
jd_file = st.file_uploader("Select Job Description", type=[
                           'pdf', 'docx', 'html'], help="Upload PDF, DOCX, or HTML file for the job description.")

cv_files = st.file_uploader("Select Candidate CVs", type=[
                            'pdf', 'docx', 'html'], accept_multiple_files=True, help="Upload multiple PDF, DOCX, or HTML files for candidate CVs.")

if jd_file is not None:
    st.success(f"Job Description uploaded: {jd_file.name}")
else:
    st.info("Please upload a job description.")

if cv_files:
    st.success(
        f"Uploaded {len(cv_files)} CV file(s): {', '.join([f.name for f in cv_files])}")
else:
    st.info("Please upload at least one CV.")

# Match button
if st.button("Match Candidates", disabled=(jd_file is None or len(cv_files) == 0)):
    try:
        with st.spinner("Processing files and matching candidates..."):
            # Extract JD text
            jd_text = extract_text(jd_file)

            # Extract CV texts and names
            cv_texts = []
            cv_names = []
            for cv_file in cv_files:
                text = extract_text(cv_file)
                cv_texts.append(text)
                cv_names.append(cv_file.name)

            # Rank candidates
            results = rank_candidates(
                jd_text, cv_texts, cv_names, method=method)

            # Display results
            st.subheader("Top 3 Matching Candidates")
            for i, result in enumerate(results, 1):
                st.markdown(f"**#{i}: {result['name']}**")
                st.write(f"Similarity Score: {result['score']:.3f}")
                st.write(f"Explanation: {result['explanation']}")
                st.divider()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    if jd_file is None or len(cv_files) == 0:
        st.warning(
            "Please upload both a job description and at least one CV to proceed.")

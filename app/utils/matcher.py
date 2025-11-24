import numpy as np
from .text_processing import clean_text, extract_experience, match_skills
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global model cache
_sbert_model = None
_glm_model = None
_glm_tokenizer = None


def get_sbert_model():
    """Load SBERT model once."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer('all-mpnet-base-v2')
    return _sbert_model


def get_glm_model():
    """Load GLM model once."""
    global _glm_model, _glm_tokenizer
    if _glm_model is None:
        # Small GPT-2 based model for faster testing
        model_name = "microsoft/DialoGPT-small"
        _glm_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False)
        _glm_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=False)
    return _glm_model, _glm_tokenizer


def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def rank_candidates(jd_text, cv_texts, cv_names, method="SBERT"):
    """Rank candidates based on semantic similarity or LLM evaluation and return top 3 with explanations."""
    if method == "SBERT":
        model = get_sbert_model()
        # Preprocess and embed JD
        jd_clean = clean_text(jd_text)
        jd_embedding = model.encode([jd_clean])[0]

        results = []
        for cv_text, cv_name in zip(cv_texts, cv_names):
            cv_clean = clean_text(cv_text)
            cv_embedding = model.encode([cv_clean])[0]
            similarity_score = compute_similarity(jd_embedding, cv_embedding)

            # Explanations
            exp_years = extract_experience(cv_text)
            from .text_processing import extract_keywords as get_keywords
            jd_keywords = get_keywords(jd_text)
            cv_keywords = get_keywords(cv_text)
            matched = jd_keywords.intersection(cv_keywords)
            matched_list = sorted(list(matched), key=len, reverse=True)
            matched_display = ", ".join(
                matched_list[:5]) + ("..." if len(matched_list) > 5 else "")
            total_jd_skills = len(jd_keywords)
            explanation = f"Similarity score: {similarity_score:.3f}. Matched skills: {matched_display} ({len(matched)}/{total_jd_skills}). Experience: {exp_years} years."

            results.append({
                'name': cv_name,
                'score': similarity_score,
                'explanation': explanation
            })
    elif "LLM" in method:
        model, tokenizer = get_glm_model()
        results = []
        for cv_text, cv_name in zip(cv_texts, cv_names):
            # Prompt for DialoGPT (dialog model)
            prompt = f"Human: As an HR expert, evaluate how this CV matches the job description.\n\nJob: {jd_text[:1000]}\n\nCV: {cv_text[:1000]}\n\nPlease give a score 0-1 and brief explanation.\n\nAssistant:"

            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512)
            # Move inputs to model device to avoid device mismatch warnings
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
            response = tokenizer.decode(
                outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Parse response
            score_match = re.search(
                r'Score:\s*([0-9.]+)', response, re.IGNORECASE)
            explanation_match = re.search(
                r'Explanation:\s*(.+)', response, re.IGNORECASE | re.DOTALL)

            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp to 0-1
            else:
                score = 0.5  # Default

            explanation = explanation_match.group(1).strip(
            ) if explanation_match else "LLM evaluation completed."

            results.append({
                'name': cv_name,
                'score': score,
                'explanation': explanation
            })
    else:
        raise ValueError(
            "Unsupported method. Use 'SBERT' or a method containing 'LLM'.")

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:3]

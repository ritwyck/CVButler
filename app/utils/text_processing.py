import re
import string


def clean_text(text):
    """Clean text: lowercase, remove punctuation, excess whitespace."""
    if not text:
        return ""
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove phone numbers (various patterns)
    text = re.sub(r'\(?\d{3}\)?\s*\d{3}\s*\d{4}', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove dates (MM/DD/YYYY, M/D/YY, Jan 2023, etc.)
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)
    text = re.sub(
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_experience(text):
    """Extract years of experience from text using regex."""
    if not text:
        return 0
    pattern = re.compile(r'\b(\d+)\s*years?\b', re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return max([int(m) for m in matches])  # Take maximum
    return 0


def extract_keywords(text, exclude_common=True, max_keywords=30):
    """Extract potential skill keywords from text."""
    if not text:
        return set()
    # First, get uppercase words from original text
    original_words = re.findall(r'\b\w+\b', text)
    uppercase_candidates = set(
        w.lower() for w in original_words if w.istitle() or any(c.isupper() for c in w))

    clean = clean_text(text)
    words = clean.split()

    # Quality filters
    quality_words = []
    for word in words:
        if len(word) < 4 or len(word) > 15:
            continue
        if re.search(r'\d', word):  # No digits
            continue
        if word not in uppercase_candidates:  # Must have had uppercase in original
            continue
        quality_words.append(word)

    if exclude_common:
        # Expanded common words (stops, months, sections, etc.)
        common_words = {
            # Previous stop words
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'a', 'an', 'this', 'that', 'these', 'those', 'will', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'may', 'might', 'must', 'am', 'be', 'been', 'being',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'me', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'when', 'where', 'why', 'how', 'all', 'each', 'some', 'many', 'much', 'other', 'another', 'such',
            # Additional
            'there', 'here', 'then', 'now', 'page', 'city', 'state', 'country', 'address', 'phone', 'email', 'website', 'contact', 'company', 'position', 'department', 'team', 'project', 'work', 'job', 'role', 'experience', 'education', 'skills', 'resume', 'cv', 'profile', 'summary', 'objective',
            # Months
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            # Days
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
            # Numbers as words (keep 'one' etc.? but for tech, maybe not)
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
        }
        quality_words = [
            word for word in quality_words if word not in common_words]

    # Sort by quality (prefer longer words) and cap
    quality_words = sorted(set(quality_words), key=lambda x: (
        len(x), x), reverse=True)[:max_keywords]
    return set(quality_words)


def match_skills(jd_text, cv_text):
    """Return matched skills count and list."""
    jd_keywords = extract_keywords(jd_text)
    cv_keywords = extract_keywords(cv_text)
    matched = jd_keywords.intersection(cv_keywords)
    return len(matched), list(matched)

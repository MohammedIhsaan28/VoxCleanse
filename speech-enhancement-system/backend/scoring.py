from __future__ import annotations

import re
from typing import Any, Dict, List

FILLERS = {
    "um",
    "uh",
    "ah",
    "er",
    "erm",
    "like",
    "you",
    "know",
    "actually",
    "basically",
    "literally",
    "so",
}

CONNECTORS = {
    "however",
    "therefore",
    "because",
    "moreover",
    "meanwhile",
    "although",
    "first",
    "second",
    "finally",
    "then",
    "also",
    "thus",
    "hence",
}


_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _sentences(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    chunks = _SENTENCE_SPLIT_RE.split(raw)
    return [c.strip() for c in chunks if c and c.strip()]


def _speech_rate_score(wpm: float) -> float:
    if wpm <= 0:
        return 0.0
    if 120 <= wpm <= 160:
        return 1.0
    if wpm < 120:
        return _clamp(wpm / 120.0)
    if wpm <= 180:
        # Mild penalty for slightly fast pace.
        return _clamp(1.0 - ((wpm - 160.0) / 20.0) * 0.25)
    # Stronger penalty as pace becomes hard to follow.
    return _clamp(0.75 - ((wpm - 180.0) / 120.0))


def _pause_score(pause_count: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.5
    pauses_per_min = pause_count / max(duration_sec / 60.0, 1e-6)
    return _clamp(1.0 - (pauses_per_min / 22.0))


def _grammar_score(cleaned_text: str) -> float:
    sents = _sentences(cleaned_text)
    if not sents:
        return 0.0

    complete = [s for s in sents if s.endswith((".", "!", "?")) and len(_tokenize(s)) >= 3]
    complete_ratio = len(complete) / len(sents)

    starts_upper = 0
    for s in sents:
        stripped = s.lstrip()
        if stripped and stripped[0].isupper():
            starts_upper += 1
    capitalization_ratio = starts_upper / len(sents)

    return _clamp(0.6 * complete_ratio + 0.4 * capitalization_ratio)


def _coherence_score(cleaned_text: str) -> float:
    words = _tokenize(cleaned_text)
    sents = _sentences(cleaned_text)
    if not words or not sents:
        return 0.0

    connector_count = sum(1 for w in words if w in CONNECTORS)
    target_connector_count = max(1, len(sents) - 1)
    connector_score = _clamp(connector_count / target_connector_count)

    sent_lengths = [len(_tokenize(s)) for s in sents if _tokenize(s)]
    if not sent_lengths:
        consistency_score = 0.0
    else:
        mean_len = sum(sent_lengths) / len(sent_lengths)
        variance = sum((x - mean_len) ** 2 for x in sent_lengths) / len(sent_lengths)
        std = variance ** 0.5
        cv = std / mean_len if mean_len > 0 else 1.0
        consistency_score = _clamp(1.0 - cv)

    return _clamp(0.5 * connector_score + 0.5 * consistency_score)


def _highlight_fillers(text: str) -> str:
    if not text:
        return ""

    highlighted = text
    patterns = [
        r"\byou\s+know\b",
        r"\bum\b",
        r"\buh\b",
        r"\bah\b",
        r"\berm\b",
        r"\ber\b",
        r"\blike\b",
        r"\bbasically\b",
        r"\bactually\b",
        r"\bliterally\b",
        r"\bso\b",
    ]

    for pattern in patterns:
        highlighted = re.sub(pattern, lambda match: f"[[{match.group(0)}]]", highlighted, flags=re.IGNORECASE)

    return highlighted


def _build_rewrite_examples(text: str) -> List[Dict[str, str]]:
    lower_text = (text or "").lower()
    examples: List[Dict[str, str]] = []

    def add_example(source: str, replacement: str) -> None:
        if len(examples) < 4:
            examples.append({"from": source, "to": replacement})

    if "you know" in lower_text:
        add_example("you know, I mean the idea is...", "I mean the idea is...")
    if re.search(r"\bum\b|\buh\b|\berm\b|\ber\b|\bah\b", lower_text):
        add_example("um, I think we should...", "I think we should...")
    if "like" in lower_text:
        add_example("like, I was saying...", "I was saying...")
    if "actually" in lower_text:
        add_example("actually, the point is...", "the point is...")
    if "basically" in lower_text:
        add_example("basically, we need to...", "we need to...")

    if not examples:
        add_example("Instead of adding a filler, pause briefly.", "Pause briefly and continue with the main idea.")

    return examples


def _map_communication_level(overall: float) -> Dict[str, str]:
    if overall < 25:
        return {"level": "A1", "label": "Beginner", "description": "Very limited communication control"}
    if overall < 45:
        return {"level": "A2", "label": "Elementary", "description": "Basic ideas are understandable"}
    if overall < 60:
        return {"level": "B1", "label": "Intermediate", "description": "Clear enough for everyday communication"}
    if overall < 75:
        return {"level": "B2", "label": "Upper-Intermediate", "description": "Good control with only minor issues"}
    if overall < 88:
        return {"level": "C1", "label": "Advanced", "description": "Strong and fluent communication"}
    return {"level": "C2", "label": "Proficient", "description": "Highly accurate and polished communication"}


def compute_scores(
    text: str,
    cleaned_text: str,
    duration_sec: float,
    pause_count: int = 0,
) -> Dict[str, Any]:
    words = _tokenize(text)
    clean_words = _tokenize(cleaned_text)

    total_words = len(words)
    unique_words = len(set(words))

    if total_words == 0 or duration_sec <= 0:
        return {
            "fluency": 0.0,
            "vocabulary": 0.0,
            "communication": 0.0,
            "overall": 0.0,
            "communication_level": "A1",
            "communication_level_label": "Beginner",
            "communication_level_description": "Very limited communication control",
            "overall_feedback": "Speak a little longer so the system can measure pace, vocabulary, and clarity more reliably.",
            "detailed_feedback": {
                "pace_feedback": ["No pace analysis available yet."],
                "vocabulary_feedback": ["No vocabulary analysis available yet."],
                "grammar_feedback": ["No grammar analysis available yet."],
                "rewrite_examples": [
                    {"from": "Instead of adding a filler, pause briefly.", "to": "Pause briefly and continue with the main idea."}
                ],
                "filler_highlights": [],
                "highlighted_transcript": "",
                "improvement_tips": ["Try a longer sample with a clear sentence or two."],
            },
            "metrics": {
                "total_words": total_words,
                "duration_sec": round(float(duration_sec), 2),
            },
            "feedback": ["Speak a bit longer to compute meaningful communication scores."],
        }

    # Fluency metrics
    wpm = (total_words / duration_sec) * 60.0
    filler_count = 0
    i = 0
    while i < total_words:
        w = words[i]
        if w == "you" and i + 1 < total_words and words[i + 1] == "know":
            filler_count += 1
            i += 2
            continue
        if w in FILLERS:
            filler_count += 1
        i += 1

    filler_ratio = filler_count / total_words
    repetition_count = sum(1 for idx in range(1, total_words) if words[idx] == words[idx - 1])
    repetition_ratio = repetition_count / total_words

    speech_rate_score = _speech_rate_score(wpm)
    pauses = max(0, int(pause_count))
    pause_quality = _pause_score(pauses, duration_sec)

    fluency = (
        0.4 * speech_rate_score
        + 0.3 * (1.0 - _clamp(filler_ratio))
        + 0.2 * (1.0 - _clamp(repetition_ratio))
        + 0.1 * pause_quality
    ) * 100.0

    # Vocabulary metrics
    lexical_diversity = unique_words / total_words
    avg_word_len = sum(len(w) for w in words) / total_words
    normalized_avg_word_len = _clamp(avg_word_len / 8.0)
    advanced_word_ratio = sum(1 for w in words if len(w) >= 7) / total_words

    vocabulary = (0.6 * lexical_diversity + 0.4 * normalized_avg_word_len) * 100.0

    # Communication metrics
    grammar = _grammar_score(cleaned_text)
    coherence = _coherence_score(cleaned_text)
    sentence_structure = _clamp(0.6 * grammar + 0.4 * coherence)
    clarity = _clamp(len(clean_words) / total_words)

    communication = (
        0.4 * grammar
        + 0.3 * clarity
        + 0.3 * sentence_structure
    ) * 100.0

    overall = 0.4 * fluency + 0.3 * vocabulary + 0.3 * communication
    highlighted_transcript = _highlight_fillers(text)

    pace_feedback: List[str] = []
    vocabulary_feedback: List[str] = []
    grammar_feedback: List[str] = []
    improvement_tips: List[str] = []

    if wpm < 110:
        pace_feedback.append(f"Your pace is {wpm:.0f} WPM, which is slower than the ideal 120-160 WPM range.")
        improvement_tips.append("Speak a little faster, but keep sentences separated with short natural pauses.")
    elif wpm <= 160:
        pace_feedback.append(f"Your pace is {wpm:.0f} WPM, which is within the ideal speaking range.")
    elif wpm <= 170:
        pace_feedback.append(f"Your pace is {wpm:.0f} WPM, which is slightly fast but still manageable.")
        improvement_tips.append("Slow down slightly on key phrases so words stay crisp.")
    else:
        pace_feedback.append(f"Your pace is {wpm:.0f} WPM, which is fast and may reduce clarity.")
        improvement_tips.append("Reduce speed a bit so listeners can process each phrase comfortably.")

    pace_feedback.append(f"Filler density is {filler_ratio * 100:.1f}% and repetition density is {repetition_ratio * 100:.1f}%.")
    if filler_ratio > 0.08:
        improvement_tips.append("Replace filler words like um, uh, and like with a brief pause.")
    if repetition_ratio > 0.04:
        improvement_tips.append("Reduce repeated words by planning the next phrase before speaking.")

    vocabulary_feedback.append(
        f"Lexical diversity is {lexical_diversity * 100:.1f}%, with an average word length of {avg_word_len:.1f} characters."
    )
    if lexical_diversity < 0.45:
        vocabulary_feedback.append("Vocabulary variety is limited. Use more distinct words instead of repeating the same terms.")
        improvement_tips.append("Swap repeated everyday words with more specific alternatives when possible.")
    else:
        vocabulary_feedback.append("Vocabulary variety is healthy for a short live sample.")

    if advanced_word_ratio > 0.2:
        vocabulary_feedback.append(f"Advanced-word usage is {advanced_word_ratio * 100:.1f}%, which adds richness to the speech.")
    else:
        vocabulary_feedback.append(f"Advanced-word usage is {advanced_word_ratio * 100:.1f}%; adding a few more precise words could improve style.")

    grammar_feedback.append(
        f"Grammar and sentence structure scored {grammar * 100:.1f}% with coherence at {coherence * 100:.1f}% and clarity at {clarity * 100:.1f}%."
    )
    if grammar < 0.7:
        grammar_feedback.append("Sentence completeness can improve. Finish thoughts with a clear subject and verb.")
        improvement_tips.append("Close each sentence cleanly before moving to the next idea.")
    else:
        grammar_feedback.append("Sentence structure is reasonably clear and easy to follow.")

    if not improvement_tips:
        improvement_tips.append("Keep the same pace, word variety, and sentence structure to maintain this quality.")

    rewrite_examples = _build_rewrite_examples(text)
    filler_counts = []
    seen_fillers = set()
    for token in words:
        if token in FILLERS or token == "know":
            if token in seen_fillers:
                continue
            count = words.count(token)
            if count > 0:
                filler_counts.append({"word": token, "count": count})
                seen_fillers.add(token)

    if overall >= 80:
        overall_feedback = "Strong communication overall. Your speech is clear, balanced, and easy to follow."
    elif overall >= 65:
        overall_feedback = "Good overall performance. A small reduction in fillers or a slight boost in vocabulary variety would improve it further."
    elif overall >= 50:
        overall_feedback = "Moderate overall performance. Focus on pace control, clearer sentence endings, and fewer repetitions."
    else:
        overall_feedback = "Overall communication needs improvement. Speak a bit longer, slow down slightly, and reduce filler usage for better clarity."

    level_info = _map_communication_level(overall)

    feedback: List[str] = []
    feedback.extend(pace_feedback)
    feedback.extend(vocabulary_feedback)
    feedback.extend(grammar_feedback)
    feedback.append(f"Overall feedback: {overall_feedback}")
    feedback.extend(f"Tip: {tip}" for tip in improvement_tips)

    return {
        "fluency": round(fluency, 2),
        "vocabulary": round(vocabulary, 2),
        "communication": round(communication, 2),
        "overall": round(overall, 2),
        "communication_level": level_info["level"],
        "communication_level_label": level_info["label"],
        "communication_level_description": level_info["description"],
        "overall_feedback": overall_feedback,
        "detailed_feedback": {
            "pace_feedback": pace_feedback,
            "vocabulary_feedback": vocabulary_feedback,
            "grammar_feedback": grammar_feedback,
            "rewrite_examples": rewrite_examples,
            "filler_highlights": filler_counts,
            "highlighted_transcript": highlighted_transcript,
            "improvement_tips": improvement_tips,
        },
        "metrics": {
            "duration_sec": round(float(duration_sec), 2),
            "total_words": total_words,
            "unique_words": unique_words,
            "wpm": round(wpm, 2),
            "filler_count": filler_count,
            "filler_ratio": round(filler_ratio, 4),
            "pause_count": pauses,
            "repetition_count": repetition_count,
            "repetition_ratio": round(repetition_ratio, 4),
            "lexical_diversity": round(lexical_diversity, 4),
            "avg_word_length": round(avg_word_len, 3),
            "advanced_word_ratio": round(advanced_word_ratio, 4),
            "grammar_score": round(grammar, 4),
            "coherence_score": round(coherence, 4),
            "clarity_score": round(clarity, 4),
        },
        "feedback": feedback,
    }

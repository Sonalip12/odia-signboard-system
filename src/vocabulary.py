"""
Closed vocabulary of common Odia signboard phrases and their English
translations. This acts as (a) the label set for the synthetic dataset and
(b) a lightweight offline "translation dictionary" so the whole pipeline can
run without any internet access or external translation API.

Odia script strings are UTF-8 encoded Odia (ଓଡ଼ିଆ) text.
"""

SIGNBOARD_VOCAB = [
    ("ଡାକ୍ତରଖାନା", "Hospital"),
    ("ପୋଲିସ ଷ୍ଟେସନ", "Police Station"),
    ("ବିଦ୍ୟାଳୟ", "School"),
    ("ରେଳବାଇ ଷ୍ଟେସନ", "Railway Station"),
    ("ବସ ଷ୍ଟାଣ୍ଡ", "Bus Stand"),
    ("ମନ୍ଦିର", "Temple"),
    ("ବଜାର", "Market"),
    ("ହୋଟେଲ", "Hotel"),
    ("ପେଟ୍ରୋଲ ପମ୍ପ", "Petrol Pump"),
    ("ବ୍ୟାଙ୍କ", "Bank"),
    ("ଡାକଘର", "Post Office"),
    ("ଗ୍ରନ୍ଥାଗାର", "Library"),
    ("ଫାର୍ମାସି", "Pharmacy"),
    ("ବିମାନବନ୍ଦର", "Airport"),
    ("ଉଦ୍ୟାନ", "Garden / Park"),
    ("ଚାଳକ ତାଲିମ କେନ୍ଦ୍ର", "Driving Training Centre"),
    ("ଜରୁରୀକାଳୀନ ପ୍ରସ୍ଥାନ", "Emergency Exit"),
    ("ପାର୍କିଂ", "Parking"),
    ("ଶୌଚାଳୟ", "Toilet"),
    ("ପ୍ରବେଶ ନିଷେଧ", "No Entry"),
]


def translate(odia_text: str) -> str:
    """Dictionary lookup translation; returns a fallback message if the
    recognized text doesn't exactly match a known vocabulary entry
    (common when OCR is imperfect on degraded images)."""
    mapping = {o: e for o, e in SIGNBOARD_VOCAB}
    if odia_text in mapping:
        return mapping[odia_text]
    # fuzzy fallback: closest match by character overlap
    best, best_score = None, 0
    for o, e in SIGNBOARD_VOCAB:
        score = len(set(odia_text) & set(o))
        if score > best_score:
            best, best_score = e, score
    if best and best_score >= max(2, len(odia_text) // 3):
        return f"{best} (approximate match)"
    return "Translation unavailable (text not recognized confidently)"

"""V5.27.11 PDF hybrid routing policy.

The V5.25.3 semantic parser remains the first choice whenever a *specific
component* (stem / option / solution) is semantically trustworthy.  The
V5.26.9 visual parser is the fallback only for components whose formula layer
looks fragmented or corrupt.

This is intentionally finer-grained than V5.27.1: one PDF, and even one
question, may mix semantic V5.25.3 content with a visual V5.26.9 fallback.
"""
from __future__ import annotations

import re

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

_TEX_FONT_RE = re.compile(
    r"(?:CMR|CMMI|CMSY|CMEX|LMMath|LatinModern|TeXGyre|STIX|XITS)", re.I
)
_MATH_TEXT_RE = re.compile(
    r"(?:[=<>≤≥≠≈±×÷√∫∑∏∞π∆Δαβγθλμσφω∠⊥∥∪∩°]|"
    r"\\(?:frac|dfrac|tfrac|sqrt|log|ln|sin|cos|tan|vec|overrightarrow|int|sum|lim)\b|"
    r"\b(?:sin|cos|tan|log|ln|lim|max|min)\s*[({]|"
    r"\b[A-Za-z]\s*[\^_]\s*\{?)"
)
_STRAY_NUMERIC_CLUSTER_RE = re.compile(r"(?<!\d)\d+(?:\s+\d+){2,}(?!\d)")
# Common signatures produced when Word/SymbolMT PDF glyphs were concatenated
# as prose instead of reconstructed as one formula.
_BROKEN_MATH_TEXT_RE = re.compile(
    r"(?:\b\d+\s+\d+\s+\d+(?:\s+\d+)*\b|"
    r"(?:^|\s)[+\-=/]\s*(?:$|[.,;:])|"
    r"\b(?:sin|cos|tan|log|ln)\s*$)", re.I
)
# V5.27.6: Word/SymbolMT-derived PDFs sometimes turn a degree sign into
# ``^o`` / ``^{o}`` / ``60 o``.  This is a visual corruption signature, not
# reliable semantic math, so that exact component should fall back to the
# V5.26.9 raster crop where the printed 60° remains intact.
_BROKEN_ANGLE_RE = re.compile(
    r"(?:\^\s*\{?\s*[oO0]\s*\}?|\b\d{1,3}\s+[oO](?=\s|[.,;:)\]]|$))"
)

# V5.27.6: common PDF extraction artefacts seen in the Physics sample.
# If a component contains a page marker, or a radiation option lost its
# beta/gamma symbol and only leaves ``Tia -`` / ``Tia +`` / ``Tia .``,
# the semantic layer is not trustworthy even if the surrounding prose is.
_PAGE_MARKER_RE = re.compile(r"\bTrang\s+\d+\b", re.I)
_BROKEN_RADIATION_OPTION_RE = re.compile(
    r"^\s*Tia\s*(?:[+\-]|\.)\s*\.?\s*(?:Trang\s+\d+)?\s*$", re.I
)
# V5.27.9: a semantic option such as ``y = .`` / ``f(x)=`` means the PDF
# math layer dropped the entire right-hand side.  This must never be accepted
# merely because the document is classified as native-TeX.
_DANGLING_EQUALS_RE = re.compile(
    r"(?:^|\s)(?:[A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*=\s*[.,;:]?\s*$"
)
_FLATTENED_POWER_RE = re.compile(r"(?<![A-Za-z])(?:[xyzmabt])\d(?![A-Za-z])", re.I)


def analyze_pdf_profile(path):
    result = {
        "kind": "unknown",
        "primary_math_safe": False,
        "pages": 0,
        "span_count": 0,
        "char_count": 0,
        "single_char_span_ratio": 1.0,
        "tex_font_char_ratio": 0.0,
        "dominant_font_char_ratio": 0.0,
        "dominant_size_char_ratio": 0.0,
    }
    if fitz is None:
        return result
    doc = fitz.open(str(path))
    font_chars = {}
    size_chars = {}
    spans = chars = one_char_spans = tex_chars = 0
    try:
        result["pages"] = len(doc)
        for page in doc:
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = str(span.get("text") or "")
                        if not text:
                            continue
                        n = len(text)
                        spans += 1
                        chars += n
                        if len(text.strip()) == 1:
                            one_char_spans += 1
                        font = str(span.get("font") or "")
                        font_chars[font] = font_chars.get(font, 0) + n
                        if _TEX_FONT_RE.search(font):
                            tex_chars += n
                        size = round(float(span.get("size") or 0.0) * 2.0) / 2.0
                        size_chars[size] = size_chars.get(size, 0) + n
    finally:
        doc.close()
    result["span_count"] = spans
    result["char_count"] = chars
    if spans:
        result["single_char_span_ratio"] = round(one_char_spans / spans, 4)
    if chars:
        result["tex_font_char_ratio"] = round(tex_chars / chars, 4)
        result["dominant_font_char_ratio"] = round(max(font_chars.values(), default=0) / chars, 4)
        result["dominant_size_char_ratio"] = round(max(size_chars.values(), default=0) / chars, 4)

    native_tex = result["tex_font_char_ratio"] >= 0.08
    clean_vector = (
        result["single_char_span_ratio"] <= 0.25
        and result["dominant_font_char_ratio"] >= 0.65
        and result["dominant_size_char_ratio"] >= 0.80
    )
    if native_tex:
        result["kind"] = "native-tex"
        result["primary_math_safe"] = True
    elif clean_vector:
        result["kind"] = "clean-vector"
        result["primary_math_safe"] = True
    else:
        result["kind"] = "fragmented-or-mixed"
        result["primary_math_safe"] = False
    return result


def _value(block):
    return str(block.get("value") or block.get("formula_text") or "")


def _latex_sane(value):
    value = str(value or "")
    stripped = value.strip()
    if not stripped or "\ufffd" in value or "merror" in value.lower():
        return False
    if value.count("{") != value.count("}"):
        return False

    # V5.27.10 — native-TeX PDFs can still contain a few atom groups that the
    # semantic parser joins incorrectly.  These signatures are local to one
    # component, so only that stem/option falls back to the V5.26.9 visual crop.
    #
    # 1) ``y=`` / ``f(x)=`` => the right-hand side disappeared.
    if re.search(r"=\s*[.,;:]?\s*$", stripped):
        return False
    # 2) A lone prime/apostrophe is an orphan derivative glyph.  In the
    #    12-main sample it appeared as an extra block after the question text.
    if re.fullmatch(r"(?:['’′]+|\\prime)", stripped):
        return False
    # 3) ``\\dfrac{+1}{...}`` is a characteristic split numerator: the
    #    preceding radical/term was dropped from the numerator.
    if re.search(r"\\(?:d?frac|tfrac)\{\s*\+", stripped):
        return False
    # 4) Repeated bare variables such as ``2xx`` are produced when numerator /
    #    denominator rows are accidentally concatenated.  Restrict this check
    #    to common math variables so command names are not affected.
    if re.search(r"(?<![A-Za-z\\])\d*\s*([xyz])\1(?![A-Za-z])", stripped):
        return False
    return True


def blocks_have_math(blocks):
    for block in blocks or []:
        kind = str(block.get("type") or "").lower()
        value = _value(block)
        if kind == "latex":
            return True
        if value and (_MATH_TEXT_RE.search(value) or _STRAY_NUMERIC_CLUSTER_RE.search(value)):
            return True
    return False


def blocks_primary_reliable(blocks, pdf_profile=None):
    """Can V5.25.3 safely supply this exact component?

    On a fragmented PDF, mathematical content is accepted only when it has a
    real semantic math block (LaTeX / formula_text) and the surrounding prose
    does not contain classic glyph-fragment corruption such as ``6 1 3 10``.
    Plain prose remains safe and therefore stays V5.25.3-first.
    """
    blocks = list(blocks or [])
    if not blocks:
        return False
    visible = False
    semantic_math = False
    text_parts = []
    for block in blocks:
        kind = str(block.get("type") or "").lower()
        value = _value(block)
        if kind == "latex":
            if not _latex_sane(value):
                return False
            semantic_math = True
            visible = True
        elif kind == "text":
            if value.strip():
                visible = True
                text_parts.append(value)
        elif kind == "image":
            if block.get("url") or block.get("data") or value:
                visible = True
            # Formula images made by the old semantic pipeline are acceptable
            # evidence only when they carry a formula_text representation.
            if value:
                semantic_math = True
        elif kind == "break":
            continue
    if not visible:
        return False

    joined = " ".join(text_parts)
    all_values = " ".join(_value(block) for block in blocks)
    stripped_joined = joined.strip()
    if (
        _STRAY_NUMERIC_CLUSTER_RE.search(joined)
        or _BROKEN_MATH_TEXT_RE.search(joined)
        or _BROKEN_ANGLE_RE.search(all_values)
        or _PAGE_MARKER_RE.search(all_values)
        or _BROKEN_RADIATION_OPTION_RE.match(stripped_joined)
        or _DANGLING_EQUALS_RE.search(all_values.strip())
        or ('=' in all_values and len(_FLATTENED_POWER_RE.findall(all_values))>=3)
    ):
        return False

    has_math = blocks_have_math(blocks)
    if not has_math:
        return True
    if (pdf_profile or {}).get("primary_math_safe"):
        return True
    # Fragmented/mixed PDF: mathematical content must be represented by a
    # semantic formula block rather than only as loose text glyphs.
    return semantic_math


def question_has_math(question):
    if not question:
        return False
    if blocks_have_math(question.get("stem") or []):
        return True
    if blocks_have_math(question.get("solution") or []):
        return True
    for option in question.get("options") or []:
        if blocks_have_math(option.get("content") or []):
            return True
    return False


def component_prefers_visual(blocks, pdf_profile=None):
    """Route one stem/option/solution, not the entire question."""
    return not blocks_primary_reliable(blocks, pdf_profile)



def visual_option_requires_whole_block(primary_blocks, visual_blocks, pdf_profile=None):
    """Return True when a long/multiline option must be kept as one visual crop.

    Fragmented Word/SymbolMT PDFs can make the semantic pipeline reconstruct
    the first formula line correctly while silently dropping or shifting a
    second line.  The V5.26.9 visual crop already preserves the whole printed
    option.  For a tall visual option in a mixed PDF, prefer that complete
    crop whenever the semantic component contains math or is structurally
    complex.  Short one-line options remain V5.25.3-first.
    """
    primary_blocks=list(primary_blocks or [])
    visual_blocks=list(visual_blocks or [])
    if not primary_blocks or not visual_blocks:
        return False
    if (pdf_profile or {}).get("primary_math_safe"):
        return False

    visual_heights=[]
    for block in visual_blocks:
        if str(block.get("type") or "").lower() != "image":
            continue
        try:
            h=float(block.get("display_height_px") or block.get("height") or 0)
        except Exception:
            h=0
        if h > 0:
            visual_heights.append(h)
    if not visual_heights or max(visual_heights) < 64:
        return False

    visible_primary=[
        block for block in primary_blocks
        if str(block.get("type") or "").lower() in ("text","latex","image")
        and (_value(block).strip() or block.get("url") or block.get("data"))
    ]
    joined=" ".join(_value(block) for block in primary_blocks).strip()
    complex_semantic=(
        blocks_have_math(primary_blocks)
        or len(visible_primary) >= 3
        or len(joined) >= 150
    )
    return complex_semantic

def question_prefers_visual(primary_question, pdf_profile):
    """Compatibility helper used by older callers.

    V5.27.6 no longer blindly switches every math question in a mixed PDF.
    It switches the whole question only when its stem itself is unreliable;
    options are routed independently by ``component_prefers_visual``.
    """
    if not primary_question:
        return True
    stem = primary_question.get("stem") or []
    return component_prefers_visual(stem, pdf_profile)

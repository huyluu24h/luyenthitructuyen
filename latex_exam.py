from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable


SECTION_RE = re.compile(r"PHẦN\s*(I{1,3}|IV|[1-4])\b", re.I)
SECTION_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "1": 1, "2": 2, "3": 3, "4": 4}
# Alternative Vietnamese exam section markers frequently used by ex_test-style
# LaTeX sources. Both wrapper environments and standalone marker commands are
# supported:
#   \begin{cautn} ... \end{cautn}  or  \cautn  -> Part I
#   \begin{cauds} ... \end{cauds}  or  \cauds  -> Part II
#   \begin{caukq} ... \end{caukq}  or  \caukq  -> Part III
ALT_SECTION_MAP = {"cautn": 1, "cauds": 2, "caukq": 3}
ALT_SECTION_RE = re.compile(
    r"\\begin\{(cautn|cauds|caukq)\}|\\(cautn|cauds|caukq)\b",
    re.I,
)


class LatexExamError(RuntimeError):
    pass


def _strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        buf = []
        escaped = False
        for ch in line:
            if ch == "%" and not escaped:
                break
            buf.append(ch)
            if ch == "\\" and not escaped:
                escaped = True
            else:
                escaped = False
        out.append("".join(buf))
    return "\n".join(out)


def _balanced(text: str, open_pos: int, opener: str = "{", closer: str = "}") -> tuple[str, int]:
    if open_pos >= len(text) or text[open_pos] != opener:
        raise LatexExamError("Cấu trúc ngoặc LaTeX không hợp lệ.")
    depth = 0
    i = open_pos
    start = open_pos + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    raise LatexExamError("Thiếu dấu ngoặc đóng trong file LaTeX.")


def _command_arg(text: str, command: str, start: int = 0) -> tuple[str | None, int, int]:
    m = re.search(r"\\" + re.escape(command) + r"\b", text[start:])
    if not m:
        return None, -1, -1
    cmd_start = start + m.start()
    pos = start + m.end()
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != "{":
        return None, cmd_start, pos
    content, end = _balanced(text, pos)
    return content, cmd_start, end


def _command_arg_optional(text: str, command: str, start: int = 0) -> tuple[str | None, str | None, int, int]:
    """Read \\command[optional]{required} or \\command{required}."""
    m = re.search(r"\\" + re.escape(command) + r"\b", text[start:])
    if not m:
        return None, None, -1, -1
    cmd_start = start + m.start()
    pos = start + m.end()
    while pos < len(text) and text[pos].isspace():
        pos += 1
    optional = None
    if pos < len(text) and text[pos] == "[":
        optional, pos = _balanced(text, pos, "[", "]")
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if pos >= len(text) or text[pos] != "{":
        return None, optional, cmd_start, -1
    content, end = _balanced(text, pos)
    return content, optional, cmd_start, end


def _choice_args(text: str, command: str) -> tuple[list[str], int, int]:
    """Read four braced options after \\choice/\\choiceTF.

    ex_test sources commonly write ``\\choiceTF[t]`` (and occasionally other
    bracket options).  The option controls source-side formatting only and is
    not one of the four statements, so skip it before consuming the braces.
    """
    m = re.search(r"\\" + re.escape(command) + r"\b", text)
    if not m:
        return [], -1, -1
    pos = m.end()
    while pos < len(text) and text[pos].isspace():
        pos += 1
    # Optional formatter, e.g. \\choiceTF[t].
    if pos < len(text) and text[pos] == "[":
        try:
            _, pos = _balanced(text, pos, "[", "]")
        except LatexExamError:
            return [], m.start(), pos
    args: list[str] = []
    last = pos
    while len(args) < 4:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            break
        content, pos = _balanced(text, pos)
        args.append(content)
        last = pos
    return args, m.start(), last


def _find_section(source: str, position: int) -> int | None:
    """Return the explicitly active exam section at *position*.

    Only questions that belong to a real Part I/II/III/IV marker are imported.
    This deliberately ignores any ``ex`` environments in headers, examples,
    notes, answer appendices or other material that appears before the first
    recognized section marker.
    """
    markers: list[tuple[int, int]] = []

    for m in SECTION_RE.finditer(source, 0, position):
        markers.append((m.start(), SECTION_MAP.get(m.group(1).upper(), 1)))

    for m in ALT_SECTION_RE.finditer(source, 0, position):
        name = (m.group(1) or m.group(2) or "").lower()
        mapped = ALT_SECTION_MAP.get(name)
        if mapped:
            markers.append((m.start(), mapped))

    if not markers:
        return None
    markers.sort(key=lambda item: item[0])
    return markers[-1][1]


def _unwrap_formatting_macros(text: str) -> str:
    """Remove presentation-only text macros while preserving nested content.

    Regex-only stripping breaks on constructs such as
    ``\\textit{... $\\text{cm}^3$ ...}`` because the outer argument contains
    nested braces.  Use the parser's balanced-brace reader instead.
    """
    commands = ("textit", "textbf", "emph", "textnormal", "textup", "textsl", "textsc", "textsf", "texttt")
    pat = re.compile(r"\\(?:" + "|".join(commands) + r")\s*\{")
    search_from = 0
    while True:
        m = pat.search(text, search_from)
        if not m:
            break
        try:
            content, end = _balanced(text, m.end() - 1)
        except LatexExamError:
            # Malformed formatting should never leak the command to the UI.
            text = text[:m.start()] + text[m.end():]
            search_from = m.start()
            continue
        text = text[:m.start()] + content + text[end:]
        search_from = max(0, m.start())
    return text


def _remove_outer_layout(text: str) -> str:
    # Presentation-only environments/macros; math is preserved separately later.
    text = re.sub(r"\\begin\{(?:center|flushleft|flushright)\}", "", text)
    text = re.sub(r"\\end\{(?:center|flushleft|flushright)\}", "", text)
    text = text.replace("\\allowdisplaybreaks", "")
    text = _unwrap_formatting_macros(text)
    # Keep \textrm{...$...$...} intact while it is still part of a larger
    # math formula; _normalise_math will safely rewrite the nested math shift.
    text = re.sub(r"\\textrm\s*\{([^{}$]*)\}", r"\1", text)
    text = text.replace("\\lq", "“").replace("\\rq", "”")
    text = text.replace("~", " ")
    return text


def _convert_heva(tex: str) -> str:
    # Vietnamese exam macro: \heva{&a &b\\&c &d} -> MathJax cases.
    # Many exam files start each row with a leading ``&`` and then use a
    # second ``&`` to separate the formula from the condition. Keep the real
    # column separator so MathJax renders a proper two-column piecewise form.
    pat = re.compile(r"\\heva\s*\{")
    while True:
        m = pat.search(tex)
        if not m:
            break
        content, end = _balanced(tex, m.end() - 1)
        rows = []
        for raw_row in re.split(r"(?<!\\)\\\\", content):
            row = raw_row.strip()
            row = re.sub(r"^\s*&\s*", "", row)
            rows.append(row)
        repl = r"\begin{cases}" + r"\\".join(rows) + r"\end{cases}"
        tex = tex[:m.start()] + repl + tex[end:]
    return tex


def _rewrite_text_macros_with_inner_math(tex: str) -> str:
    r"""Rewrite text macros that contain $...$ while already inside math mode.

    Vietnamese exam sources commonly use constructs such as::

        $f(x)=\heva{&x-4 &\textrm{ khi $x<4$}\\ ...}$

    TeX accepts the inner math shift because ``\textrm`` enters text mode, but
    MathJax's TeX input can leave those dollar signs visible in some contexts.
    Turn it into ``\text{ khi }x<4`` (and analogous mixed text/math
    fragments) before sending the formula to MathJax.
    """
    commands = ("textrm", "text", "mathrm")
    for command in commands:
        search_from = 0
        while True:
            m = re.search(r"\\" + command + r"\s*\{", tex[search_from:])
            if not m:
                break
            start = search_from + m.start()
            brace = search_from + m.end() - 1
            try:
                content, end = _balanced(tex, brace)
            except LatexExamError:
                break
            if "$" not in content:
                search_from = end
                continue

            pieces = []
            cursor = 0
            for mm in re.finditer(r"(?<!\\)\$([^$]+?)(?<!\\)\$", content, re.S):
                text_part = content[cursor:mm.start()]
                if text_part:
                    pieces.append(r"\text{" + text_part + "}")
                math_part = mm.group(1).strip()
                if math_part:
                    pieces.append(math_part)
                cursor = mm.end()
            tail = content[cursor:]
            if tail:
                pieces.append(r"\text{" + tail + "}")

            # If no complete inner $...$ pair was found, leave the macro alone.
            if cursor == 0:
                search_from = end
                continue
            replacement = " ".join(pieces)
            tex = tex[:start] + replacement + tex[end:]
            search_from = start + len(replacement)
    return tex


def _normalise_math(tex: str) -> str:
    tex = tex.strip()
    tex = _convert_heva(tex)
    tex = _rewrite_text_macros_with_inner_math(tex)
    tex = tex.replace(r"\mathscr", r"\mathcal")
    tex = re.sub(r"\\wideparen\s*\{([^{}]+)\}", r"\\overset{\\frown}{\1}", tex)
    tex = tex.replace(r"\overrightarrow", r"\overrightarrow")
    tex = re.sub(r"\\begin\{eqnarray\*?\}", r"\\begin{aligned}", tex)
    tex = re.sub(r"\\end\{eqnarray\*?\}", r"\\end{aligned}", tex)
    return tex.strip()


def _extract_math_tokens(text: str) -> list[tuple[int, int, str, str]]:
    tokens: list[tuple[int, int, str, str]] = []
    i = 0
    while i < len(text):
        if text.startswith("$$", i):
            j = text.find("$$", i + 2)
            if j >= 0:
                tokens.append((i, j + 2, text[i + 2:j], "block")); i = j + 2; continue
        if text.startswith(r"\[", i):
            j = text.find(r"\]", i + 2)
            if j >= 0:
                tokens.append((i, j + 2, text[i + 2:j], "block")); i = j + 2; continue
        if text.startswith(r"\(", i):
            j = text.find(r"\)", i + 2)
            if j >= 0:
                tokens.append((i, j + 2, text[i + 2:j], "inline")); i = j + 2; continue
        if text[i] == "$" and (i == 0 or text[i - 1] != "\\"):
            # Some Vietnamese exam macros place $...$ inside \textrm{...}
            # while that text command itself sits in a larger $...$ formula.
            # Match the closing dollar only at the same brace depth as the
            # opener so inner text-mode math shifts do not cut the formula.
            j = i + 1
            depth = 0
            while j < len(text):
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}" and depth > 0:
                    depth -= 1
                elif text[j] == "$" and depth == 0:
                    tokens.append((i, j + 1, text[i + 1:j], "inline")); i = j + 1; break
                j += 1
            else:
                i += 1
            continue
        # Common display environments in solution bodies.
        em = re.match(r"\\begin\{(aligned|align\*?|eqnarray\*?)\}", text[i:])
        if em:
            env = em.group(1)
            endm = re.search(r"\\end\{" + re.escape(env) + r"\}", text[i + em.end():])
            if endm:
                start_content = i + em.end()
                end_content = i + em.end() + endm.start()
                full_end = i + em.end() + endm.end()
                body = text[start_content:end_content]
                if env.startswith("eqnarray"):
                    body = r"\begin{aligned}" + body + r"\end{aligned}"
                else:
                    body = r"\begin{" + env + "}" + body + r"\end{" + env + "}"
                tokens.append((i, full_end, body, "block")); i = full_end; continue
        i += 1
    return tokens


def _plain_text(text: str) -> str:
    text = _remove_outer_layout(text)
    # Enumerations/itemize become readable line breaks; keep content.
    text = re.sub(r"\\begin\{enumerate\}(?:\[[^\]]*\])?", "", text)
    text = re.sub(r"\\end\{enumerate\}", "", text)
    text = re.sub(r"\\begin\{(?:itemize|itemchoice)\}", "", text)
    text = re.sub(r"\\end\{(?:itemize|itemchoice)\}", "", text)
    text = re.sub(r"\\item(?:ch)?\b", "\n• ", text)
    text = text.replace(r"\\", "\n")
    text = text.replace(r"\ ", " ")
    # Remove harmless formatting commands left outside math.
    text = re.sub(r"\\(?:small|footnotesize|normalsize|displaystyle)\b", "", text)
    text = re.sub(r"\\(?:noindent|par)\b", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _render_tabular(tabular: str, media_dir: Path, url_for_name: Callable[[str], str]) -> dict:
    """Render a LaTeX tabular environment faithfully instead of exposing raw TeX."""
    digest = hashlib.sha256(("tabular:" + tabular).encode("utf-8")).hexdigest()[:24]
    out = media_dir / f"latex_table_{digest}.png"
    if not out.is_file():
        media_dir.mkdir(parents=True, exist_ok=True)
        if not shutil.which("xelatex") or not shutil.which("pdftocairo"):
            raise LatexExamError("Máy chủ thiếu XeLaTeX/pdftocairo để dựng bảng LaTeX.")
        document = r"""\documentclass[border=3pt]{standalone}
\usepackage{fontspec}
\usepackage{amsmath,amssymb,mathtools,array}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{DejaVu Serif}}
\begin{document}
\fontsize{13pt}{14.95pt}\selectfont
\renewcommand{\arraystretch}{1.15}
""" + tabular + "\n\\end{document}\n"
        with tempfile.TemporaryDirectory(prefix="latex_table_") as td:
            td = Path(td)
            tex = td / "table.tex"
            tex.write_text(document, encoding="utf-8")
            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=td, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=40,
            )
            pdf = td / "table.pdf"
            if proc.returncode != 0 or not pdf.is_file():
                tail = "\n".join(proc.stdout.splitlines()[-30:])
                raise LatexExamError("Không dựng được bảng LaTeX. Chi tiết: " + tail[-1800:])
            pngbase = td / "table"
            proc2 = subprocess.run(
                ["pdftocairo", "-singlefile", "-png", "-r", "180", str(pdf), str(pngbase)],
                cwd=td, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=30,
            )
            png = td / "table.png"
            if proc2.returncode != 0 or not png.is_file():
                raise LatexExamError("Không chuyển được bảng LaTeX sang PNG.")
            shutil.copyfile(png, out)
    width = height = None
    try:
        from PIL import Image
        with Image.open(out) as im:
            width, height = im.size
    except Exception:
        pass
    block = {"type": "image", "url": url_for_name(out.name), "role": "figure", "figure_kind": "table", "source": "latex-tabular"}
    if width and height:
        maxw = 900
        scale = min(1.0, maxw / float(width))
        block["display_width_px"] = int(width * scale)
        block["display_height_px"] = int(height * scale)
    return block


def _split_top_level_commas(value: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(value[start:i].strip())
            start = i + 1
        i += 1
    parts.append(value[start:].strip())
    return parts


def _extract_command_brace_args(source: str, command: str, count: int) -> list[str] | None:
    m = re.search(r"\\" + re.escape(command) + r"(?:\s*\[[^\]]*\])?\s*\{", source)
    if not m:
        return None
    args: list[str] = []
    pos = m.end() - 1
    try:
        for _ in range(count):
            content, end = _balanced(source, pos)
            args.append(content)
            pos = end
            while pos < len(source) and source[pos].isspace():
                pos += 1
            if len(args) < count:
                if pos >= len(source) or source[pos] != "{":
                    return None
        return args
    except LatexExamError:
        return None


def _tkztab_to_web_block(tikz: str, media_dir: Path, url_for_name: Callable[[str], str]) -> dict:
    """Fail-safe browser table for tkz-tab when local MiKTeX/PGF rejects it.

    Some Windows MiKTeX/PGF combinations fail inside tkz-tab with generated
    node names such as M31 even though the source is valid.  A variation table
    must not abort the whole exam upload, so preserve its mathematical data in
    a native table block as a deterministic fallback.
    """
    init = _extract_command_brace_args(tikz, "tkzTabInit", 2)
    if not init:
        return {"type": "text", "value": "[Bảng biến thiên]", "source": "latex-tkztab-fallback"}
    labels_raw, x_raw = init
    labels = _split_top_level_commas(labels_raw)
    labels = [re.sub(r"\\s*/\\s*[-+]?\\d+(?:\\.\\d+)?\\s*$", "", x).strip() for x in labels]
    xs = _split_top_level_commas(x_raw)
    n = max(1, len(xs))

    # Use 2*n-1 visual slots: point, interval, point, ...
    width = max(1, 2 * n - 1)
    def blank_blocks():
        return []
    def cell_blocks(v: str):
        v = (v or "").strip()
        if not v:
            return []
        return fragment_to_blocks(v, media_dir, url_for_name)

    xcells: list[list[dict]] = [cell_blocks(labels[0] if labels else "$x$")]
    for idx, xv in enumerate(xs):
        if idx:
            xcells.append(blank_blocks())
        xcells.append(cell_blocks(xv))
    rows: list[list[list[dict]]] = [xcells]

    line_args = _extract_command_brace_args(tikz, "tkzTabLine", 1)
    if line_args:
        seq = _split_top_level_commas(line_args[0])
        # leading/trailing empty tokens are layout sentinels, not data cells
        while seq and not seq[0].strip(): seq.pop(0)
        while seq and not seq[-1].strip(): seq.pop()
        symbols = {"z": "0", "d": "∥", "t": "│", "h": "", "": ""}
        vals = [symbols.get(v.strip(), v.strip()) for v in seq]
        vals = (vals + [""] * width)[:width]
        label = labels[1] if len(labels) > 1 else "$f'(x)$"
        rows.append([cell_blocks(label)] + [cell_blocks(v) for v in vals])

    var_args = _extract_command_brace_args(tikz, "tkzTabVar", 1)
    if var_args:
        vals = _split_top_level_commas(var_args[0])
        expanded = [""] * width
        for i, v in enumerate(vals[:n]):
            # tkz-tab entries often look like +/$4$ or -/$-1$: keep value after slash.
            vv = v.split("/", 1)[1].strip() if "/" in v else v.strip()
            expanded[min(2*i, width-1)] = vv
        label = labels[2] if len(labels) > 2 else "$f(x)$"
        rows.append([cell_blocks(label)] + [cell_blocks(v) for v in expanded])

    return {
        "type": "table",
        "rows": rows,
        "role": "figure",
        "figure_kind": "variation-table",
        "source": "latex-tkztab-web-fallback",
    }


def _render_tikz(tikz: str, media_dir: Path, url_for_name: Callable[[str], str]) -> dict:
    digest = hashlib.sha256(tikz.encode("utf-8")).hexdigest()[:24]
    out = media_dir / f"latex_tikz_{digest}.png"
    if not out.is_file():
        media_dir.mkdir(parents=True, exist_ok=True)
        if not shutil.which("xelatex") or not shutil.which("pdftocairo"):
            raise LatexExamError("Máy chủ thiếu XeLaTeX/pdftocairo để dựng hình TikZ.")
        # Build a small standalone document around the extracted picture.
        # Many Vietnamese exam sources use helper packages on top of TikZ.
        # Detect those commands here so an isolated figure still has the same
        # dependencies as it had in the original source file.
        extra_packages: list[str] = []

        # tkz-tab: variation/sign tables (\tkzTabInit, \tkzTabLine, \tkzTabVar, ...).
        # Match any \tkzTab* command instead of a fixed list so later variants
        # do not regress into "Undefined control sequence".
        if re.search(r"\\tkzTab[A-Za-z@]*\b", tikz):
            extra_packages.append(r"\usepackage{tkz-tab}")

        # tkz-euclide helpers occasionally appear in geometry TikZ pictures.
        if re.search(r"\\tkz(?:Def|Draw|Mark|Label|Find|Get|Inter|Fill)[A-Za-z@]*\b", tikz):
            extra_packages.append(r"\usepackage{tkz-euclide}")

        document = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage{fontspec}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{xcolor}
\usepackage{tikz}
""" + ("\n".join(extra_packages) + "\n" if extra_packages else "") + r"""\usetikzlibrary{calc,angles,quotes,arrows.meta,intersections,patterns,positioning,decorations.pathreplacing,decorations.markings,decorations.pathmorphing,3d,backgrounds}
\begin{document}
""" + tikz + "\n\\end{document}\n"
        with tempfile.TemporaryDirectory(prefix="latex_tikz_") as td:
            td = Path(td)
            # V5.29.2: bundle tkz-tab with the application so Windows/MiKTeX
            # does not need the user to install/update that package manually.
            # TeX searches the current working directory before the system tree,
            # therefore a local tkz-tab.sty is picked up transparently.
            if re.search(r"\\tkzTab[A-Za-z@]*\b", tikz):
                bundled_tkz_tab = Path(__file__).resolve().parent / "vendor" / "tex" / "tkz-tab.sty"
                if bundled_tkz_tab.is_file():
                    shutil.copyfile(bundled_tkz_tab, td / "tkz-tab.sty")
            tex = td / "figure.tex"
            tex.write_text(document, encoding="utf-8")
            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=td, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=40,
            )
            pdf = td / "figure.pdf"
            if proc.returncode != 0 or not pdf.is_file():
                tail = "\n".join(proc.stdout.splitlines()[-30:])
                low = proc.stdout.lower()
                if "tkz-tab.sty" in low and "not found" in low:
                    raise LatexExamError(
                        "Không dựng được bảng biến thiên: thiếu tkz-tab.sty. "
                        "Bản V5.29.11 có kèm sẵn gói này trong vendor/tex; "
                        "hãy kiểm tra thư mục dự án không bị phần mềm giải nén/diệt virus bỏ mất file vendor/tex/tkz-tab.sty."
                    )
                if "undefined control sequence" in low and re.search(r"\\tkztab[a-z@]*\b", low):
                    raise LatexExamError(
                        "Không dựng được bảng biến thiên vì lệnh tkz-tab chưa được nạp đúng. "
                        "Bản V5.29.11 đã kèm tkz-tab nội bộ; vui lòng dùng đúng thư mục FULL sau khi giải nén hoàn toàn."
                    )
                if "tkz-euclide.sty" in low and "not found" in low:
                    raise LatexExamError(
                        "Không dựng được hình hình học vì MiKTeX chưa có gói tkz-euclide. "
                        "Hãy cài 'tkz-euclide' trong MiKTeX Console rồi chạy lại."
                    )
                if "canvas is xz plane at y" in low or "canvas is yz plane at x" in low or "canvas is xy plane at z" in low:
                    raise LatexExamError(
                        "Không dựng được hình TikZ 3D vì thư viện TikZ '3d' chưa được nạp. "
                        "Bản V5.29.11 đã tự nạp thư viện 3d cho file dựng hình tạm."
                    )
                if re.search(r"\\tkzTab[A-Za-z@]*\b", tikz):
                    return _tkztab_to_web_block(tikz, media_dir, url_for_name)
                raise LatexExamError("Không dựng được hình TikZ trong file LaTeX. Chi tiết: " + tail[-1800:])
            pngbase = td / "figure"
            proc2 = subprocess.run(
                ["pdftocairo", "-singlefile", "-png", "-r", "180", str(pdf), str(pngbase)],
                cwd=td, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=30,
            )
            png = td / "figure.png"
            if proc2.returncode != 0 or not png.is_file():
                raise LatexExamError("Không chuyển được hình TikZ sang PNG.")
            shutil.copyfile(png, out)
    width = height = None
    try:
        from PIL import Image
        with Image.open(out) as im:
            width, height = im.size
    except Exception:
        pass
    block = {"type": "image", "url": url_for_name(out.name), "role": "figure", "source": "latex-tikz"}
    if width and height:
        maxw = 620
        scale = min(1.0, maxw / float(width))
        block["display_width_px"] = int(width * scale)
        block["display_height_px"] = int(height * scale)
    return block


def fragment_to_blocks(fragment: str, media_dir: Path, url_for_name: Callable[[str], str]) -> list[dict]:
    fragment = fragment.strip()
    if not fragment:
        return []

    # First replace TikZ pictures with placeholders so internal $...$ labels are not split as ordinary text.
    tikz_blocks: list[dict] = []
    placeholder_map: dict[str, dict] = {}
    table_blocks: list[dict] = []
    table_pat = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.S)

    def table_repl(m):
        key = f"@@TABLE_{len(table_blocks)}@@"
        source = m.group(0)
        # V5.29.8: ordinary exam tables are rendered natively in the browser.
        # This removes the MiKTeX/XeLaTeX dependency that caused uploads to stop
        # on some Windows installations. Complex tables still get the faithful
        # XeLaTeX renderer first, with the same browser-table fallback.
        complex_table = bool(re.search(
            r"\\(?:multicolumn|multirow|cline|cmidrule|specialrule|resizebox|adjustbox|begin\{(?:tabularx|longtable|array)\})",
            source,
        ))
        if not complex_table:
            block = _tabular_to_web_block(source, media_dir, url_for_name)
        else:
            try:
                block = _render_tabular(source, media_dir, url_for_name)
            except LatexExamError as exc:
                block = _tabular_to_web_block(source, media_dir, url_for_name)
                block["render_warning"] = str(exc)[-1200:]
        table_blocks.append(block)
        placeholder_map[key] = block
        return "\n" + key + "\n"

    fragment = table_pat.sub(table_repl, fragment)
    pat = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.S)

    def _local_tikz_prefix(start: int) -> str:
        # Keep color definitions declared just before a picture. These are
        # common in Vietnamese exam sources and are semantically part of the
        # figure even though they sit outside the tikzpicture environment.
        # Without them TikZ reports e.g. "I do not know the key /tikz/cadetblue".
        prefix = fragment[:start]
        # Limit to the current question/nearby context to avoid dragging large
        # amounts of unrelated TeX into the standalone renderer.
        prefix = prefix[-8000:]
        commands: list[str] = []
        for mm in re.finditer(r"\\definecolor\s*\{[^{}]+\}\s*\{[^{}]+\}\s*\{[^{}]+\}", prefix):
            commands.append(mm.group(0))
        for mm in re.finditer(r"\\colorlet\s*\{[^{}]+\}\s*\{[^{}]+\}", prefix):
            commands.append(mm.group(0))
        # Preserve order while removing duplicates.
        seen = set()
        unique = []
        for cmd in commands:
            if cmd not in seen:
                seen.add(cmd); unique.append(cmd)
        return "\n".join(unique)

    def repl(m):
        key = f"@@TIKZ_{len(tikz_blocks)}@@"
        local_prefix = _local_tikz_prefix(m.start())
        source = (local_prefix + "\n" if local_prefix else "") + m.group(0)
        block = _render_tikz(source, media_dir, url_for_name)
        tikz_blocks.append(block); placeholder_map[key] = block
        return "\n" + key + "\n"
    fragment = pat.sub(repl, fragment)

    # Unwrap common exam macros \immini{TEXT}{FIGURE} and \immini[...]{TEXT}{FIGURE}. TikZ has already
    # become a placeholder, so both arguments can safely be concatenated.
    while True:
        m = re.search(r"\\immini(?:\s*\[[^\]]*\])?\s*\{", fragment)
        if not m:
            break
        try:
            first, end1 = _balanced(fragment, m.end() - 1)
            pos = end1
            while pos < len(fragment) and fragment[pos].isspace():
                pos += 1
            if pos >= len(fragment) or fragment[pos] != "{":
                fragment = fragment[:m.start()] + first + fragment[end1:]
                continue
            second, end2 = _balanced(fragment, pos)
            fragment = fragment[:m.start()] + first + "\n" + second + fragment[end2:]
        except LatexExamError:
            fragment = fragment[:m.start()] + fragment[m.end():]
            break
    # Unwrap one-argument layout macros around figures/text.
    for layout_cmd in ("centerline", "mbox"):
        while True:
            m = re.search(r"\\" + layout_cmd + r"\s*\{", fragment)
            if not m:
                break
            try:
                content, end = _balanced(fragment, m.end() - 1)
                fragment = fragment[:m.start()] + content + fragment[end:]
            except LatexExamError:
                fragment = fragment[:m.start()] + fragment[m.end():]
                break
    fragment = _remove_outer_layout(fragment)

    tokens = _extract_math_tokens(fragment)
    blocks: list[dict] = []
    cursor = 0
    for start, end, math, display in tokens:
        if start < cursor:
            continue
        left = fragment[cursor:start]
        if left:
            _append_text_and_placeholders(blocks, left, placeholder_map)
        value = _normalise_math(math)
        if value:
            blocks.append({"type": "latex", "value": value, "display": display, "source": "latex-source"})
        cursor = end
    if cursor < len(fragment):
        _append_text_and_placeholders(blocks, fragment[cursor:], placeholder_map)

    # Merge adjacent text blocks while keeping intentional newlines.
    merged: list[dict] = []
    for b in blocks:
        if b.get("type") == "text" and not b.get("value", "").strip():
            continue
        if merged and b.get("type") == "text" and merged[-1].get("type") == "text":
            merged[-1]["value"] += (" " if not merged[-1]["value"].endswith(("\n", " ")) else "") + b["value"]
        else:
            merged.append(b)
    return merged


def _append_text_and_placeholders(blocks: list[dict], text: str, placeholder_map: dict[str, dict]) -> None:
    if not text:
        return
    pattern = re.compile(r"@@(?:TIKZ|TABLE)_\d+@@")
    pos = 0
    for m in pattern.finditer(text):
        pre = _plain_text(text[pos:m.start()])
        if pre:
            blocks.append({"type": "text", "value": pre, "source": "latex-source"})
        block = placeholder_map.get(m.group(0))
        if block:
            blocks.append(dict(block))
        pos = m.end()
    tail = _plain_text(text[pos:])
    if tail:
        blocks.append({"type": "text", "value": tail, "source": "latex-source"})


def _clean_true(option: str) -> tuple[str, bool]:
    m = re.match(r"\s*\\True\b\s*", option)
    if m:
        return option[m.end():], True
    return option, False



def _tabular_to_web_block(tabular: str, media_dir: Path, url_for_name: Callable[[str], str]) -> dict:
    """Convert a common LaTeX tabular into a browser-native table fallback.

    This supports the exam-style tables used in question stems: column specs,
    hline rules, rows separated by double backslashes, and ampersand cells.
    Each cell is sent through fragment_to_blocks so inline math remains MathJax.
    """
    m = re.match(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}(.*?)\\end\{tabular\}", tabular, re.S)
    if not m:
        return {"type": "text", "value": _plain_text(tabular), "source": "latex-tabular-fallback"}
    body = m.group(1)
    body = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)\b", "", body)
    raw_rows = re.split(r"\\\\(?:\s*\[[^\]]*\])?", body)
    rows = []
    for raw_row in raw_rows:
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        cells = re.split(r"(?<!\\)&", raw_row)
        parsed_cells = []
        for cell in cells:
            cell = cell.replace(r"\&", "&").strip()
            parsed_cells.append(fragment_to_blocks(cell, media_dir, url_for_name))
        if any(parsed_cells):
            rows.append(parsed_cells)
    if not rows:
        return {"type": "text", "value": _plain_text(tabular), "source": "latex-tabular-fallback"}
    return {
        "type": "table",
        "rows": rows,
        "role": "figure",
        "figure_kind": "table",
        "source": "latex-tabular-web",
    }

def _normalise_short_answer(value: str) -> str:
    """Convert answer-only LaTeX punctuation to the plain value shown/stored.

    Vietnamese sources often protect the decimal comma as ``{,}`` in math
    mode.  For grading/display metadata we want the literal answer ``0,18``.
    """
    value = (value or "").strip()
    value = value.replace("{,}", ",")
    value = value.replace(r"\,", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def parse_latex_exam(path: Path, media_dir: Path, url_for_name: Callable[[str], str]) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    source = _strip_comments(raw)
    ex_re = re.compile(r"\\begin\{ex\}(.*?)\\end\{ex\}", re.S)
    questions: list[dict] = []
    sec_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for match in ex_re.finditer(source):
        section = _find_section(source, match.start())
        if section is None:
            # Ignore question-like environments outside explicit Parts I-IV.
            continue
        sec_counts[section] += 1
        body = match.group(1).strip()
        source_tag = None
        tag_match = re.match(r"^\s*\[([^]\n]+)\]\s*", body)
        if tag_match:
            source_tag = tag_match.group(1).strip()
            body = body[tag_match.end():].lstrip()
        global_order = len(questions) + 1
        qtype = {1: "mcq", 2: "tf", 3: "short", 4: "essay"}[section]

        solution, sol_start, sol_end = _command_arg(body, "loigiai")
        before_solution = body[:sol_start] if sol_start >= 0 else body
        solution_blocks = fragment_to_blocks(solution or "", media_dir, url_for_name)

        options: list[dict] = []
        correct_option = None
        tf_answers: list[bool] = []
        short_answer = None

        if qtype in {"mcq", "tf"}:
            cmd = "choiceTF" if qtype == "tf" else "choice"
            args, choice_start, choice_end = _choice_args(before_solution, cmd)
            if len(args) != 4:
                raise LatexExamError(f"Phần {section} - Câu {sec_counts[section]}: {cmd} phải có đúng 4 phương án.")
            stem_source = before_solution[:choice_start]
            for idx, arg in enumerate(args):
                cleaned, is_true = _clean_true(arg)
                key = "ABCD"[idx]
                options.append({"key": key, "content": fragment_to_blocks(cleaned, media_dir, url_for_name)})
                if qtype == "mcq" and is_true:
                    correct_option = key
                if qtype == "tf":
                    tf_answers.append(is_true)
            if qtype == "mcq" and correct_option is None:
                raise LatexExamError(f"Phần I - Câu {sec_counts[section]}: chưa thấy \\True trong 4 phương án.")
        elif qtype == "short":
            short, short_mode, short_start, short_end = _command_arg_optional(before_solution, "shortans")
            if short_start < 0 or short_end < 0:
                raise LatexExamError(
                    f"Phần III - Câu {sec_counts[section]}: thiếu \\shortans{{...}} hoặc \\shortans[oly]{{...}}."
                )
            short_answer = _normalise_short_answer(short or "")
            # Both forms are answer metadata only. Remove the COMPLETE command,
            # including optional [oly], from the visible question stem.
            stem_source = before_solution[:short_start] + before_solution[short_end:]
        else:
            stem_source = before_solution

        stem_blocks = fragment_to_blocks(stem_source, media_dir, url_for_name)
        if not stem_blocks:
            raise LatexExamError(f"Phần {section} - Câu {sec_counts[section]}: đề bài rỗng sau khi phân tích LaTeX.")

        questions.append({
            "original_order": global_order,
            "section_order": sec_counts[section],
            "source_tag": source_tag,
            "section": section,
            "qtype": qtype,
            "stem": stem_blocks,
            "options": options,
            "correct_option": correct_option,
            "short_answer": short_answer,
            "tf_answers": tf_answers,
            "solution": solution_blocks,
        })

    if not questions:
        raise LatexExamError("Không tìm thấy môi trường \\begin{ex}...\\end{ex} trong file LaTeX.")
    return questions

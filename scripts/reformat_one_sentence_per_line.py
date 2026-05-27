#!/usr/bin/env python3
"""Reformat a LaTeX-style .txt so every sentence sits on its own line and
paragraphs are separated by exactly one blank line.

Rules
-----
1. The file is split into blocks by one-or-more blank lines.
2. Each block is classified as either:
   - **structural**: starts with `\section`, `\subsection`, `\label{...}` alone,
     or contains a `\begin{...}` / `\end{...}` for a known environment
     (figure*, figure, itemize, table, equation). These blocks are kept
     verbatim — sentence-splitting inside a figure caption is left for the
     human, since captions are usually one logical sentence anyway.
   - **prose**: everything else. All internal newlines are joined to a
     single string, then split on sentence boundaries.
3. Sentence boundary = ". " (period + whitespace) preceded by a
   lowercase letter, a closing brace `}`, or a closing paren `)`, AND
   followed by an uppercase letter or backslash. This avoids splitting
   on decimals (`0.5`), single-letter abbreviations, or inline math.
4. `\paragraph{...}` and `\textbf{Label.}` are bumped onto their own
   line so the bold label stays separate from the prose that follows.

Idempotent: running twice produces the same output.

Usage
-----
    python reformat_one_sentence_per_line.py <input> <output>
"""

import re
import sys
from pathlib import Path

ENV_RE = re.compile(
    r'\\(begin|end)\{(figure\*?|itemize|enumerate|table\*?|tabular|equation\*?|align\*?)'
)


def is_structural(block: str) -> bool:
    stripped = block.lstrip()
    if ENV_RE.search(block):
        return True
    if stripped.startswith(('\\section', '\\subsection', '\\subsubsection',
                            '\\chapter')):
        return True
    # Lone \label{...} (no other content)
    if stripped.startswith('\\label{') and '\n' not in block.strip():
        return True
    return False


def split_sentences(text: str) -> list:
    """Split a flat prose string into sentences."""
    # First, bump \paragraph{...} or \textbf{Word.} prefix onto its own line
    text = re.sub(r'(\\paragraph\{[^}]+\})\s+', r'\1\n', text)
    text = re.sub(r'(\\textbf\{[^{}.]+\.\})\s+(?=[A-Z\\])', r'\1\n', text)
    # Split on period+space between lowercase/close-brace/paren and
    # uppercase/backslash. Re-attach the consumed period.
    parts = re.split(r'(?<=[a-z\)\}])\.[\s\n]+(?=[A-Z\\])', text)
    out = []
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        if i < len(parts) - 1:
            out.append(p + '.')
        else:
            out.append(p)
    return out


def reformat_prose(block: str) -> str:
    # Flatten all internal whitespace to single spaces, then split
    joined = re.sub(r'\s+', ' ', block.strip())
    sentences = split_sentences(joined)
    return '\n'.join(sentences)


def reformat(text: str) -> str:
    # Normalize line endings and split on blank lines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Ensure prose adjacent to an environment becomes its own block by
    # injecting a blank line before `\begin{...}` and after `\end{...}`
    # when one is not already present.
    text = re.sub(r'(?<!\n)\n(\\begin\{)', r'\n\n\1', text)
    text = re.sub(r'(\\end\{[^}]+\}[ \t]*)\n(?!\n)', r'\1\n\n', text)
    blocks = re.split(r'\n[ \t]*\n', text.strip())
    out_blocks = []
    for blk in blocks:
        if is_structural(blk):
            # Preserve as-is but strip trailing whitespace
            out_blocks.append(blk.rstrip())
        else:
            out_blocks.append(reformat_prose(blk))
    return '\n\n'.join(out_blocks) + '\n'


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    new = reformat(src.read_text())
    dst.write_text(new)
    print(f'Wrote {dst} ({len(new.splitlines())} lines)')


if __name__ == '__main__':
    main()

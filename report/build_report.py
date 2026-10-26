#!/usr/bin/env python3
"""Convert the source-faithful Markdown to the official NeurIPS LaTeX wrapper."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'report.md').read_text()
# Source comments remain in Markdown; Pandoc does not typeset HTML comments.
source = re.sub(r'<!--.*?-->', '', source, flags=re.S)
front, content = source.split('## Abstract\n', 1)
abstract, content = content.split('## 1. Task and evaluation objective', 1)
content = '## 1. Task and evaluation objective' + content
main, tail = content.split('## References\n', 1)
refs, appendix = tail.split('## Appendix A. Reproduction guide\n', 1)
appendix = '## Appendix A. Reproduction guide\n' + appendix
# Explicit heading numbers in Markdown are preserved visually, not doubled.
def latex(text):
    text = re.sub(r'^## ', '# ', text, flags=re.M)
    text = re.sub(r'^### ', '## ', text, flags=re.M)
    return subprocess.check_output(['pandoc', '--from=markdown+tex_math_dollars+autolink_bare_uris', '--to=latex', '--listings', '--wrap=none'], input=text, text=True)

preamble = r'''\documentclass{article}
\usepackage[final,sglblindworkshop,nonatbib]{neurips_2026}
\workshoptitle{Predictive AI Evaluation Competition (PAIEC)}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array,calc}
\usepackage{graphicx}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\usepackage{listings}
\usepackage{needspace}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible,keepspaces=true,showstringspaces=false}
\providecommand{\passthrough}[1]{#1}
\setcounter{secnumdepth}{-1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\setlength{\emergencystretch}{2em}
\title{Team Drift: Hierarchical Test-Time Adaptation for Predictive AI Evaluation}
\author{Alex Alvord\\Cloud Lab Works\\Team Drift}
\begin{document}
\maketitle
'''
status = '\\noindent\\textbf{Selected Codabench submission: 981740 (v2).} Technical-report release candidate; not submitted.\n\n'
# Include the release paragraph from frontmatter, without its Markdown quote marker.
release = '\n'.join(line[2:] for line in front.splitlines() if line.startswith('> '))
out = preamble + '\\begin{abstract}\n' + latex(abstract) + '\\end{abstract}\n' + status + latex(release) + latex(main)
out += '\\label{main-text-end}\n\\clearpage\n\\section*{References}\n' + latex(refs)
out += '\\clearpage\n' + latex(appendix) + '\n\\end{document}\n'
out = out.replace('−', r'\ensuremath{-}')
out = out.replace(r'\subsection{6.2', r'\Needspace{12\baselineskip}' + '\n' + r'\subsection{6.2')
(ROOT / 'report.tex').write_text(out)
print('Generated report/report.tex; compile from report/ with tectonic report.tex')

"""Compile the GlassBox Machine Learning Lab Project Report in LaTeX to PDF."""
from pathlib import Path
import subprocess
import sys

REPORT_DIR = Path(r"D:\sem5\ML\project\report")
TEX_FILE = REPORT_DIR / "glassbox_lab_report.tex"
PDF_FILE = REPORT_DIR / "glassbox_lab_report.pdf"

print("Compiling Pass 1 of pdflatex...")
res1 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "glassbox_lab_report.tex"],
                      cwd=str(REPORT_DIR), capture_output=True, text=True)
if res1.returncode != 0:
    print("Pass 1 Failed! Output:")
    print(res1.stdout[-1500:])
    sys.exit(1)

print("Compiling Pass 2 of pdflatex (resolving TOC, LOF, LOT, cross-references)...")
res2 = subprocess.run(["pdflatex", "-interaction=nonstopmode", "glassbox_lab_report.tex"],
                      cwd=str(REPORT_DIR), capture_output=True, text=True)
if res2.returncode != 0:
    print("Pass 2 Failed! Output:")
    print(res2.stdout[-1500:])
    sys.exit(1)

print(f"Compilation Successful! PDF generated at: {PDF_FILE}")

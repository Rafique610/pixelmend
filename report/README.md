# PixelMend Technical Report (IEEE Format)

This folder contains the complete, self-contained IEEE research paper for **PixelMend**, prepared strictly according to the assignment requirements.

## Files & Structure

```
report/
├── main.tex           # Complete IEEE paper source code
├── references.bib     # BibTeX citations
├── README.md          # Compilation guide
└── figures/           # 16 high-resolution experimental diagrams and grids
    ├── oxford_pets.png
    ├── corruptions.png
    ├── task1_alpha.png
    ├── task1_reconstructions.png
    ├── task1_failures.png
    ├── task2_confusion.png
    ├── task2_reconstructions.png
    ├── task2_failures.png
    ├── task3_heatmap.png
    ├── task3_routing_galleries.png
    ├── task3_comparison.png
    ├── task3_failures.png
    ├── task4_styles.png
    ├── task4_results.png
    ├── task4_failures.png
    └── stitch_dashboard.jpg
```

---

## How to Compile

### Option 1: Overleaf (Recommended - 1 Click)
1. Zip this `report` folder (containing `main.tex`, `references.bib`, and the `figures/` directory).
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** $\rightarrow$ **Upload Project**, and select the zip.
3. Click **Recompile**. Overleaf automatically recognizes `IEEEtran` and compiles the PDF with all figures, tables, and equations.

### Option 2: Local LaTeX (pdflatex / latexmk)
If you have MiKTeX or TeX Live installed:
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Submission Metadata
- **Author**: Muhammad Rafique (`i230747@isb.nu.edu.pk`)
- **Institution**: Department of Computer Science, National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad
- **GitHub Repository**: [https://github.com/Rafique610/pixelmend](https://github.com/Rafique610/pixelmend)
- **Demonstration Video**: [https://www.youtube.com/watch?v=mO7WHRdSPoE](https://www.youtube.com/watch?v=mO7WHRdSPoE)

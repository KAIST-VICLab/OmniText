# Legacy notebooks (original, unmodified)

These are the **original Jupyter notebooks** used to produce the results in the
OmniText paper (ICLR 2026). They are preserved here verbatim for **provenance and
reproducibility** — they are the ground-truth reference for the refactored code.

They are **not the maintained entry point**. For normal use, prefer the command-line
scripts in [`scripts/`](../../scripts), which are extracted from these notebooks and
adapted to the released **OmniText-Bench** dataset:

| Application | Legacy notebook | Maintained script |
|---|---|---|
| Text Removal | `OmniText_Removal.ipynb` | `scripts/run_removal.py` |
| Text Editing | `OmniText_Editing_Ref1.ipynb` | `scripts/run_editing.py` (`--ref ref1`) |
| Style-Based Text Editing | `OmniText_Editing_Ref2.ipynb` | `scripts/run_editing.py` (`--ref ref2`) |
| Text Insertion | `OmniText_Insertion_Ref1.ipynb` | `scripts/run_insertion.py` (`--ref ref1`) |
| Style-Based Text Insertion | `OmniText_Insertion_Ref2.ipynb` | `scripts/run_insertion.py` (`--ref ref2`) |
| Text Repositioning | `OmniText_Repositioning.ipynb` | `scripts/run_repositioning.py` |
| Text Rescaling | `OmniText_Rescaling.ipynb` | `scripts/run_rescaling.py` |

> Note: these notebooks reference the **old** `OmniText_Dataset/omni_labels.json`
> layout (e.g. the `addition` application key and a top-level `editing` text field).
> The maintained scripts target the released `OmniText-Bench/labels.json` schema.

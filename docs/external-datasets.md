# External dataset inference

`run_external.py` adapts native inputs to the existing removal and editing scripts.
The original `run_*.py --dataset-root OmniText-Bench ...` commands are unchanged.
Install the package (`pip install -e .`) and inference dependencies as in the README.
Run commands from the repository root.

## Formats and task mapping

- **SCUT-EnsText → removal**. Point `--dataset-root` at an extracted split
  containing `all_images/`. The dataset is a scene text removal benchmark, not a
  paired text editing benchmark. Extracted split directory names vary; pass the
  split explicitly. `all_labels/` can contain removal ground truth: it is **never
  interpreted as text annotations or read by inference**.
- **ScenePair → removal then editing**. Point the root at a directory containing
  `i_s/`, `i_s.txt`, `i_t.txt`. Text files contain `filename text`, one record per
  line. Pairing uses filenames, not line order; spaces in text are retained.
  `t_f/` is edited GT and is never read. `i_full/` and `*_full.txt` are not required:
  this command operates on the provided word crops, not full-scene reconstruction.

Format evidence: [SCUT-EnsText release](https://github.com/HCIILAB/SCUT-EnsText),
[the linked EraseNet loader](https://github.com/lcy0604/EraseNet/blob/master/data/dataloader.py),
[TextCtrl ScenePair instructions](https://github.com/weichaozeng/TextCtrl#4-evaluation),
and [TextCtrl inference loader](https://github.com/weichaozeng/TextCtrl/blob/main/src/dataset/textdata.py).
These upstream docs/code were inspected; the restricted dataset archives were not
obtained or audited in this implementation.

## Text region masks

OmniText needs spatial text regions. Supply `--mask-dir` with one grayscale PNG
per input (`<source-stem>.png`), at the **original image dimensions**. White
(values >=128) selects text to remove; black preserves background. Multiple text
regions are supported by a single binary mask. Use supplied compatible masks or
an external text detector/annotation tool; this CLI does not run a detector or
infer masks from GT differences. Masks are not guaranteed to ship with every
SCUT archive. The default mask directory is `<dataset-root>/mask`.

For ScenePair, supply a mask enclosing the source word. Its bounding rectangle
sets the target placement and the source style-reference region. The original
source crop is the reference; the reference's source transcription and desired
target transcription are passed separately. No target GT style reference is used.
Alternatively, `--whole-crop` explicitly allows a full-crop mask when no mask is
present. This is a coarse fallback that removes the whole crop, can destroy
background, and is not an established benchmark protocol. A provided mask always
takes precedence. SCUT requires masks and rejects `--whole-crop`.

## Usage

Validate paths, text pairs, mask sizes, and 512-square preparation on CPU:

```bash
python scripts/run_external.py --dataset-format scut-enstext \
  --dataset-root /data/SCUT-EnsText/test --mask-dir /data/scut_masks \
  --output-dir experiments/scut --dry-run

python scripts/run_external.py --dataset-format scene-pair \
  --dataset-root /data/ScenePair --mask-dir /data/scenepair_masks \
  --output-dir experiments/scenepair --dry-run
```

Remove `--dry-run` to load weights and perform inference. Use `--gpu 0 --limit 1`
for an initial GPU smoke run. Optional `--removal-config` and `--editing-config`
select the existing YAML configurations. ScenePair automatically runs removal
before editing; it does not require a separately generated removal directory.

Both stages retain the existing **512×512** model grid. Inputs and masks are
resized together (Lanczos images / nearest-neighbor masks); editing reference
coordinates are shifted into the right half of the 1024×512 grid. This is an
explicit preprocessing choice and can distort aspect ratios. Final results are
resized back to original dimensions and saved as `<source-stem>.png` plus a
`manifest.json` recording source names, text, masks, sizes, and internal IDs.
Input stems must be unique. Evaluation tools must account for PNG output names
when the source was JPEG. No metric computation is included. GPU intermediates
are temporary and removed on exit, including failures; final files are exported
only after both stages finish successfully.

## Verification status

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

CPU tests cover synthetic fixtures, name-based text pairing, missing/malformed
inputs, mask dimensions, prepared coordinates, the original benchmark helpers,
CLI preflight, and orchestration/export with stubbed GPU subprocesses.
They do not establish image quality or benchmark scores. Real archive validation,
actual model-weight loading, CUDA execution, source-style transfer with unequal
text lengths, and evaluation against the published protocols still require a
separate run with dataset access and a GPU. No benchmark results are claimed.

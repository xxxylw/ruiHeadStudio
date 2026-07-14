# HeadStudio Quantitative Evaluation

CPU-only evaluation for the fixed 104-image set: 26 prompts × four `it10000-*.png` views.

Create the environment with `conda create -n headstudio-eval python=3.10`, install a CPU build of PyTorch and then `pip install -r evaluation/requirements.txt`.

Run one metric with:

```powershell
conda run -n headstudio-eval python evaluation/run_evaluation.py `
  --batch-root "D:\_materials\_paper\ruiHeadStudio\headstudio_batch_20260623_flashavatar_flame_mouth_closure" `
  --output-dir evaluation/results/piqe_full --device cpu --metrics piqe
```

The final artifact directory contains per-image scores, retrieval rankings, a JSON summary, provenance, and a Markdown report. PIQE is lower-is-better and is only a no-reference image-distortion proxy; it is not a geometric reconstruction metric.

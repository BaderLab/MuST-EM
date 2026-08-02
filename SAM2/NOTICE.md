# Third-Party Code Notice

This directory contains a modified copy of [SAM 2](https://github.com/facebookresearch/sam2)
(Segment Anything in Images and Videos), developed by Meta Platforms, Inc., and
released under the Apache License, Version 2.0 (see `LICENSE` in this directory).

SAM2 was used, with modifications, as the video mask-propagation backbone for the
vascular and cellular instance segmentation pipeline described in the accompanying
manuscript (see `sam2maskpropagator.py` at the repository root). The upstream demo
notebooks and example assets are not included here; only the `sam2` package and the
model configuration files required to run the pipeline are retained.

Please cite the original SAM2 publication if you use this code:

> Ravi, N. et al. SAM 2: Segment Anything in Images and Videos. *arXiv preprint*
> arXiv:2408.00714 (2024).

# Third-Party Code Notice

This directory contains a modified copy of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
(`nnunetv2`), developed by the Division of Medical Image Computing, German Cancer
Research Center (DKFZ), and released under the Apache License, Version 2.0 (see
`LICENSE` in this directory).

nnU-Net was used, with modifications, to train and run the organelle segmentation
models described in the accompanying manuscript. Changes made to the original
source in this copy include project-specific dataset conversion, trainer, and
experiment-planning adjustments used for the mitochondria/ER segmentation task; the
core nnU-Net framework is otherwise unmodified.

Please cite the original nnU-Net publication if you use this code:

> Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method
> for deep learning-based biomedical image segmentation. *Nat Methods* 18, 203–211
> (2021). https://doi.org/10.1038/s41592-020-01008-z

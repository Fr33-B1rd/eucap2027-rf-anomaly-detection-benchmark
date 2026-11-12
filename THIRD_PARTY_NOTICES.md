# Third-party notices

This repository contains independent benchmark glue and adaptations of several
published anomaly-detection implementations. The project license applies only
to code for which the project authors hold the necessary rights; upstream
licenses continue to apply to derived portions and downloaded assets.

## CFLOW-AD

`iad_detection/methods/cflow_official.py` adapts the official CFLOW-AD
architecture from <https://github.com/gudovskiy/cflow-ad> at commit
`b2ebf9e673a0aa46992a3b18367ec066a57bba89`. The upstream BSD 3-Clause notice
is preserved in `berzelius/CFLOW_LICENSE.txt`.

## EfficientAD / Anomalib

The EfficientAD PDN-medium adapter follows the Apache-2.0 Anomalib reference
implementation and uses teacher weights distributed by the Anomalib project:
<https://github.com/open-edge-platform/anomalib>. Downloaded weights and the
ImageNette auxiliary dataset are not committed to this repository.

## PatchCore

The PatchCore adapter implements the method described by Roth et al. and is
informed by the official implementation:
<https://github.com/amazon-science/patchcore-inspection>.

## VAE-PER and IAD

The VAE architecture and percentile error ratio follow the authors' reference
repository: <https://github.com/QXSLAB/vae_ism_ano>. That repository currently
does not include an explicit software license. Confirm redistribution
permission before publishing a public repository containing code derived from
it. The IAD dataset is downloaded separately using the authors' instructions
and is not redistributed here.

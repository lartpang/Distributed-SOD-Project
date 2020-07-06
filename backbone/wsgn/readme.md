# Weight Standardization

Weight Standardization (WS) is a normalization method to accelerate micro-batch training. Micro-batch training is hard because small batch sizes are not enough for training networks with Batch Normalization (BN), while other normalization methods that do not rely on batch knowledge still have difficulty matching the performances of BN in large-batch training.

## Related Links

* Paper: https://arxiv.org/pdf/1903.10520.pdf
* GitHub: https://github.com/joe-siyuan-qiao/WeightStandardization

```text
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
```

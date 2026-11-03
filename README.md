# scDVICE

scDVICE clusters single-cell RNA-seq data using pretrained attribute and graph encoders. This repository provides six datasets, pretrained weights, and the training and evaluation code.

## Installation

Python 3.12 and a CUDA-capable GPU are recommended. Install the PyTorch build appropriate for your CUDA driver, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The code has been checked locally with PyTorch 2.5.1, PyTorch Geometric 2.8.0 and the package versions in `requirements.txt`. GPU execution depends on a compatible PyTorch/CUDA installation.

## Pretrained weights

Download the `.pt` file for each dataset you want to run from [GitHub Releases](https://github.com/yyyttt120/scDVICE/releases/latest) and place it in `pretrained/`. For example:

```text
pretrained/Quake_10x_Limb_Muscle.pt
```

Each weight file contains the model parameters and the features used to initialize the clustering head. `pretrained/manifest.json` provides the expected checksums; training checks the dataset and weight files before loading them.

## Train and evaluate

```bash
python train.py --dataset Quake_10x_Limb_Muscle --device cuda
```

Other dataset names are `Adam`, `Quake_Smart-seq2_Diaphragm`, `Quake_Smart-seq2_Limb_Muscle`, `Quake_Smart-seq2_Trachea`, and `Romanov`. Run `python train.py --help` for input and output path options. The same training settings are used for all datasets. Training runs for 200 epochs, and results are computed from the final epoch.

Results are saved in `outputs/<dataset>/`:

- `results.npz`: `embedding` (Z_mix), `probabilities`, predicted `labels`, and `cell_ids` in input order.
- `metrics.json`: final NMI, ARI, and clustering accuracy. Ground-truth labels are used for evaluation.

The pretrained weights are specific to the corresponding bundled dataset and are not interchangeable. The program checks the files against the manifest.

## Data sources

The bundled files use benchmark HDF5 representations. The source studies are [Adam et al.](https://doi.org/10.1242/dev.151142), [Tabula Muris](https://doi.org/10.1038/s41586-018-0590-4), and [Romanov et al.](https://doi.org/10.1038/nn.4462). Related benchmark data are available from [scziDesk](https://github.com/xuebaliang/scziDesk).

## License

The source code is provided under the [MIT License](LICENSE). The datasets are subject to their original sources' terms.

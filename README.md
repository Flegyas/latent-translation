# latent-translation

<p align="center">
    <a href="https://github.com/Flegyas/latent-translation/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/Flegyas/latent-translation/Test%20Suite/main?label=main%20checks></a>
    <a href="https://Flegyas.github.io/latent-translation"><img alt="Docs" src=https://img.shields.io/github/deployments/Flegyas/latent-translation/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

""

# Latent Space Translation via Semantic Alignment

Full code implementation coming soon via [Latentis](https://github.com/Flegyas/Latentis)!

## BibTeX

```bibtex
@inproceedings{
    maiorca2023latent,
    title={Latent Space Translation via Semantic Alignment},
    author={Valentino Maiorca and Luca Moschella and Antonio Norelli and Marco Fumero and Francesco Locatello and Emanuele Rodol{\`a}},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=pBa70rGHlr}
}
```


## Installation

```bash
pip install git+ssh://git@github.com/Flegyas/latent-translation.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:Flegyas/latent-translation.git
cd latent-translation
conda env create -f env.yaml
conda activate latent-translation
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```

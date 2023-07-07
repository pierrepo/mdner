# 📑 Molecular dynamics named entity recognition 🧬

**A Named Entity Recognition model for molecular dynamics data.**

MDNER is a NER model developed specifically to extract information from MD simulations.

[![Python 3.10.9](https://img.shields.io/badge/python-%E2%89%A5_3.10.9-blue.svg)](https://www.python.org/downloads/release/python-397/)
[![Conda 22.11.1](https://img.shields.io/badge/conda-%E2%89%A5_22.11.1-green.svg)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub last commit](https://img.shields.io/github/last-commit/pierrepo/mdner.svg)](https://github.com/pierrepo/mdner)
[![Black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/pierrepo/mdner.svg?style=social)](https://github.com/pierrepo/mdner)

## ⚙️ Prerequisites

### Hardware

For the GPU code, it's essential to have a relatively new Nvidia GPU that has a minimum memory capacity of 8.0 GiB.
It should also be noted that you must have a CUDA driver installed on your system. The code uses CUDA, a parallel computing platform developed by NVIDIA, to interact with GPUs. No specific requirements are needed for the CPU code. To use spaCy, see the [spaCy documentation](https://spacy.io/usage).

## 📦 Setup your environment

Clone the repository and install [mamba](https://github.com/mamba-org/mamba) :

```bash
git clone https://github.com/pierrepo/mdner.git
conda install mamba -n base -c conda-forge
```

Create the `mdner` conda environment and load it :

```
mamba env create -f binder/environment.yml
conda activate mdner
```

Note: you can also update the conda environment with :

```bash
mamba env update -f binder/environment.yml
```

To deactivate an active environment, use :

```
conda deactivate
```

## ✍ Generate annotations data

Generate json files for spaCy NER and text files containing titles and descriptions of our MD datasets available [here](https://sandbox.zenodo.org/record/1171298).

Launch the generation of text files and json files :

```
python3 scripts/generate_annotation.py
```

### Parameters

```
usage: generate_annotation.py [-h] [-c] [-p {mbart,bart-paraphrase,pegasus}] [-s SEED] [threshold] [cutoff]

Generate text and json files in the annotation folder to be used as training sets.

positional arguments:
  threshold             The threshold for the length of the descriptive texts. By default
                        the value is 594.
  cutoff                Select the descriptive texts where the cosine similarity is below
                        the threshold. By default the value is 0.2.

options:
  -h, --help            show this help message and exit
  -c, --clear           Clear the annotation folder and generate files.
  -p {mbart,bart-paraphrase,pegasus}, --paraphrase {mbart,bart-paraphrase,pegasus}
                        Paraphrase the annotation according to three paraphrasing models.
  -s SEED, --seed SEED  Set the seed for reproducibility in paraphrase. By default the
                        value is 42.
```

Annotating json files requires manual annotation and must be in the `annotations` folder. Use the `Entity Annotator` to annotate and edit json files by typing the following command :

```
streamlit run scripts/Entity_Annotator.py
```

There are various other tools for annotating such as [Prodigy](https://prodi.gy/) or a site that allows it: [https://tecoholic.github.io/ner-annotator/](https://tecoholic.github.io/ner-annotator/).

If you think you don't have enough data, you can duplicate the annotated texts with the following command:

```
python3 scripts/generate_annotation.py -p mbart
```

Duplication consists of paraphrasing, i.e. keeping the context of the original text and reformulating it in another way. Here you will use the mBART model for paraphrasing.

A presentation of the annotation structure can be found on [ANNOTATIONS](https://github.com/pierrepo/mdner/blob/main/docs/ANNOTATIONS.md).

## 📑 Create MDNER

The `mdner.py` script is used to create the model according to the defined parameters.

### Parameters

```
usage: mdner.py [-h] [-c] [-t d f p r] [-n NAME] [-g] [-p] [-m] [-s SEED]

Create a model for the molecular dynamics data.

options:
  -h, --help            show this help message and exit
  -c, --create          Create a dedicated Named Entity Recognition model for our molecular
                        dynamics data.
  -t d f p r, --train d f p r
                        Hyperparameters for the learning process where d is the percentage
                        of dropout. The f, p and r scores define what SpaCy believes to be
                        the best model after the learning process. Each value must be between
                        0 and 1, and the sum of these three values must be equal to 1.
  -n NAME, --name NAME  Name of the model.
  -g, --gpu             Use GPU for learning.
  -p, --paraphrase      Add paraphrase in the learning dataset.
  -m, --mol             Use only MOL entities.
  -s SEED, --seed SEED  Seed used to sample data sets for reproducibility. By default the
                        value is 42.
```

To create the `mdner`, the `-c`, `-t` and `-n` options must be used. The `-c` option tells the script to create a model. The `-t` option is the hyperparameters to be used to train the model. The `-n` option corresponds to the name model that will be created.

You can introduce paraphrases only in the learning set (training + test) with the `-p` option.

### Example

```
python3 scripts/mdner.py -c -t 0.4 0.0 1.0 0.0 -n my_model -g
```

Here, we define a model where the dropout will be 0.4 (40% of the nodes will be deactivate). The three other values correspond to the metrics. They allow us to consider what is the best model. Here we prefer the precision score rather than the recall score. The sum of these 3 values must be equal to 1.0. We have also chosen to create a model based on Transformers by using the `-g` option. If the `-g` option is not chosen, the model generated will be based on the cpu and will use a basic spaCy model.

At the end of the code execution, the best NER model will be evaluated on the validation set.

## 📈 Results
From the original and paraphrased texts obtained with the mBART model, we have trained two NER model based on the Transformers "*BioMed-RoBERTa-base*" and we evaluated the models on the validation set as shown in Table 1. The models were obtained on seed 7522.

<figure class="table" align="center">
<figcaption> Table 1 - Mean precision scores with standard deviation for each entity of the Transformers model based on "<i>BioMed-RoBERTa-base</i>" without and with paraphrase. Each model was generated over 3 replicates. The best precision scores per entity are shown in bold.</figcaption>
<table align="center">
<thead>
  <tr>
    <th>Entities<br></th>
    <th colspan="2">Precision score (%)<br></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td align="center">Transformers</td>
    <td align="center">Transformers + Paraphrase</td>
  </tr>
  <tr style="font-weight: bold;">
    <td align="center">MOL (molecule)</td>
    <td align="center">75 ± 1.3</td>
    <td align="center"><b>84 ± 1.4</b></td>
  </tr>
  <tr>
    <td align="center">FFM (force field &amp; model)</td>
    <td align="center">86 ± 2.1</td>
    <td align="center"><b>90 ± 1.6</b></td>
  </tr>
  <tr>
    <td align="center">TEMP (temperature)</td>
    <td align="center">90 ± 2.1</td>
    <td align="center"><b>91 ± 1.9</b></td>
  </tr>
  <tr>
    <td align="center">STIME (simulation time)</td>
    <td align="center">71 ± 6.3</td>
    <td align="center"><b>73 ± 8.2</b></td>
  </tr>
  <tr>
    <td align="center">SOFT (software)</td>
    <td align="center"><b>77 ± 1.5</b></td>
    <td align="center">66 ± 5.3</td>
  </tr>
  <tr>
    <td align="center">Total</td>
    <td align="center">75 ± 0.7</td>
    <td align="center"><b>82 ± 1.4</b></td>
  </tr>
</tbody>
</table>
</figure>

We note an increase in the precision score, particularly for our key entity, the MOL entity, which rises from 75% to 84%. Performance for the other entities is improved slightly, except for the SOFT entity.
The NER models were able to identify molecule names not present in the learning dataset, perfectly underlining the ability of the NER model to generalize and identify the desired entities, and demonstrating the relevance of fine-tuning on Transformer models [[1]](#1).


## 🚀 Use MDNER

![](https://raw.githubusercontent.com/pierrepo/mdner/master/assets/webapp.gif)

In order to run an example, you can launch a website with [Streamlit](https://streamlit.io/) to apply the MDNER model to a text and evaluate it. Simply enter the name of the model as an argument, as in the following command :

```
streamlit run scripts/MDner.py -- --model my_model
```

Using MDNER does not require a GPU. It may be advantageous to use a GPU to speed up the predictions of the NER model, but it is not mandatory.

## References
<a id="1">[1]</a> 
Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. 2020. Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8342–8360, Online. Association for Computational Linguistics.


"""Script generating a set of text and json files to be used as learning sets."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import glob
import os
import json
from datetime import datetime
import unicodedata
import re

parser = argparse.ArgumentParser(
    description="Generate text and json files in the annotation folder to be used as learning sets."
)
parser.add_argument("-c", "--clear", help="Clear the annotation.", action="store_true")
parser.add_argument(
    "threshold",
    help="The threshold for the description length.",
    nargs="?",
    default=594,
)
parser.add_argument(
    "cutoff",
    help="Select the description where the cosine similarity is below the threshold.",
    nargs="?",
    default=0.8,
)
args = parser.parse_args()


def load_data():
    """Retrieve our data and loads it into the pd.DataFrame object.

    Returns
    -------
    dict
        returns a tuple containing the pd.DataFrame objects of our datasets.
    """
    datasets = pd.read_parquet(
        "https://sandbox.zenodo.org/record/1169962/files/datasets.parquet"
    )
    gro = pd.read_parquet(
        "https://sandbox.zenodo.org/record/1169962/files/gromacs_gro_files.parquet"
    )
    gro_data = pd.merge(
        gro,
        datasets,
        how="left",
        on=["dataset_id", "dataset_origin"],
        validate="many_to_one",
    )
    gro_data["description_dataset"] = gro_data["title"] + " " + gro_data["description"]
    return gro_data


def description_length(id_selected: list, df: pd.DataFrame, threshold: float):
    """
    Select datasets that have a description length greater than the threshold.

    Parameters
    ----------
    df: pandas.DataFrame
        A datasets.
    threshold: int
        The threshold for the description length.

    Returns
    -------
    pandas.DataFrame
        The selected datasets.
    """
    data = df[df["dataset_id"].isin(id_selected)].copy()
    # print(len(data))
    data = data.drop_duplicates("dataset_id")
    # print(len(data))
    data.loc[:, "description_length"] = (
        data["description"].str.len() + data["title"].str.len()
    )
    if data["description_length"].isnull().values.any():
        data = data.dropna(subset=["description_dataset", "description_length"])
    data = data[data["description_length"] > threshold]
    selected = data["dataset_id"].tolist()
    print(
        f"[{datetime.now()}] [INFO]",
        len(selected),
        "descriptions selected according the threshold length",
    )
    return selected


def homogeneous_composition(id_selected, df: pd.DataFrame):
    """
    Select data with protein and lipid homogeneity.

    Parameters
    ----------
    df: pandas.DataFrame
        A datasets.

    Returns
    -------
    pandas.DataFrame
        The selected datasets.
    """
    data = df[df["dataset_id"].isin(id_selected)].copy()
    data = data.groupby("dataset_id").agg(
        has_lipid=("has_lipid", "any"),
        has_protein=("has_protein", "any"),
    )
    size_sample = min(data.sum())
    lipid_data = data["has_lipid"].sample(n=size_sample)
    protein_data = data["has_protein"].sample(n=size_sample)
    selected = set(list(protein_data.index) + list(lipid_data.index))
    print(
        f"[{datetime.now()}] [INFO] ",
        len(selected),
        " descriptions selected according the composition",
    )
    return selected


def corpus_similarity(id_selected: list, df: pd.DataFrame, cutoff: float):
    """
    Select n data with low similarity to other texts.

    Parameters
    ----------
    df: pandas.DataFrame
        A datasets.
    cutoff: float
        The threshold for the corpus sililarity.

    Returns
    -------
    pandas.DataFrame
        The selected datasets.
    """
    data = df[df["dataset_id"].isin(id_selected)].copy()
    data = data.drop_duplicates("dataset_id")
    vectorizer = TfidfVectorizer()
    trsfm = vectorizer.fit_transform(data["description_dataset"])
    cos_data = pd.DataFrame(
        [
            cosine_similarity(trsfm[i : i + 1], trsfm)[0]
            for i in range(len(data["description_dataset"]))
        ]
    )
    cos_data.index = data.index
    cos_data.columns = data.index
    if 0.0 <= cutoff <= 1.0:
        np.fill_diagonal(cos_data.values, -1)
        index_remove = cos_data[(cos_data > cutoff).any(axis=1)].index
        data = data.drop(index=index_remove.tolist(), axis=0)
    else:
        data = df
    selected = data["dataset_id"]
    print(
        f"[{datetime.now()}] [INFO] ",
        len(selected),
        " descriptions selected according the corpus similarity",
    )
    return selected


def clear_annotation(annotation: str):
    """
    Remove some characters from the annotation.

    Parameters
    ----------
    annotation: str
        A annotation containing title and description.

    Returns
    -------
    str
        The annotation without cleared characters.
    """
    # Replace _ by space
    annotation = annotation.replace("_", " ")
    # Add space between a character and a parenthesis
    annotation = re.sub(r"([^ ])(\()", r"\1 \2", annotation)
    # Remove special characters and uniform the form of writing (unicode normalization)
    annotation = unicodedata.normalize("NFKD", annotation)
    # Replace multiple special characters by a space
    annotation = re.sub(r"[– ]+", " ", annotation)
    annotation = re.sub("=+", " ", annotation)
    # Replace accentuated characters by their non-accentuated version
    annotation = re.sub("`|’", "'", annotation)
    return annotation


def create_annotation(df: pd.DataFrame):
    """
    Create a file for each annotation.

    Parameters
    ----------
    df: pandas.DataFrame
        A datasets.

    Returns
    -------
    pandas.DataFrame
        The selected datasets.
    """
    path = "../annotations/"
    print(f"[{datetime.now()}] [INFO] Writing annotations in files ...")
    for i in range(len(df)):
        path_file = path + df.loc[i, "dataset_origin"] + "_" + df.loc[i, "dataset_id"]
        with open(
            path_file + ".txt",
            "w",
        ) as f:
            f.write(df.loc[i, "description_dataset"])
        if not os.path.exists(path_file + ".json"):
            with open(
                path_file + ".json",
                "w",
            ) as json_file:
                dict_annotations = {
                    "classes": [
                        "TEMPERATURE",
                        "SOFTWARE",
                        "SIMULATION TIME",
                        "MOLECULE",
                        "FF & MODEL",
                    ],
                    "annotations": [
                        [df.loc[i, "description_dataset"], {"entities": []}]
                    ],
                }
                json.dump(dict_annotations, json_file)


def generate_annotation(threshold: int, cutoff: float):
    """
    Generate text files containing the title and description of the dataset.

    Parameters
    ----------
    threshold: int
        The threshold for the description length.
    cutoff: float
        The threshold for the corpus sililarity.
    """
    gro_data = load_data()
    id_selected = gro_data["dataset_id"].tolist()
    # Prerequise for the annotation
    id_selected = description_length(id_selected, gro_data, threshold)
    id_selected = corpus_similarity(id_selected, gro_data, cutoff)
    id_selected = homogeneous_composition(id_selected, gro_data)
    # Setups the data
    data = gro_data[gro_data["dataset_id"].isin(id_selected)].copy()
    data.drop_duplicates("dataset_id", inplace=True)
    data = data.reset_index(drop=True)
    # Cleaning up the annotation
    data["description_dataset"] = data["description_dataset"].apply(clear_annotation)
    # Write the annotations in files
    create_annotation(data)
    print(f"[{datetime.now()}] [INFO] Generation completed")


def clear_folder():
    """Remove all files in the folder."""
    path = "../annotations/"
    files = glob.glob(path + "*.txt")
    for f in files:
        os.remove(f)
    print(f"[{datetime.now()}] [INFO] Folder cleared")


if __name__ == "__main__":
    os.chdir(os.path.split(os.path.abspath(__file__))[0])
    if args.clear:
        clear_folder()
    generate_annotation(int(args.threshold), float(args.cutoff))

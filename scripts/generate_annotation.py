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
    return gro_data


def description_length(df: pd.DataFrame, threshold: float):
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
    df["description_length"] = df["description"].str.len() + df["title"].str.len()
    if df["description_length"].isnull().values.any():
        df = df.dropna(subset=["description_length", "title", "description"])
    data = df[df["description_length"] > threshold]
    print(f"[{datetime.now()}] [INFO] Number of description : ", data.shape[0])
    data.reset_index(drop=True, inplace=True)
    return data


def homogeneous_composition(df: pd.DataFrame):
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
    lipid_data = df[(df["has_protein"] == False) & (df["has_lipid"] == True)]
    protein_data = df[(df["has_protein"] == True) & (df["has_lipid"] == False)]
    size_sample = min(lipid_data.shape[0], protein_data.shape[0])
    sample_lipid = lipid_data.sample(n=size_sample)
    sample_protein = protein_data.sample(n=size_sample)
    data = pd.concat([sample_lipid, sample_protein], ignore_index=True)
    print(
        f"[{datetime.now()}] [INFO] Number of description : ",
        data.shape[0],
    )
    data.reset_index(drop=True, inplace=True)
    return data


def corpus_similarity(df: pd.DataFrame, cutoff: float):
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
    cols = ["title", "description"]
    df["corpus"] = df[cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    vectorizer = TfidfVectorizer()
    trsfm = vectorizer.fit_transform(df["corpus"])
    cos_data = pd.DataFrame(
        [
            cosine_similarity(trsfm[i : i + 1], trsfm)[0]
            for i in range(len(df["corpus"]))
        ]
    )
    if 0.0 <= cutoff <= 1.0:
        np.fill_diagonal(cos_data.values, -1)
        index_remove = cos_data[(cos_data > cutoff).any(axis=1)].index
        data = df.drop(index=index_remove.tolist(), axis=0)
    else:
        data = df
    print(
        f"[{datetime.now()}] [INFO] Number of description : ",
        data.shape[0],
    )
    data.reset_index(drop=True, inplace=True)
    return data


def clear_annotation(df: pd.DataFrame):
    """
    Remove some characters and urls from the annotation.

    Parameters
    ----------
    df: pandas.DataFrame
        A datasets.

    Returns
    -------
    pandas.DataFrame
        The selected datasets.
    """
    df = df.replace("_", " ", regex=True)
    df["annotation"] = df["annotation"].str.replace(
        r'https?://[^\s<>"]+|www\.[^\s<>"]+',
        "https://removed",
        regex=True,
    )
    return df


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
        with open(
            path + df.loc[i, "dataset_origin"] + "_" + df.loc[i, "dataset_id"] + ".txt",
            "w",
        ) as f:
            f.write(df.loc[i, "annotation"])
        with open(
            path
            + df.loc[i, "dataset_origin"]
            + "_"
            + df.loc[i, "dataset_id"]
            + ".json",
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
                "annotations": [[df.loc[i, "annotation"], {"entities": []}]],
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
    # Remove the duplicate description
    gro_data = gro_data.drop_duplicates("description")
    # Prerequise for the annotation
    df_length = description_length(gro_data, threshold)
    df_similarity = corpus_similarity(df_length, cutoff)
    df_composition = homogeneous_composition(df_similarity)
    # Setup and cleaning up the annotation
    df = df_composition[["dataset_id", "dataset_origin"]]
    df = df.copy()
    df["annotation"] = df_composition["title"] + " " + df_composition["description"]
    # Write the annotations in files
    create_annotation(df)
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

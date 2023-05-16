"""Script generating a set of text and json files to be used as learning sets."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import glob
import os
import json
import unicodedata
import re
import logging
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy

parser = argparse.ArgumentParser(
    description="Generate text and json files in the annotation folder to be used as training sets."
)
parser.add_argument("-c", "--clear", help="Clear the annotation.", action="store_true")
parser.add_argument(
    "threshold",
    help="The threshold for the length of the descriptive texts.",
    nargs="?",
    default=594,
)
parser.add_argument(
    "cutoff",
    help="Select the descriptive texts where the cosine similarity is below the threshold.",
    nargs="?",
    default=0.2,
)
parser.add_argument(
    "-d", "--duplicate", help="Duplicate the annotation.", action="store_true"
)
args = parser.parse_args()


def load_data():
    """Retrieve our data and loads it into the pd.DataFrame object.

    Returns
    -------
    dict
        returns a tuple containing the pd.DataFrame objects of our datasets.
    """
    df = pd.read_parquet(
        "https://sandbox.zenodo.org/record/1169962/files/datasets.parquet"
    )
    df["text_dataset"] = df["title"] + "\n" + df["description"]
    df.loc[:, "text_length"] = df["description"].str.len() + df["title"].str.len() + 1
    return df


def text_length(id_selected: list, df: pd.DataFrame, threshold: float):
    """
    Select datasets where the length of the text is greater than the threshold.

    Parameters
    ----------
    df: pandas.DataFrame
        A datasets.
    threshold: int
        The threshold for the length of the descriptive texts.

    Returns
    -------
    pandas.DataFrame
        The selected datasets.
    """
    data = df[df["dataset_id"].isin(id_selected)].copy()
    if data["text_length"].isnull().values.any():
        data = data.dropna(subset=["text_dataset", "text_length"])
    data = data[data["text_length"] > threshold]
    selected = data["dataset_id"].tolist()
    logging.info(f"{len(selected)} texts selected according the threshold length")
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
    vectorizer = TfidfVectorizer(stop_words="english")
    trsfm = vectorizer.fit_transform(data["text_dataset"])
    cos_data = pd.DataFrame(
        [
            cosine_similarity(trsfm[i : i + 1], trsfm)[0]
            for i in range(len(data["text_dataset"]))
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
    logging.info(f"{len(selected)} texts selected according the corpus similarity")
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
    annotation = re.sub(r"[^\x00-\x7f]", r" ", annotation)
    # Replace multiple special characters by a space
    annotation = re.sub(r"[– ]+", " ", annotation)
    annotation = re.sub("=+", " ", annotation)
    # Replace accentuated characters by their non-accentuated version
    annotation = re.sub("`|’", "'", annotation)
    return annotation


def create_json(path_file: str, text: str, ents: list):
    """
    Create a json file for each annotation.
    
    Parameters
    ----------
    path_file: str
        The path of the file.
    text: str
        The text of the annotation.
    ents: list
        The entities of the annotation.
    """
    with open(path_file, "w") as json_file:
        dict_annotations = {
            "classes": [
                "TEMP",
                "SOFT",
                "STIME",
                "MOL",
                "FFM",
            ],
            "annotations": [[text, {"entities": ents}]],
        }
        json.dump(dict_annotations, json_file)


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
    path = "annotations/"
    for i in range(len(df)):
        path_file = path + df.loc[i, "dataset_origin"] + "_" + df.loc[i, "dataset_id"]
        with open(
            path_file + ".txt",
            "w",
        ) as f:
            f.write(df.loc[i, "text_dataset"])
        if not os.path.exists(path_file + ".json"):
            create_json(path_file + ".json", df.loc[i, "text_dataset"], [])


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
    df = load_data()
    id_selected = df["dataset_id"].tolist()
    # Prerequise for the annotation
    id_selected = text_length(id_selected, df, threshold)
    id_selected = corpus_similarity(id_selected, df, cutoff)
    # Setups the data
    data = df[df["dataset_id"].isin(id_selected)].copy()
    data = data.reset_index(drop=True)
    # Cleaning up the annotation
    data["text_dataset"] = data["text_dataset"].apply(clear_annotation)
    # Write the annotations in files
    create_annotation(data)
    logging.info("Generation completed")


def clear_folder():
    """Remove all files in the folder."""
    path = "annotations/"
    for txt_file in glob.glob(path + "*.txt"):
        os.remove(txt_file)
    for json_file in glob.glob(path + "*.json"):
        os.remove(json_file)
    logging.info("Folder cleared")


# ------------------------------ START IN PROGRESS -----------------------------------#


def load_model_paraphrase():
    model = BartForConditionalGeneration.from_pretrained("eugenesiow/bart-paraphrase")
    tokenizer = BartTokenizer.from_pretrained("eugenesiow/bart-paraphrase")
    return model, tokenizer


def get_entities(annotations, text):
    ents = []
    for ent in annotations["entities"]:
        ents.append((text[ent[0] : ent[1]], ent[2]))
    return ents


def generate_paraphrase(input_sentence, model, tokenizer):
    batch = tokenizer(input_sentence, return_tensors="pt", padding="longest")
    generated_ids = model.generate(
        batch["input_ids"],
        temperature=1.2,
        num_beams=8,
        num_beam_groups=8,
        early_stopping=True,
        num_return_sequences=1,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        max_length=1024,
    )
    generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_sentence

def split_text(text):
    # List of punctuation to identify sentences
    punctuations = ["."]
    
    # Divide the text into parts with a maximum size of 1024 characters
    parts = []
    while len(text) > 0:
        # Find the last punctuation in the part
        last_punctuation_index = -1
        for punctuation in punctuations:
            index = text.rfind(punctuation, 0, 1024)
            if index > last_punctuation_index:
                last_punctuation_index = index
        
        if last_punctuation_index == -1:
            # No punctuation found, divide the part to 1024 characters
            parts.append(text[:1024])
            text = text[1024:]
        else:
            # Divide the part at the last punctuation found
            parts.append(text[:last_punctuation_index+1])
            text = text[last_punctuation_index+1:]
    
    return parts

def predict_paraphrase(text, model, tokenizer) :
    # The length must be less than 1024 characters according the model
    if len(text) < 1024:
        predict_text = generate_paraphrase(text, model, tokenizer)[0]
    else :
        # Split the text into parts with a maximum size of 1024 characters
        splited_text = split_text(text)
        predict_text = " ".join(generate_paraphrase(text)[0] for text in splited_text)
    return predict_text

def found_entity(to_found, text, entities):
    # Find the entity in the text
    exist = False
    to_found = to_found.lower()
    text = text.lower()
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)
    for i in range(len(text)):
        if text.startswith(to_found, i):
            start = i
            end = i + len(to_found)
            # Check if the found entity can be aligned
            span = doc.char_span(start, end, label="TEST", alignment_mode="strict")
            if span is not None:
                # Check if the entity already exists
                for ent in entities:
                    if (ent[0] == start or ent[1] == end) or (
                        start > ent[0] and end < ent[1]
                    ):
                        exist = True
                if not exist:
                    return start, end
            exist = False
    return None


def duplicate_annotation():
    path = "annotations/"
    model, tokenizer = load_model_paraphrase()
    for json_file in glob.glob(path + "*.json"):
        path_duplicate = (
            path + json_file.split(".")[2].split("/")[2] + "_" + "duplicate" + ".json"
        )
        if "duplicate" not in json_file or not os.path.exists(path_duplicate):
            with open(json_file, "r") as f_json:
                c_json = json.load(f_json)
                text = c_json["annotations"][0][0]
                annotations = c_json["annotations"][0][1]
                ents = get_entities(annotations)
                predict_text = predict_paraphrase(text, model, tokenizer)
                new_entities = [[pos[0], pos[1], ent[1]] for ent in ents if (pos := found_entity(ent[0], predict_text, new_entities)) is not None]
                create_json(path_duplicate, predict_text, new_entities)


# ------------------------------ END IN PROGRESS -----------------------------------#

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.NOTSET,
    )
    if args.clear:
        clear_folder()
    generate_annotation(int(args.threshold), float(args.cutoff))
    if args.duplicate:
        duplicate_annotation()

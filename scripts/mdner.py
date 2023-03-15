import argparse
import os
import glob
import json
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy import displacy
from random import sample


parser = argparse.ArgumentParser(
    description="Create or call a model for the molecular dynamics data."
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-p",
    "--predict",
    help="Call an existing model and extracts the MD information which can be viewed via a link. A path of a text file is required from this file.",
    type=str,
    nargs=1,
)
group.add_argument(
    "-c",
    "--create",
    help="Create a dedicated Named Entity Recognition model for our molecular dynamics data.",
    action="store_true",
)
args = parser.parse_args()

# /!\ add independent dataset for training
def create_data(is_train: bool):
    path = "../annotations/"
    json_files = [file.split("/")[-1] for file in glob.glob(path + "*.json")]
    data = {"classes": [], "annotations": []}
    size = int(len(json_files) * 0.8 if is_train else len(json_files) * 0.2)
    sample_json = sample(json_files, size)
    for json_name in sample_json:
        with open(path + json_name, "r") as f:
            data_json = json.load(f)
            if len(data["classes"]) == 0:
                data["classes"] = data_json["classes"]
            data["annotations"].append(data_json["annotations"][0])
    if is_train:
        print("Train data created !")
    else:
        print("Test data created !")
    return data


def create_spacy_object(data, is_train: bool):
    nlp = spacy.blank("en")  # Load a new spacy model
    db = DocBin()  # Create a DocBin object
    description = "Spacy training object" if is_train else "Spacy testing object"
    annotations = tqdm(data["annotations"], desc=description)
    for text, annot in annotations:
        doc = nlp.make_doc(text)  # Create doc object from text
        ents = []
        for start, end, label in annot["entities"]:  # Add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if not span is None:
                ents.append(span)
        doc.ents = ents  # Label the text with the ents
        db.add(doc)
    if is_train:
        db.to_disk("../results/outputs/train_data.spacy")
    else:
        db.to_disk("../results/outputs/test_data.spacy")


if __name__ == "__main__":
    os.chdir(os.path.split(os.path.abspath(__file__))[0])
    if args.create:
        train_data = create_data(is_train=True)
        test_data = create_data(is_train=False)
        create_spacy_object(train_data, is_train=True)
        create_spacy_object(test_data, is_train=False)
        os.system(
            "python -m spacy init config ../results/outputs/config.cfg --lang en --pipeline ner --optimize efficiency --force"
        )
        os.system(
            "python -m spacy train ../results/outputs/config.cfg --output ../results/models --paths.train ../results/outputs/train_data.spacy --paths.dev ../results/outputs/test_data.spacy"
        )
    elif args.predict:
        nlp_ner = spacy.load("../results/models/model-last")
        # Text processing by our NER
        path_file = args.predict[0]
        if os.path.isfile(path_file):
            with open(path_file, "r") as f:
                text = f.read()
            doc = nlp_ner(text)
            displacy.serve(doc, style="ent")
        else:
            print("[ERROR] Cannot find the path specified.")

"""This script is used to create a model for the molecular dynamics data by using the SpaCy library.

To understand how the model works, please read the documentation of SpaCy in the following link: https://spacy.io/usage/training
"""

import argparse
import subprocess
import os
import logging
import glob
import json
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy import displacy
import random
import re
from datetime import datetime
import pandas as pd
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

# 493
# 1645
# 7522
# 112
random.seed(112)

parser = argparse.ArgumentParser(
    description="Create or call a model for the molecular dynamics data."
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-p",
    "--predict",
    help="Call an existing model and extracts the MD information which can be viewed via HTML file.",
    action="store_true",
)
group.add_argument(
    "-c",
    "--create",
    help="Create a dedicated Named Entity Recognition model for our molecular dynamics data.",
    action="store_true",
)
parser.add_argument(
    "-t",
    "--train",
    help="Hyperparameters for the training process where d is the percentage of dropout. The f, p and r scores define what SpaCy believes to be the best model after the training process.",
    nargs=4,
    type=float,
    metavar=("d", "f", "p", "r"),
)
parser.add_argument("-g", "--gpu", help="Use GPU for training.", action="store_true")
args = parser.parse_args()


# def free_gpu_cache():
#     print("Initial GPU Usage")
#     gpu_usage()

#     torch.cuda.empty_cache()

#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)

#     print("GPU Usage after emptying the cache")
#     gpu_usage()


def create_data():
    """
    Create training, test and evaluation data from the annotations.

    Returns:
    --------
    data: list
        List of dictionaries with the training, test and evaluation data.
    """
    # Setup our training data and test data
    path = "../annotations/"
    json_files = [file.split("/")[-1] for file in glob.glob(path + "*.json")]
    size_train = int(len(json_files) * 0.8)
    size_test = int(len(json_files) * 0.15)
    size_eval = int(len(json_files) * 0.05)
    data = [{"classes": [], "annotations": []} for i in range(3)]
    sample_train = random.sample(json_files, size_train)
    sample_test = random.sample(
        [i for i in json_files if i not in sample_train], size_test
    )
    sample_eval = random.sample(
        [i for i in json_files if i not in sample_train + sample_test], size_eval
    )
    samples = [sample_train, sample_test, sample_eval]
    ignored = 0
    for i in range(len(samples)):
        for json_file in samples[i]:
            with open(path + json_file, "r") as f:
                data_json = json.load(f)
                if data_json["annotations"][0][1]["entities"] != []:
                    if (
                        len(data_json["annotations"][0][1]["entities"])
                        / len(data_json["annotations"][0][0].split())
                        > 0.01
                    ):
                        if len(data[i]["classes"]) == 0:
                            data[i]["classes"] = data_json["classes"]
                        data[i]["annotations"].append(data_json["annotations"][0])
                    else:
                        ignored += 1
    if ignored != 0:
        logging.warning(f"{ignored} files ignored because there are not many entities")
    return data


def create_spacy_object(data, name_file):
    """
    Create a spacy object from the data with the name of the file.

    Parameters:
    ----------
    data: list
        Dictionary of the data (training, test or evaluation).
    name_file: str
        Name of the json file to save the data.
    """
    with open("../results/outputs/" + name_file + ".json", "w") as f:
        json.dump(data, f, indent=None)
    nlp = spacy.blank("en")  # Load a new spacy model
    db = DocBin()  # Create a DocBin object
    description = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}] [INFO] Creating {name_file.split('_')[0]} data object"
    annotations = tqdm(
        data["annotations"],
        desc=description,
        total=len(data["annotations"]),
        bar_format="{l_bar} Size: " + str(len(data["annotations"])),
    )
    for text, annot in annotations:
        doc = nlp.make_doc(text)  # Create doc object from text
        ents = []
        for start, end, label in annot["entities"]:  # Add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        doc.ents = ents  # Label the text with the ents
        db.add(doc)

    db.to_disk(f"../results/outputs/{name_file}.spacy")


def setup_config(d, f, p, r):
    """
    Change parameters in the config file for the training process.

    Parameters:
    ----------
    d: float
        Percentage of dropout.
    f: float
        The f score of the best model you want to achieve.
    p: float
        The p score of the best model you want to achieve.
    r: float
        The r score of the best model you want to achieve.
    """
    old_params = [
        "train = null",
        "dev = null",
        "dropout = 0.1",
        "ents_f = 1.0",
        "ents_p = 0.0",
        "ents_r = 0.0",
        "eval_frequency = 200",
        'vectors = "en_core_web_lg"',
        "batch_size = 1000",
        "init_tok2vec = null",
    ]
    new_params = [
        "train = ../results/outputs/train_data.spacy",
        "dev = ../results/outputs/test_data.spacy",
        f"dropout = {d}",
        f"ents_f = {f}",
        f"ents_p = {p}",
        f"ents_r = {r}",
        "eval_frequency = 200",
        "vectors = null",
        "batch_size = 32",
        'init_tok2vec = "en_core_sci_lg"',
    ]
    with open("../results/outputs/config.cfg", "r+") as f:
        file_contents = f.read()
        for i in range(len(old_params)):
            text_pattern = re.compile(re.escape(old_params[i]), 0)
            file_contents = text_pattern.sub(new_params[i], file_contents)
            f.seek(0)
            f.truncate()
            f.write(file_contents)


def entities_to_csv():
    """Create a csv file with the number of entities per file."""
    path = "../annotations/"
    json_files = [file.split("/")[-1] for file in glob.glob(path + "*.json")]
    to_pandas = []
    for json_file in json_files:
        with open(path + json_file, "r") as f:
            data_json = json.load(f)
            count = {"MOL": 0, "TEMP": 0, "STIME": 0, "SOFT": 0, "FFM": 0}
            for ent in data_json["annotations"][1]["entities"]:
                if ent in count.keys():
                    count[ent] += 1
            to_pandas.append([json_file] + list(count.values()))
    df = pd.DataFrame(
        to_pandas, columns=["JSON FILE", "MOL", "TEMP", "STIME", "SOFT", "FFM"]
    )
    df.to_csv("../results/outputs/entities_train.csv", index=False)


def display_command(command, display=True):
    logging.info(f"Running command: {command}")
    subprocess.run(
        command,
        shell=True,
        stdout=subprocess.DEVNULL if not display else None,
    )


def check_debug(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    pos = text.find("Summary")
    text = ansi_escape.sub("", text[pos:])
    checked = re.findall(r"\d+", text)
    str_checked = ["checks passed", "warnings", "failed"]
    to_print = ""
    for i in range(len(checked)):
        to_print += f"{checked[i]} {str_checked[i]}"
        if i != len(checked) - 1:
            to_print += ", "
    logging.info(to_print)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.NOTSET,
    )
    if args.create and args.train:
        d, f, p, r = args.train
        # Check if data is already created
        json_files = glob.glob("../results/outputs/*.json")
        if len(json_files) != 3:
            # Create data and save it in spacy files and json files
            data = create_data()
            create_spacy_object(data[0], "train_data")
            create_spacy_object(data[1], "test_data")
            create_spacy_object(data[2], "eval_data")
        else:
            logging.info("Data already created")
        # Create config file depending on the GPU availability
        if args.gpu:
            logging.info("Checking GPU availability")
            is_using_gpu = spacy.prefer_gpu()
            if not is_using_gpu:
                logging.error(
                    "GPU is not available. Please check if you have a GPU and if you have installed the correct version of CUDA."
                )
                exit()
            else:
                logging.error(
                    "GPU is not available. Please check if you have a GPU and if you have installed the correct version of CUDA."
                )
                command = "python -m spacy init config ../results/outputs/config.cfg --lang en --pipeline transformer,ner --optimize accuracy -G --force"
                display_command(command, display=False)
        else:
            command = "python -m spacy init config ../results/outputs/config.cfg --lang en --pipeline ner --optimize accuracy --force"
            display_command(command, display=False)
        # Setup config file with the parameters
        setup_config(d=d, f=f, p=p, r=r)
        # Check if the config file is correct
        command = "python -m spacy debug data ../results/outputs/config.cfg"
        logging.info(f"Running command: {command}")
        text = subprocess.run(command, shell=True, capture_output=True, text=True)
        check_debug(text.stdout)
        # Train the model and evaluate it depending on the GPU availability
        if args.gpu:
            command = f"python -m spacy train ../results/outputs/config.cfg --output ../results/models_{d}_{f}_{p}_{r} --gpu-id 0 | tee ../results/outputs/train_{d}_{f}_{p}_{r}.log"
            display_command(command)
            command = f"python -m spacy benchmark accuracy ../results/models_{d}_{f}_{p}_{r}/model-best/ ../results/outputs/eval_data.spacy --gpu-id 0"
            display_command(command)
        else:
            command = f"python -m spacy train ../results/outputs/config.cfg --output ../results/models_{d}_{f}_{p}_{r} | tee ../results/outputs/train_{d}_{f}_{p}_{r}.log"
            display_command(command)
            command = f"python -m spacy benchmark accuracy ../results/models_{d}_{f}_{p}_{r}/model-best/ ../results/outputs/eval_data.spacy"
            display_command(command)
    elif args.predict:
        # Load the model and predict the entities by saving the results in an html file
        nlp_ner = spacy.load("../results/models_0.4_0.0_0.9_0.1/model-best")
        nlp_annotated = spacy.blank("en")
        if os.path.isfile("../results/outputs/eval_data.json"):
            with open("../results/outputs/eval_data.json") as f:
                eval_json = json.load(f)
            colors = {
                "TEMP": "#FF0000",
                "SOFT": "#FFA500",
                "STIME": "#FD6C9E",
                "FFM": "#00FFFF",
                "MOL": "#FFFF00",
            }
            options = {
                "ents": [
                    "TEMP",
                    "SOFT",
                    "STIME",
                    "FFM",
                    "MOL",
                ],
                "colors": colors,
            }
            docs_annotated = []
            docs_ner = []
            html = ""
            for text, annotation in eval_json["annotations"]:
                doc_ner = nlp_ner(text + 2 * "\n")
                example = Example.from_dict(
                    nlp_annotated.make_doc(text + 2 * "\n"), annotation
                )
                html += "<hr>\n<b>Texte annoté par Mohmo:</b>" + 3 * "\n"
                html += displacy.render(example.reference, style="ent", options=options)
                html += "<b>Texte annoté par MDNER:</b>" + 3 * "\n"
                html += displacy.render(doc_ner, style="ent", options=options)
            with open("../results/outputs/html_predict.html", "w") as f:
                f.write(html)
            logging.info("HTML file created")
        else:
            logging.error("Cannot find the evaluation data.")
    else:
        logging.error("Please specify a valid argument.")

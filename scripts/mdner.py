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
from spacy.training.loop import train as train_nlp
from spacy.training.initialize import init_nlp
from spacy.util import load_config

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
parser.add_argument("-n", "--name", help="Name of the model.", type=str)
parser.add_argument("-g", "--gpu", help="Use GPU for training.", action="store_true")
args = parser.parse_args()


def create_data() -> list:
    """
    Create training, test and evaluation data from the annotations.

    Returns:
    --------
    data: list
        List of dictionaries containing training, test and evaluation data.
    """
    # Get all json files
    path = "annotations/"
    json_files = [file.split("/")[-1] for file in glob.glob(path + "*.json")]
    # Split the data into training, test and evaluation data
    size_train = int(len(json_files) * 0.75)
    size_test = int(len(json_files) * 0.15)
    size_eval = int(len(json_files) * 0.10)
    sample_train = random.sample(json_files, size_train)
    sample_test = random.sample(
        [i for i in json_files if i not in sample_train], size_test
    )
    sample_eval = random.sample(
        [i for i in json_files if i not in sample_train + sample_test], size_eval
    )
    samples = [sample_train, sample_test, sample_eval]
    data = [{"classes": [], "annotations": []} for i in range(3)]
    ignored = 0
    # Read each json file and check if contains enough entities
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
                        # to_keep = []
                        # for entity in data_json["annotations"][0][1]["entities"]:
                        #     if entity[2] == "MOL" :
                        #         to_keep.append(entity)
                        # data_json["annotations"][0][1]["entities"] = to_keep
                        data[i]["annotations"].append(data_json["annotations"][0])
                    else:
                        ignored += 1
    if ignored != 0:
        logging.warning(f"{ignored} files ignored because there are not many entities")
    return data


def create_spacy_object(data: dict, name_file: str, name_model: str):
    """
    Create a spacy object from the data with the name of the file.

    Parameters:
    ----------
    data: dict
        Dictionary of the data (training, test or evaluation).
    name_file: str
        Name of the json file to save the data.
    name_model: str
        Name of the model to save the spacy object.
    """
    with open("results/outputs/" + name_file + ".json", "w") as f:
        json.dump(data, f, indent=None)
    # Load a new spacy model
    nlp = spacy.blank("en")
    # Create a DocBin object will be used to create a .spacy file
    db = DocBin()
    # Config the tqdm bar to display
    description = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}] [INFO] Creating {name_file.split('_')[0]} data object"
    annotations = tqdm(
        data["annotations"],
        desc=description,
        total=len(data["annotations"]),
        bar_format="{l_bar} Size: " + str(len(data["annotations"])),
    )
    # Read each annotation and check if the entity is a valid entity
    for text, annot in annotations:
        # Create doc object from text
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            # Align the character spans to the tokens
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        # Label the text with the ents
        doc.ents = ents
        db.add(doc)
    # Save the DocBin object
    db.to_disk(f"results/outputs/{name_file}.spacy")
    if not os.path.exists(f"results/models/{name_model}"):
        os.makedirs(f"results/models/{name_model}")
    db.to_disk(f"results/models/{name_model}/{name_file}.spacy")


def setup_config(d: float, f: float, p: float, r: float):
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
        'name = "roberta-base"' if args.gpu else "init_tok2vec = null",
    ]
    new_params = [
        "train = results/outputs/train_data.spacy",
        "dev = results/outputs/test_data.spacy",
        f"dropout = {d}",
        f"ents_f = {f}",
        f"ents_p = {p}",
        f"ents_r = {r}",
        "eval_frequency = 200",
        "vectors = null",
        "batch_size = 32",
        'name = "allenai/biomed_roberta_base"'
        if args.gpu
        else 'init_tok2vec = "en_core_sci_lg"',
    ]
    # Change the parameters in the config file
    with open("results/outputs/config.cfg", "r+") as f:
        file_contents = f.read()
        for i in range(len(old_params)):
            text_pattern = re.compile(re.escape(old_params[i]), 0)
            file_contents = text_pattern.sub(new_params[i], file_contents)
            f.seek(0)
            f.truncate()
            f.write(file_contents)


def entities_to_csv():
    """Create a csv file with the number of entities per file."""
    # Get the json files
    path = "annotations/"
    json_files = [file.split("/")[-1] for file in glob.glob(path + "*.json")]
    to_pandas = []
    # Read each json file and count the number of entities
    for json_file in json_files:
        with open(path + json_file, "r") as f:
            data_json = json.load(f)
            count = {"MOL": 0, "TEMP": 0, "STIME": 0, "SOFT": 0, "FFM": 0}
            for ent in data_json["annotations"][1]["entities"]:
                if ent in count.keys():
                    count[ent] += 1
            to_pandas.append([json_file] + list(count.values()))
    # Save the data in a csv file
    df = pd.DataFrame(
        to_pandas, columns=["JSON FILE", "MOL", "TEMP", "STIME", "SOFT", "FFM"]
    )
    df.to_csv("results/outputs/entities_train.csv", index=False)


def display_command(command: str, display: bool = True):
    """
    Display the command and run it.

    Parameters:
    ----------
    command: str
        Command to run.
    display: bool
        If True, the output of the command will be displayed.
    """
    logging.info(f"Running command: {command}")
    subprocess.run(
        command,
        shell=True,
        stdout=subprocess.DEVNULL if not display else None,
    )


def check_debug(output_command: str) -> bool:
    """
    Extract the number of checks passed, warnings and failed.

    Execute the command and check the output of the command to see if there
    are any errors.

    Parameters:
    ----------
    output_command: str
        Output of the command.

    Returns:
    -------
    bool
        True if the number of failed checks is greater than 0, False otherwise.
    """
    # Delete of ANSI codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    pos = output_command.find("Summary")
    output_command = ansi_escape.sub("", output_command[pos:])
    # Get the number of checks passed, warnings and failed
    checked = re.findall(r"\d+", output_command)
    str_checked = ["checks passed", "warnings", "failed"]
    # Create a string with the results to print
    to_print = ", ".join(
        [f"{checked[i]} {str_checked[i]}" for i in range(len(checked))]
    )
    logging.info(to_print)
    return checked[2] > 0 if len(checked) > 2 else False


def generate_html():
    """Generate the html file with the evaluation of the model."""
    # Load the best model and an empty model
    nlp_ner = spacy.load("results/colab/only_mol/models/model-best")
    nlp_annotated = spacy.blank("en")
    # Check if evaluation data exists
    if os.path.isfile("results/outputs/eval_data.json"):
        # Load the evaluation data
        with open("results/outputs/eval_data.json") as f:
            eval_json = json.load(f)
        # Define the colors for the entities
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
        # Generate the html file with the annotations of the model and the annotations of the annotator
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
        # Save the html file
        with open("results/outputs/html_predict.html", "w") as f:
            f.write(html)
        logging.info("HTML file created")
    else:
        logging.error("Cannot find the evaluation data.")


def generate_data(name_model: str):
    """
    Generate train, test and evaluation data in json files and spacy files.

    Parameters:
    ----------
    name_model: str
        Name of the model to use.
    """
    json_files = glob.glob("results/outputs/*.json")
    # Check if data is already created
    if len(json_files) != 3:
        # Create data and save it in spacy files and json files
        data = create_data()
        create_spacy_object(data[0], "train_data", name_model)
        create_spacy_object(data[1], "test_data", name_model)
        create_spacy_object(data[2], "eval_data", name_model)
    else:
        logging.info("Data already created")


def check_gpu():
    """
    Check if GPU is available.

    Returns:
    -------
    bool
        True if GPU is available, False otherwise.
    """
    logging.info("Checking GPU availability")
    is_using_gpu = spacy.prefer_gpu()
    if not is_using_gpu:
        logging.error(
            "GPU is not available. Please check if you have a GPU and if you have installed the correct version of CUDA."
        )
        return False
    else:
        logging.info("GPU is available")
        return True


def debug_config():
    """Run the command to debug the config file and check the output results."""
    command = "python -m spacy debug data results/outputs/config.cfg"
    logging.info(f"Running command: {command}")
    output_command = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(output_command.stdout)
    # have_error = check_debug(output_command.stdout)
    # if have_error:
    #     logging.error(f"A error has been detected :\n{output_command.stdout}")
    #     exit(1)


def create_config(option_gpu: bool):
    """
    Create the config file depending if GPU is available or not.

    Parameters:
    ----------
    option_gpu: bool
        True if GPU is available, False otherwise.
    """
    if option_gpu:
        have_gpu = check_gpu()
        if have_gpu:
            command = "python -m spacy init config results/outputs/config.cfg --lang en --pipeline transformer,ner --optimize accuracy -G --force"
            display_command(command, display=False)
        else:
            exit(1)
    else:
        command = "python -m spacy init config results/outputs/config.cfg --lang en --pipeline ner --optimize accuracy --force"
        display_command(command, display=False)


def training_process(
    option_gpu: bool, d: float, f: float, p: float, r: float, name: str
):
    """
    Execute the commands to train the model and evaluate it.

    Parameters:
    ----------
    option_gpu: bool
        True if GPU is available, False otherwise.
    d: float
        The dropout rate.
    f: float
        The f score of the best model you want to achieve.
    p: float
        The p score of the best model you want to achieve.
    r: float
        The r score of the best model you want to achieve.
    name: str
        The name of the model.
    """
    command = f"python -m spacy train results/outputs/config.cfg --output results/models/{name} {'--gpu-id 0' if option_gpu else ''} | tee results/models/{name}/train.log"
    display_command(command)

    # config = load_config("results/outputs/config.cfg")
    # nlp = init_nlp(config)

    # file = open("results/CID-Synonym-filtered")
    # i = 0
    # patterns = []
    # for line in file:
    #     mol = line.split()[1]
    #     if mol:
    #         patterns.append({"label": "MOL", "pattern": mol})
    #     i += 1
    #     if i == 100000:
    #         break
    # file.close()
    # entity_ruler = nlp.add_pipe("entity_ruler", after="ner")
    # entity_ruler.add_patterns(patterns)
    # nlp, _ = train_nlp(nlp, None)

    command = f"python -m spacy benchmark accuracy results/models/{name}/model-best/ results/outputs/eval_data.spacy {'--gpu-id 0' if option_gpu else ''}"
    display_command(command)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.NOTSET,
    )
    if args.create and args.train and args.name:
        d, f, p, r = args.train
        generate_data(args.name)
        # Create config file depending on the GPU availability
        create_config(args.gpu)
        # Change parameters in the config file depending on the arguments
        setup_config(d=d, f=f, p=p, r=r)
        # Check if the config file is correct
        debug_config()
        # Train the model and evaluate it depending on the GPU availability
        training_process(args.gpu, d, f, p, r, args.name)
    elif args.predict:
        generate_html()
    else:
        logging.error("Please specify a valid argument.")

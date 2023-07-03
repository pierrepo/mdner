"""This script is used to create a model for the molecular dynamics data by using the SpaCy library.

To understand how the model has been built and works, please read the
documentation of SpaCy in the following link: https://spacy.io/usage/training
"""

import argparse
import subprocess
import logging
import glob
import json
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
import random
import re
from datetime import datetime


parser = argparse.ArgumentParser(
    description="Create a model for the molecular dynamics data.",
)
parser.add_argument(
    "-c",
    "--create",
    help="Create a dedicated Named Entity Recognition model for our molecular dynamics data.",
    action="store_true",
)
parser.add_argument(
    "-t",
    "--train",
    help="Hyperparameters for the learning process where d is the percentage of dropout. The f, p and r scores define what SpaCy believes to be the best model after the learning process.",
    nargs=4,
    type=float,
    metavar=("d", "f", "p", "r"),
)
parser.add_argument("-n", "--name", help="Name of the model.", type=str)
parser.add_argument("-g", "--gpu", help="Use GPU for learning.", action="store_true")
parser.add_argument(
    "-p",
    "--paraphrase",
    help="Add paraphrase in the learning dataset.",
    action="store_true",
)
parser.add_argument("-m", "--mol", help="Use only MOL entities.", action="store_true")
parser.add_argument(
    "-s", "--seed", help="Seed used to sample data sets for reproducibility.", type=int, default=42
)
args = parser.parse_args()
random.seed(args.seed)


def get_files() -> list:
    """Find all the files in the annotations folder."""
    path = "annotations/"
    all_files = [file.split("/")[-1] for file in glob.glob(path + "*.json")]
    paraphrase_files = [file for file in all_files if file.count("_") == 2]
    references_files = [file for file in all_files if file not in paraphrase_files]
    return references_files, paraphrase_files


def create_eval_data(references_files: list) -> tuple:
    """
    Create the evaluation dataset by using 10% of the references files.

    Parameters
    ----------
    references_files : list
        List of the references files.

    Returns
    -------
    tuple
        Tuple with the references files and the paraphrase files.
    """
    size_eval = int(len(references_files) * 0.10)
    sample_eval = random.sample(references_files, size_eval)
    for model in ["mbart", "pegasus", "bart-paraphrase"]:
        sample_eval_paraphrase = [
            file.replace(".json", f"_{model}.json") for file in sample_eval
        ]
    return sample_eval, sample_eval_paraphrase


def add_paraphrase_in_data(
    references_files: list,
    paraphrase_files: list,
    sample_eval: list,
    sample_eval_paraphrase: list,
    add_paraphrase: bool,
) -> list:
    """
    Add paraphrase in the training and test dataset.

    Parameters
    ----------
    references_files : list
        List of the references files.
    paraphrase_files : list
        List of the paraphrase files.
    sample_eval : list
        List of the references files used for the evaluation.
    sample_eval_paraphrase : list
        List of the paraphrase files used for the evaluation.
    add_paraphrase : bool
        If True, add paraphrase in the training dataset.

    Returns
    -------
    list
        List of the files used for the training and test dataset.
    """
    references_files = [file for file in references_files if file not in sample_eval]
    paraphrase_files = [
        file for file in paraphrase_files if file not in sample_eval_paraphrase
    ]
    if add_paraphrase:
        logging.info("Add paraphrase in the training dataset")
        learn_files = references_files + paraphrase_files
    else:
        learn_files = references_files
    return learn_files


def create_learn_data(learn_files: list) -> tuple:
    """
    Create the training and test dataset from the learn files.

    Parameters
    ----------
    learn_files : list
        List of the files used for the training and test dataset.

    Returns
    -------
    tuple
        Tuple with the training and test dataset.
    """
    size_train = int(len(learn_files) * 0.80)
    size_test = int(len(learn_files) * 0.20)
    sample_train = random.sample(learn_files, size_train)
    sample_test = random.sample(
        [file for file in learn_files if file not in sample_train], size_test
    )
    return sample_train, sample_test


def extract_mol_entities(data_json: dict) -> list:
    """
    Extract the MOL entities from the json file.

    Parameters
    ----------
    data_json : dict
        Content of the json file.

    Returns
    -------
    list
        List of the MOL entities.
    """
    to_keep = []
    for entity in data_json["annotations"][0][1]["entities"]:
        if entity[2] == "MOL":
            to_keep.append(entity)
    return to_keep


def data_selection(samples: list, only_mol: bool) -> list:
    """
    Filter the data by keeping a maximum of 1% of entities or only MOL entities.

    Parameters
    ----------
    samples : list
        List of the samples.
    only_mol : bool
        If True, keep only MOL entities.

    Returns
    -------
    list
        List of the samples with the filtered data.
    """
    path = "annotations/"
    data = [{"classes": [], "annotations": []} for i in range(3)]
    ignored = 0
    # Read each json file
    for i in range(len(samples)):
        for json_file in samples[i]:
            with open(path + json_file, "r") as f:
                data_json = json.load(f)
                # Check if there are entities
                if data_json["annotations"][0][1]["entities"] != []:
                    # Check if there are many entities
                    if (
                        len(data_json["annotations"][0][1]["entities"])
                        / len(data_json["annotations"][0][0].split())
                        > 0.01
                    ):
                        if len(data[i]["classes"]) == 0:
                            data[i]["classes"] = data_json["classes"]
                        # If only_mol is True, keep only MOL entities
                        if only_mol:
                            to_keep = extract_mol_entities(data_json)
                            data_json["annotations"][0][1]["entities"] = to_keep
                        # Add annotations in the data
                        data[i]["annotations"].append(data_json["annotations"][0])
                    else:
                        ignored += 1
    if ignored != 0:
        logging.warning(f"{ignored} files ignored because there are not many entities")
    if only_mol:
        logging.info("Create data with only MOL entities")
    return data


def create_data(add_paraphrase: bool, only_mol: bool) -> list:
    """
    Create training, test and evaluation data from the annotations.

    Parameters:
    -----------
    add_paraphrase: bool
        If True, add paraphrase in the training dataset.
    only_mol: bool
        If True, use only MOL entities.

    Returns:
    --------
    data: list
        List of dictionaries containing training, test and evaluation data.
    """
    # Get all the files in the annotations folder and split them in references and paraphrase
    references_files, paraphrase_files = get_files()
    # Create evaluation data
    sample_eval, sample_eval_paraphrase = create_eval_data(references_files)
    # Create files for training and test data and add paraphrase if needed
    learn_files = add_paraphrase_in_data(
        references_files,
        paraphrase_files,
        sample_eval,
        sample_eval_paraphrase,
        add_paraphrase,
    )
    # Create training and test data
    sample_train, sample_test = create_learn_data(learn_files)
    # Join all the samples in a list
    samples = [sample_train, sample_test, sample_eval]
    # Select annotations with enough entities or only MOL entities if needed
    data = data_selection(samples, only_mol)
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
    # Load a blank spacy model
    nlp = spacy.blank("en")
    # Create a DocBin object will be used to create a .spacy file
    db = DocBin()
    # Config the tqdm bar to display the progress
    description = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}] [INFO]"
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
        try:
            doc.ents = ents
            db.add(doc)
        except Exception:
            pass
    # Save the DocBin object
    db.to_disk(f"results/models/{name_model}/{name_file}.spacy")


def setup_config(d: float, f: float, p: float, r: float, model_name: str):
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
    model_name: str
        Name of the model used for the training.
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
        f"train = results/models/{model_name}/train_data.spacy",
        f"dev = results/models/{model_name}/test_data.spacy",
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


def generate_data(name_model: str, add_paraphrase: bool, only_mol: bool, seed: int):
    """
    Generate train, test and evaluation data from the annotated data.

    Parameters:
    ----------
    name_model: str
        Name of the model to use.
    add_paraphrase: bool
        If True, the paraphrase data will be added to the train data.
    only_mol: bool
        If True, only the molecules will be extracted from the data.
    """
    logging.info("Seed: " + str(seed))
    names_file = ["train_data", "test_data", "eval_data"]
    # Create data and save it in spacy files and json files
    data = create_data(add_paraphrase, only_mol)
    for i, name_file in enumerate(names_file):
        create_spacy_object(data[i], name_file, name_model)


def check_gpu() -> bool:
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
        # Create the config file
        command = "python -m spacy init config results/outputs/config.cfg --lang en --pipeline ner --optimize accuracy --force"
        display_command(command, display=False)


def training_process(option_gpu: bool, name: str):
    """
    Execute the commands to train the model and evaluate it.

    Parameters:
    ----------
    option_gpu: bool
        True if GPU is available, False otherwise.
    name: str
        The name of the model.
    """
    # Create the train.log file
    command = f"touch results/models/{name}/train.log"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
    # Train the model
    command = f"python -m spacy train results/outputs/config.cfg --output results/models/{name} {'--gpu-id 0' if option_gpu else ''} | tee results/models/{name}/train.log"
    display_command(command)
    # Evaluate the model
    command = f"python -m spacy benchmark accuracy results/models/{name}/model-best/ results/models/{name}/eval_data.spacy {'--gpu-id 0' if option_gpu else ''}"
    display_command(command)


def create_folder(name: str):
    """
    Create a folder if it does not exist.

    Parameters:
    ----------
    name: str
        The name of the folder to create.
    """
    command = f"mkdir results/models/{name}"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)


def remove_folder(name: str):
    """
    Remove a folder if it exists.

    Parameters:
    ----------
    name: str
        The name of the folder to remove.
    """
    command = f"rm -rf results/models/{name}"
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.NOTSET,
    )
    if args.create and args.train and args.name:
        try:
            create_folder(args.name)
            d, f, p, r = args.train
            generate_data(args.name, args.paraphrase, args.mol, args.seed)
            # Create config file depending on the GPU availability
            create_config(args.gpu)
            # Change parameters in the config file depending on the arguments
            setup_config(d=d, f=f, p=p, r=r, model_name=args.name)
            # Check if the config file is correct
            debug_config()
            # Train the model and evaluate it depending on the GPU availability
            training_process(args.gpu, args.name)
        except KeyboardInterrupt or Exception:
            logging.error("Process interrupted by the user.")
            remove_folder(args.name)
    else:
        logging.error("Please specify a valid argument.")

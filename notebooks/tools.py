import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import spacy
import json

def get_scores(paths_log: list) -> list:
    # Define metrics to plot
    metrics = ["ENTS_F", "ENTS_P", "ENTS_R"]

    # Define lists to store scores, loss values and epochs
    results = []

    # Loop over paths to extract metrics and loss values
    for path in paths_log:
        with open(path, "r") as f:
            scores = {metric: [] for metric in metrics}
            loss_validation = []
            epoch = []
            for line in f:
                if line.strip()[0].isdigit():
                    epoch.append(int(line.split()[0]))
                    scores["ENTS_F"].append(float(line.split()[-4]))
                    scores["ENTS_P"].append(float(line.split()[-3]))
                    scores["ENTS_R"].append(float(line.split()[-2]))
                    loss_validation.append(float(line.split()[3]))
            results.append(
                {"epoch": epoch, "scores": scores, "loss_validation": loss_validation}
            )
    return results


def display_plots(results: list, paths_log: list):
    # Create plots for metrics and loss evolution
    _, axs = plt.subplots(len(paths_log), 2, figsize=(12, 10))

    for i, result in enumerate(results):
        # Create a plot for metrics evolution
        df_scores = pd.DataFrame.from_dict(result["scores"])
        df_scores["epoch"] = result["epoch"]
        sns.lineplot(
            x="epoch",
            y="value",
            hue="variable",
            data=pd.melt(df_scores, ["epoch"]),
            ax=axs[0][i],
        )
        axs[0][i].set_title("Metrics evolution of " + paths_log[i].split("/")[-2])
        axs[0][i].set_xlabel("Epoch")
        axs[0][i].set_ylabel("Scores (%)")
        axs[0][i].set_ylim(top=100)

        # Create a plot for loss evolution
        df_loss = pd.DataFrame(
            {"epoch": result["epoch"], "loss_validation": result["loss_validation"]}
        )
        sns.lineplot(x="epoch", y="loss_validation", data=df_loss, ax=axs[1][i])
        axs[1][i].set_title("Loss evolution of " + paths_log[i].split("/")[-2])
        axs[1][i].set_xlabel("Epoch")
        axs[1][i].set_ylabel("Loss value")


def get_entities(doc, only_mol):
    ents = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    if only_mol:
        ents = [ent for ent in ents if ent[3] == "MOL"]
    return ents


def get_confusion_matrix(path_model, only_mol):
    ner = spacy.load(f"{path_model}/model-best/")
    with open(f"{path_model}/eval_data.spacy", "rb") as f:
        doc_bin = spacy.tokens.DocBin().from_bytes(f.read())
    tp, fp, fn, tn = 0, 0, 0, 0
    for doc in doc_bin.get_docs(ner.vocab):
        pred_ents = get_entities(ner(doc.text), only_mol)
        true_ents = get_entities(doc, only_mol)
        tp += len([ent for ent in true_ents if ent in pred_ents])
        fp += len([ent for ent in pred_ents if ent not in true_ents])
        fn += len([ent for ent in true_ents if ent not in pred_ents])
        tn += len([ent for ent in pred_ents if ent in true_ents])
    confusion_matrix = pd.DataFrame(
        [[tp, fp], [fn, tn]],
        index=["Positive", "Negative"],
        columns=["Positive", "Negative"],
    )
    acc = (tp + tn) / (tp + fp + fn + tn) * 100
    p = tp / (tp + fp) * 100
    r = tp / (tp + fn) * 100
    return confusion_matrix, [acc, p, r]


def display_confusion_matrix(cf_mtx: dict, scores: dict, paths_model: list):
    # Display confusion matrix
    _, axs = plt.subplots(1, len(paths_model), figsize=(12, 3))
    for i, path in enumerate(paths_model):
        sns.heatmap(cf_mtx[path], annot=True, cmap="Blues", fmt="d", ax=axs[i])
        axs[i].set_xlabel("Annotated")
        axs[i].set_ylabel("Predict")
        axs[i].set_title("Confusion matrix of " + path.split("/")[-1], y=-0.25)
        axs[i].tick_params(
            axis="both",
            which="major",
            labelsize=10,
            labelbottom=False,
            bottom=False,
            top=False,
            labeltop=True,
        )
        axs[i].xaxis.set_label_position("top")
        print_score = f"Accuracy: {round(scores[path][0], 2)}\nPrecision: {round(scores[path][1], 2)}\nRecall: {round(scores[path][2], 2)}"
        axs[i].text(
            0.75,
            3.1,
            print_score,
            style="italic",
            bbox={"facecolor": "blue", "alpha": 0.5, "pad": 10},
        )
        
def get_content(path_json, to_reject: list):
    is_readable = all(pattern not in path_json.split("/")[-1] for pattern in to_reject)
    path_paraphrase = path_json[:-5] + "_" + "paraphrase" + ".json"
    if is_readable:
        with open(path_json, "r") as f_ref, open(
            path_paraphrase, "r"
        ) as f_paraphrase :
            r_json = json.load(f_ref)
            p_json = json.load(f_paraphrase)
            r_txt = r_json["annotations"][0][0]
            p_txt = p_json["annotations"][0][0]
            r_annotations = r_json["annotations"][0][1]
            p_annotations = p_json["annotations"][0][1]
            return r_txt, p_txt, r_annotations, p_annotations
    else:
        return None
            

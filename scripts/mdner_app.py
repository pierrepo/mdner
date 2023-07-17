"""This script is used to annotate the text with the best model.

The best model is loaded and applied to the text. The entities are visualized
in html using Streamlit. To understand how the model works, it is recommended
to read the documentation of the model.
"""

import streamlit as st
import spacy
from spacy import displacy
import os
import argparse
import json
from datetime import date

parser = argparse.ArgumentParser(description="MDner - Streamlit")
parser.add_argument("-m", "--model", dest="model", help="Model name", required=True)
args = parser.parse_args()


def save_to_json(doc: object, text: str):
    """
    Save annotated text in json format.

    Parameters
    ----------
    doc : object
        doc obtained after annotation by the ner model.
    text : str
        the annotated text.
    """
    # Get named entities and their labels
    entities = []
    for ent in doc.ents:
        entities.append([ent.start_char, ent.end_char, ent.label_])
    # Define the annotations in the desired format
    annotations = [
        [
            text,
            {
                "entities": entities
            }
        ]
    ]
    # Create a dictionary to represent the JSON structure
    to_json = {
        "classes": ["TEMP", "SOFT", "STIME", "MOL", "FFM"],
        "annotations": annotations
    }
    # Export in JSON format
    json_file = json.dumps(to_json)
    # Get the current date
    current_date = date.today().strftime("%Y-%m-%d")
    # Create the file name with the current date
    filename = f"annotations_{current_date}.json"
    # Write the JSON to a file
    with open(f"annotations/{filename}", "w") as file:
        file.write(json_file)


def use_model(model):
    """
    Use the model to annotate the text by using Streamlit.

    Parameters
    ----------
    model : str
        Model name.
    """
    st.set_page_config(page_title="MDNER", layout="wide")
    streamlit_style = """
		<style>
			html, body {
			    font-family: 'Roboto', sans-serif;
			}
		</style>
	"""
    st.markdown(streamlit_style, unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; color: dodgerBlue;'>📑 MDNER 🧬</h1>",
        unsafe_allow_html=True,
    )
    text = st.text_area("Text to annotate", height=300)
    run_button, export_button, _ = st.columns([1, 1, 10])
    with run_button:
        apply = st.button("Run")
        if apply:
            with export_button:
                export = st.button("Export to json")
            # Load the best model
            nlp = spacy.load(f"results/models/{model}/model-best")
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
            # Apply the model to the text
            doc = nlp(text)
            # Visualize the entities in html
            html = displacy.render(doc, style="ent", options=options)
            st.divider()
            st.write(html, unsafe_allow_html=True)
            st.divider()
            if export:
                save_to_json(doc, text)


if __name__ == "__main__":
    if args.model:
        model = args.model
        path_model = f"results/models/{model}"
        # Check if the model exists
        if os.path.exists(path_model) and os.path.isdir(path_model):
            use_model(model)
        else:
            st.error("Error: The model does not exist.")
    else:
        st.error("Error: The model is required.")

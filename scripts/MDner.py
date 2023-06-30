"""This script is used to annotate the text with the best model.

The best model is loaded and applied to the text. The entities are visualized in html using Streamlit. To understand how the model works, it is recommended to read the documentation of the model.
"""

import streamlit as st
import spacy
from spacy import displacy
import os
import argparse

parser = argparse.ArgumentParser(description="MDner - Streamlit")
parser.add_argument("-m", "--model", dest="model", help="Model name", required=True)
args = parser.parse_args()

def use_model(model):
    """
    Use the model to annotate the text by using Streamlit.
    
    Parameters
    ----------
    model : str
        Model name.
    """
    st.set_page_config(page_title="MDner - Streamlit", layout="wide")
    st.title("MDner - Streamlit")
    text = st.text_area("Text to annotate", height=300)
    apply = st.button("Run")
    if apply:
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
        st.write(html, unsafe_allow_html=True)


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

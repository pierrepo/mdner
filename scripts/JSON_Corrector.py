import streamlit as st
import json
import spacy
from spacy.training import Example
import glob
import re


def display_ner(data_json, json_edit):
    colors = {
        "TEMPERATURE": "#FF0000",
        "SOFTWARE": "#FFA500",
        "SIMULATION TIME": "#FD6C9E",
        "MODEL": "#00FFFF",
        "MOLECULE": "#FFFF00",
    }
    options = {
        "ents": ["TEMPERATURE", "SOFTWARE", "SIMULATION TIME", "MODEL", "MOLECULE"],
        "colors": colors,
    }
    nlp = spacy.blank("en")
    text, _ = data_json["annotations"][0]
    example = Example.from_dict(nlp.make_doc(text), json.loads(json_edit))
    ent_html = spacy.displacy.render(
        example.reference, style="ent", jupyter=False, options=options
    )
    st.markdown(ent_html, unsafe_allow_html=True)


def display_editor(data_json):
    col_editor, col_display = st.columns([1, 1])
    with col_editor:
        ent_json = json.dumps(data_json["annotations"][0][1], indent=4)
        new_json = st.text_area(
            "JSON Editor",
            ent_json,
            height=600,
        )
        with col_display:
            try:
                display_ner(data_json, new_json)
            except Exception:
                st.error("This json is not well written !", icon="🚨")
                return ent_json, False
    return new_json, new_json != ent_json


def load_css():
    st.markdown(
        """
                <style> 
                    .block-container:first-of-type {
                        padding-top: 20px;
                        padding-left: 20px;
                        padding-right: 20px;
                    }
                </style>
                """,
        unsafe_allow_html=True,
    )


def user_interaction():
    st.set_page_config(page_title="JSON Corrector", layout="wide")
    load_css()
    st.title("JSON Corrector")
    path = "../annotations/"
    files = [file.split("/")[-1].split(".")[0] for file in glob.glob(path + "*.json")]
    if files:
        st.sidebar.title("Filters")
        select_json = st.sidebar.slider(
            "Choose a JSON file to correct:", 1, len(files), 1
        )
        search_json = st.sidebar.text_input(
            "Enter a JSON file name:", placeholder="Example : zenodo_838635"
        )
        if search_json:
            path_name = "../annotations/" + search_json
        else:
            path_name = "../annotations/" + files[select_json - 1] + ".json"
        f = open(path_name, "r")
        data_json = json.load(f)
        f.close()
        new_json, edited = display_editor(data_json)
        save = st.button("Save", disabled=not edited)
        if save:
            f = open(path_name, "w")
            ent_save = re.sub("\n| ", "", new_json)
            data_json["annotations"][0][1] = json.loads(ent_save)
            save_json = json.dumps(data_json, indent=None)
            f.write(save_json)
            f.close()
            st.success("JSON file saved !", icon="✅")
    else:
        st.error("No JSON file found !", icon="🚨")


if __name__ == "__main__":
    user_interaction()

import streamlit as st
import json
import spacy
from spacy.training import Example
import re
import glob, os


def display_ner(data_json, json_edit):
    colors = {
        "TEMPERATURE": "#FF0000",
        "SOFTWARE": "#FFA500",
        "SIMULATION TIME": "#FD6C9E",
        "FF & MODEL": "#00FFFF",
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


def display_text(name_file):
    with st.expander("See the text corresponding to the JSON file"):
        with open(f"../annotations/{name_file.split('.')[0]}.txt", "r") as f:
            text = f.read()
            st.write(text)


def display_editor(name_file, data_json):
    display_text(name_file)
    col_editor, col_display = st.columns([1, 2])
    with col_editor:
        ent_json = json.dumps(data_json["annotations"][0][1], indent=4)
        new_json = st.text_area(
            "JSON Editor",
            ent_json,
            height=600,
        )
    with col_display:
        try:
            st.markdown(
                f"<p class='font-label'>JSON preview : {name_file}</p>",
                unsafe_allow_html=True,
            )
            display_ner(data_json, new_json)
        except Exception:
            st.error("This json is not well written !", icon="🚨")
            return ent_json, False, col_editor
    return new_json, new_json != ent_json, col_editor


def display_filters(files):
    if len(files) > 1:
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
    else:
        path_name = "../annotations/" + files[0] + ".json"
    return path_name


def load_css():
    st.markdown(
        """
                <style>
                    .block-container:first-of-type {
                        padding-top: 20px;
                        padding-left: 20px;
                        padding-right: 20px;
                    }
                    .font-label {
                        font-size: 14px !important;
                    }
                </style>
                """,
        unsafe_allow_html=True,
    )


def save_json(data_json, new_json, path_name, edited):
    save = st.button("Save", disabled=not edited)
    if save:
        f = open(path_name, "w")
        ent_save = re.sub("\n| ", "", new_json)
        data_json["annotations"][0][1] = json.loads(ent_save)
        save_json = json.dumps(data_json, indent=None)
        f.write(save_json)
        f.close()
        st.success("JSON file saved !", icon="✅")


def user_interaction():
    st.set_page_config(page_title="JSON Corrector", layout="wide")
    load_css()
    st.title("JSON Corrector")
    os.chdir(os.path.split(os.path.abspath(__file__))[0])
    path = "../annotations/"
    files = [file.split("/")[-1].split(".")[0] for file in glob.glob(path + "*.json")]
    if files:
        path_name = display_filters(files)
        f = open(path_name, "r")
        data_json = json.load(f)
        f.close()
        new_json, edited, col_editor = display_editor(
            path_name.split("/")[-1], data_json
        )
        with col_editor:
            save_json(data_json, new_json, path_name, edited)
    else:
        st.error("No JSON file found !", icon="🚨")


if __name__ == "__main__":
    user_interaction()

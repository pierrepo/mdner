"""Streamlit tool to correct json files in the annotations folder."""

import streamlit as st
import json
import spacy
from spacy.training import Example
import re
import glob
import os


def display_ner(data_json: dict, json_edit: dict):
    """
    Visualizing the entity recognizer of the edited json.

    Parameters
    ----------
    data_json: dict
        The original json file.
    json_edit: dict
        The edited json file.
    """
    # Colors for the entities
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
    # Display the entities
    nlp = spacy.blank("en")
    text, _ = data_json["annotations"][0]
    example = Example.from_dict(nlp.make_doc(text), json.loads(json_edit))
    ent_html = spacy.displacy.render(
        example.reference, style="ent", jupyter=False, options=options
    )
    st.markdown(ent_html, unsafe_allow_html=True)


def found_entity(to_found, text, entities):
    """
    Find the entity in the text.

    Parameters
    ----------
    to_found: str
        The entity to add.
    text: str
        The text of the json file.
    entities: list
        The entities of the json file.

    Returns
    -------
    tuple
        The start and end positions of the entity.
    """
    # Find the entity in the text
    for found in re.finditer(to_found, text):
        # If the entity is found
        if not [
            found.start(),
            found.end(),
        ] in [[elm[0], elm[1]] for elm in entities]:
            # Add the entity to the json file
            return found.start(), found.end()
    return None


def display_add_entity(data_json):
    col_input, col_select, col_send = st.columns([1, 1, 1])
    with col_input:
        to_found = st.text_input(
            "",
            placeholder="To add an entity, enter the text",
            label_visibility="collapsed",
        )
    with col_select:
        ent = st.selectbox(
            "",
            data_json["classes"],
            index=0,
            key="select_entity",
            label_visibility="collapsed",
        )
    with col_send:
        add = st.button("Add")
    if add:
        positions = found_entity(
            to_found,
            data_json["annotations"][0][0],
            data_json["annotations"][0][1]["entities"],
        )
        return [positions[0], positions[1], ent] if positions else None


def have_text(name_file, col_msg):
    """
    Display an error message if the text file does not exist.

    Parameters
    ----------
    name_file: str
        The name of the json file.
    col_msg: streamlit.columns
        The column where the message will be displayed.
    """
    with col_msg:
        if not os.path.exists(f"../annotations/{name_file.split('.')[0]}.txt"):
            st.error(
                f"The text corresponding to the JSON file {name_file} does not exist !",
                icon="🚨",
            )


def display_editor(name_file, data_json):
    """
    Display all the elements essential to the correction of the json file.

    Parameters
    ----------
    name_file: str
        The name of the json file.
    data_json: dict
        The original json file.

    Returns
    -------
    tuple
        The edited json file, a boolean to know if it has been edited and the column of the editor.
    """
    to_add = display_add_entity(data_json)
    col_editor, col_display = st.columns([1, 2])
    # Display the editor
    with col_editor:
        if to_add != None:
            f = open(f"../annotations/{name_file}", "w")
            data_json["annotations"][0][1]["entities"].append(to_add)
            save_json = json.dumps(data_json, indent=None)
            f.write(save_json)
            f.close()
        ent_json = json.dumps(data_json["annotations"][0][1], indent=4)
        new_json = st.text_area(
            "JSON Editor",
            ent_json,
            height=600,
        )
    # Display the entities
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


def display_infos(json_files, text_files):
    """
    Display the number of json files found and the number of text files found.

    Parameters
    ----------
    json_files: list
        The list of json name files.
    text_files: list
        The list of text name files.
    """
    st.sidebar.write(len(json_files), "/", len(text_files), "json files found.")


def display_filters(json_files):
    """
    Display the filters to select the json file to correct.

    Parameters
    ----------
    json_files: list
        The list of json name files.

    Returns
    -------
    str
        The path of the json file to correct.
    """
    if len(json_files) > 1:
        st.sidebar.title("Filters")
        select_json = st.sidebar.slider(
            "Choose a JSON file to correct:", 1, len(json_files), 1
        )
        search_json = st.sidebar.text_input(
            "Enter a JSON file name:", placeholder="Example : zenodo_838635.json"
        )
        if search_json:
            path_name = "../annotations/" + search_json
        else:
            path_name = "../annotations/" + json_files[select_json - 1] + ".json"
    else:
        path_name = "../annotations/" + json_files[0] + ".json"
    return path_name


def load_css():
    """Load a css style."""
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


def save_json(data_json, new_json, path_name, edited, col_msg):
    """
    Save the edited json file.

    Parameters
    ----------
    data_json: dict
        The original json file.
    new_json: str
        The edited json file.
    path_name: str
        The path of the json file to correct.
    edited: bool
        A boolean to know if it has been edited.
    col_msg: streamlit.columns
        The column where the message will be displayed.
    """
    save = st.button("Save", disabled=not edited)
    if save:
        f = open(path_name, "w")
        ent_save = re.sub("\n| ", "", new_json)
        data_json["annotations"][0][1] = json.loads(ent_save)
        save_json = json.dumps(data_json, indent=None)
        f.write(save_json)
        f.close()
        with col_msg:
            st.success("JSON file saved !", icon="✅")


def remove_json(path_name, col_msg):
    """
    Remove the json file.

    Parameters
    ----------
    path_name: str
        The path of the json file to correct.
    col_msg: streamlit.columns
        The column where the message will be displayed.
    """
    remove = st.button("Remove")
    if remove:
        os.remove(path_name)
        with col_msg:
            st.success("JSON file removed !", icon="✅")


def user_interaction():
    """Control the streamlit application.

    Allows interaction between the user and the set of json files.
    """
    st.set_page_config(page_title="JSON Corrector", layout="wide")
    load_css()
    st.title("JSON Corrector")
    col_msg, _ = st.columns([2, 1])
    os.chdir(os.path.split(os.path.abspath(__file__))[0])
    path = "../annotations/"
    json_files = [
        file.split("/")[-1].split(".")[0] for file in glob.glob(path + "*.json")
    ]
    text_files = [
        file.split("/")[-1].split(".")[0] for file in glob.glob(path + "*.txt")
    ]
    if json_files:
        display_infos(json_files, text_files)
        path_name = display_filters(json_files)
        f = open(path_name, "r")
        data_json = json.load(f)
        f.close()
        name_file = path_name.split("/")[-1]
        have_text(name_file, col_msg)
        st.sidebar.write(name_file)
        new_json, edited, col_editor = display_editor(name_file, data_json)
        with col_editor:
            col_save, col_remove, _, _ = st.columns([1, 1, 1, 1])
            with col_save:
                save_json(data_json, new_json, path_name, edited, col_msg)
            with col_remove:
                remove_json(path_name, col_msg)
    else:
        with col_msg:
            st.error("No JSON file found !", icon="🚨")


if __name__ == "__main__":
    user_interaction()

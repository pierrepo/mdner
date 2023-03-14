"""Streamlit tool to correct json files in the annotations folder."""

import streamlit as st
import json
import spacy
from spacy.training import Example
import re
import glob
import os


def display_have_text(name_file: str, col_msg: st.columns) -> None:
    """
    Display an error message if the text file does not exist.

    Parameters
    ----------
    name_file: str
        The name of the json file.
    col_msg: st.columns
        The column where the message will be displayed.
    """
    with col_msg:
        if not os.path.exists(f"../annotations/{name_file.split('.')[0]}.txt"):
            st.error(
                f"The text corresponding to the JSON file {name_file} does not exist !",
                icon="🚨",
            )


def display_ner(name_file: str, data_json: dict) -> None:
    """
    Visualizing the entity recognizer of the edited json.

    Parameters
    ----------
    name_file: str
        The name of the json file.
    data_json: dict
        The original json file.
    """
    size_entity = len(data_json["annotations"][0][1]["entities"])
    # Colors for the entities
    colors = {
        "TEMPERATURE": "#FF0000",
        "SOFTWARE": "#FFA500",
        "SIMULATION TIME": "#FD6C9E",
        "FF & MODEL": "#00FFFF",
        "MOLECULE": "#FFFF00",
    }
    if size_entity > 1:
        entity = data_json["annotations"][0][1]["entities"][
            st.session_state["selected"] - 1
        ][2]
        color_strip = colors[entity].lstrip("#")
        rgb = "rgba" + str(
            tuple(
                int(color_strip[i : i + 2], 16) if i != 1 else 1 for i in (0, 2, 4, 1)
            )
        )
        colors[
            "SELECTED"
        ] = f"linear-gradient(180deg, {rgb} 0%, {rgb} 90%, rgba(0,0,0,1) 90%)"
        data_json["annotations"][0][1]["entities"][st.session_state["selected"] - 1][
            2
        ] = "SELECTED"
    options = {
        "ents": [
            "TEMPERATURE",
            "SOFTWARE",
            "SIMULATION TIME",
            "FF & MODEL",
            "MOLECULE",
            "SELECTED",
        ],
        "colors": colors,
    }
    # Display the entities
    nlp = spacy.blank("en")
    text, annotations = data_json["annotations"][0]  # annotations = dict
    doc = nlp.make_doc(text)
    review_annotation = []
    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            st.error(
                f'Skipping entity: The word "{text[start:end]}" ({start}, {end}) can\'t be aligned',
                icon="🚨",
            )
        else:
            review_annotation.append([start, end, label])
    example = Example.from_dict(doc, {"entities": review_annotation})
    st.write("Json file: ", name_file)
    ent_html = spacy.displacy.render(
        example.reference, style="ent", jupyter=False, options=options
    )
    st.markdown(ent_html, unsafe_allow_html=True)
    st.write("\n")
    if size_entity > 1:
        data_json["annotations"][0][1]["entities"][st.session_state["selected"] - 1][
            2
        ] = entity


def found_entity(to_found: str, text: str, entities: list) -> tuple or None:
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
    tuple or None
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


def display_add_entity(data_json: dict, col_msg: st.columns) -> None:
    """
    Display input text, select box and button to add an entity.

    Parameters
    ----------
    data_json: dict
        The original json file.
    col_msg: st.columns
        The column where the message will be displayed.
    """
    col_input, col_select = st.sidebar.columns([1, 1])
    with col_input:
        to_found = st.text_input(
            "None",
            placeholder="To add an entity, enter the text",
            label_visibility="collapsed",
        )
    with col_select:
        ent = st.selectbox(
            "None",
            data_json["classes"],
            index=0,
            key="select_entity",
            label_visibility="collapsed",
        )
    add = st.sidebar.button("Add")
    if add:
        positions = found_entity(
            to_found,
            data_json["annotations"][0][0],
            data_json["annotations"][0][1]["entities"],
        )
        if positions is not None:
            data_json["annotations"][0][1]["entities"].append(
                [positions[0], positions[1], ent]
            )
            data_json["annotations"][0][1]["entities"].sort(key=lambda x: x[0])
        else:
            with col_msg:
                st.error(
                    f"The entity {to_found} is not in the text or exists !",
                    icon="🚨",
                )


def display_editor(data_json: dict, col_msg: st.columns) -> None:
    """
    Display all the tools for the correction of the json file.

    Parameters
    ----------
    data_json: dict
        The original json file.
    col_msg: st.columns
        The column where the message will be displayed.
    """
    st.sidebar.title("Editor")
    size_entity = len(data_json["annotations"][0][1]["entities"])
    if size_entity > 1:
        st.session_state["selected"] = st.sidebar.slider(
            "Choose an entity:",
            1,
            len(data_json["annotations"][0][1]["entities"]),
            1,
        )
    else:
        st.session_state["selected"] = size_entity
    if st.session_state["selected"] > 0:
        remove = st.sidebar.button("Remove entity")
        if remove:
            data_json["annotations"][0][1]["entities"].pop(
                st.session_state["selected"] - 1
            )
            st.session_state["selected"] -= 1
    display_add_entity(data_json, col_msg)


def display_filters(json_files: list) -> str:
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
        search_json = st.sidebar.text_input(
            "Enter a JSON file name:", placeholder="Example : zenodo_838635.json"
        )
        select_json = st.sidebar.slider(
            "Choose a JSON file to correct:", 1, len(json_files), 1
        )
        if search_json:
            path_name = "../annotations/" + search_json
        else:
            path_name = "../annotations/" + json_files[select_json - 1] + ".json"
    else:
        path_name = "../annotations/" + json_files[0] + ".json"
    return path_name


def save_json(path_name: str, data_json: dict) -> None:
    """
    Save the edited json file.

    Parameters
    ----------
    path_name: str
        The path of the json file to correct.
    data_json: dict
        The modified json file to save.
    """
    with open(path_name, "r+") as f:
        save_json = json.dumps(data_json, indent=None)
        f.seek(0)
        f.write(save_json)
        f.truncate()


def display_remove_json(path_name: str, col_msg: st.columns) -> None:
    """
    Remove the json file.

    Parameters
    ----------
    path_name: str
        The path of the json file to correct.
    col_msg: st.columns
        The column where the message will be displayed.
    """
    remove = st.button("Remove json file")
    if remove:
        os.remove(path_name)
        with col_msg:
            st.success("JSON file removed !", icon="✅")


def load_css() -> None:
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
                    
                    mark > span {
                        
                    }
                </style>
                """,
        unsafe_allow_html=True,
    )


def user_interaction() -> None:
    """Control the streamlit application.

    Allows interaction between the user and the set of json files.
    """
    st.set_page_config(page_title="JSON Corrector", layout="wide")
    os.chdir(os.path.split(os.path.abspath(__file__))[0])
    load_css()
    st.title("JSON Corrector")
    col_msg, _ = st.columns([2, 1])
    path = "../annotations/"
    json_files = [
        file.split("/")[-1].split(".")[0] for file in glob.glob(path + "*.json")
    ]
    if json_files:
        # Display filters and select the json file to correct
        path_name = display_filters(json_files)
        name_file = path_name.split("/")[-1]
        with open(path_name, "r") as f:
            data_json = json.load(f)
        # Display if the json has dedicated text file
        display_have_text(name_file, col_msg)
        # Display editor tools and get the selected entity
        display_editor(data_json, col_msg)
        # Display spacy visualizer
        display_ner(name_file, data_json)
        # Display remove json file button
        display_remove_json(path_name, col_msg)
        # Save the json file automatically
        save_json(path_name, data_json)
    else:
        with col_msg:
            st.error("No JSON file found !", icon="🚨")


if __name__ == "__main__":
    user_interaction()

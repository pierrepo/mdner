"""Streamlit tool to correct json files in the annotations folder."""

import streamlit as st
import json
import spacy
from spacy.training import Example
import glob
import os
import pandas as pd


def display_have_text(name_file: str, col_msg: st.columns, data_json: dict) -> None:
    """
    Display an error message if the text file does not exist.

    Parameters
    ----------
    name_file: str
        The name of the json file.
    col_msg: st.columns
        The column where the message will be displayed.
    data_json: dict
        The selected json file.
    """
    with col_msg:
        if not os.path.exists(f"annotations/{name_file.split('.')[0]}.txt"):
            st.error(
                f"The text corresponding to the JSON file {name_file} does not exist !",
                icon="🚨",
            )
            with open(f"annotations/{name_file.split('.')[0]}.txt", "w") as f:
                f.write(data_json["annotations"][0][0])


def display_table_entities(data_json: dict):
    """
    Display the position of the entities in the text with the label

    Parameters
    ----------
    data_json: dict
        The selected json file.
    """
    data = pd.DataFrame(columns=["Start", "End", "Label", "Span"])
    with st.sidebar.expander("Entities"):
        text, annotations = data_json["annotations"][0]
        for start, end, label in annotations["entities"]:
            span = text[start:end]
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        [[start, end, label, str(span)]], columns=data.columns
                    ),
                ],
                axis=0,
            )
        data.set_index("Span", inplace=True)
        st.table(data)


def display_perc_all_entities():
    """Display the percentage and total of each entity in the data folder."""
    path = "annotations/"
    name_entities = ["MOL", "STIME", "FFM", "SOFT", "TEMP"]
    ents_dict = dict.fromkeys(name_entities, 0)
    total = 0
    with st.sidebar.expander("Percentage of entities in the data folder :"):
        for json_name in glob.glob(path + "*.json"):
            with open(json_name, "r") as json_file:
                annotations = json.load(json_file)["annotations"][0][1]
                for _, _, label in annotations["entities"]:
                    ents_dict[label] += 1
                    total += 1
        for label in ents_dict:
            ents_dict[label] = [float(ents_dict[label] / total) * 100, ents_dict[label]]
        df_entities = pd.DataFrame.from_dict(
            ents_dict, orient="index", columns=["Percentage", "Total"]
        )
        st.table(df_entities)


def display_infos_entities(data_json: dict):
    """
    Display the number of entities and the percentage of each entity in the data folder.

    Parameters
    ----------
    data_json: dict
        The selected json file.
    """
    display_table_entities(data_json)
    display_perc_all_entities()


def display_ner(
    name_file: str, data_json: dict, path_name: str, col_msg: st.columns
) -> None:
    """
    Visualizing the entity recognizer of the edited json.

    Parameters
    ----------
    name_file: str
        The name of the json file.
    data_json: dict
        The original json file.
    path_name: str
        The path of the json file.
    col_msg: st.columns
        The column where the message will be displayed.
    """
    st.write("Json file: ", name_file)
    size_entity = len(data_json["annotations"][0][1]["entities"])
    # Colors for the entities
    colors = {
        "TEMP": "#FF0000",
        "SOFT": "#FFA500",
        "STIME": "#FD6C9E",
        "FFM": "#00FFFF",
        "MOL": "#FFFF00",
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
            "TEMP",
            "SOFT",
            "STIME",
            "FFM",
            "MOL",
            "SELECTED",
        ],
        "colors": colors,
    }
    # Create a blank spaCy model
    nlp = spacy.blank("en")
    text, annotations = data_json["annotations"][0]
    doc = nlp.make_doc(text)
    spans = []
    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="strict")
        if span is None:
            with col_msg:
                st.error(
                    f'Skipping entity: The word "{text[start:end]}" ({start}, {end}) can\'t be aligned',
                    icon="🚨",
                )
                annotations["entities"].remove([start, end, label])
        else:
            spans.append(span)
    doc.ents = spans
    example = Example.from_dict(doc, annotations)
    # Display the entities in the text in html
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
    exist = False
    to_found = to_found.lower()
    text = text.lower()
    for i in range(len(text)):
        if text.startswith(to_found, i):
            start = i
            end = i + len(to_found)
            # Check if the found entity can be aligned
            nlp = spacy.blank("en")
            doc = nlp.make_doc(text)
            span = doc.char_span(start, end, label="TEST", alignment_mode="strict")
            if span is not None:
                # Check if the entity already exists
                for ent in entities:
                    if (ent[0] == start or ent[1] == end) or (
                        start > ent[0] and end < ent[1]
                    ):
                        exist = True
                if not exist:
                    return start, end
            exist = False
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
    st.sidebar.markdown(
        """<p class="font-label"> Add an entity: </p>""",
        unsafe_allow_html=True,
    )
    to_found = st.sidebar.text_input(
        "None",
        placeholder="To add an entity, enter the text",
        label_visibility="collapsed",
    )
    col_select, col_add = st.sidebar.columns([1, 1])
    with col_select:
        ent = st.selectbox(
            "None",
            data_json["classes"],
            index=0,
            key="select_entity",
            label_visibility="collapsed",
        )
    with col_add:
        add = st.button("Add")
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


def change_cursor(minus: bool, slider_json: bool):
    """
    Change the cursor of a specific slider.

    Parameters
    ----------
    minus: bool
        If the cursor is on the minus button.
    slider_json: bool
        If the cursor is on the slider_json.
    """
    if slider_json:
        if minus:
            st.session_state.slider_json = st.session_state.slider_json - 1
        else:
            st.session_state.slider_json = st.session_state.slider_json + 1
        update_selected_entity()
    else:
        if minus:
            st.session_state.slider_entity = st.session_state.slider_entity - 1
        else:
            st.session_state.slider_entity = st.session_state.slider_entity + 1


def entity_selector(size_entity: int, data_json: dict):
    """
    Select an entity in the json file.

    Parameters
    ----------
    size_entity: int
        The number of entities in the json file.
    data_json: dict
        The json file.
    """
    if "slider_entity" not in st.session_state:
        st.session_state.slider_entity = 1
    st.sidebar.markdown(
        """<p class="font-label"> Choose an entity: </p>""",
        unsafe_allow_html=True,
    )
    select_minus, select_slider, select_plus = st.sidebar.columns([1, 8, 1])
    with select_minus:
        st.button(
            "◀",
            key="minus_entity",
            on_click=change_cursor,
            args=(
                True,
                False,
            ),
            disabled=(st.session_state.slider_entity <= 1),
            use_container_width=True,
        )
    with select_plus:
        st.button(
            "▶",
            key="plus_entity",
            on_click=change_cursor,
            args=(
                False,
                False,
            ),
            disabled=(st.session_state.slider_entity >= size_entity),
            use_container_width=True,
        )
    with select_slider:
        st.session_state["selected"] = st.slider(
            "Choose an entity:",
            min_value=1,
            max_value=len(data_json["annotations"][0][1]["entities"]),
            label_visibility="collapsed",
            key="slider_entity",
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
        entity_selector(size_entity, data_json)
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


def json_search():
    """Search a json file in the data folder."""
    search_json = st.sidebar.text_input(
        "Enter a JSON file name:", placeholder="Example : zenodo_838635.json"
    )
    return search_json


def update_selected_entity():
    """Update the value of the entity slider when the json slider is changed."""
    st.session_state.slider_entity = 1


def json_selector(json_files: list):
    """
    Select a json file in the data folder with a slider.

    Parameters
    ----------
    json_files: list
        The list of json files in the data folder.
    """
    if "slider_json" not in st.session_state:
        st.session_state.slider_json = 1
    st.sidebar.markdown(
        """<p class="font-label"> Choose a JSON file to correct: </p>""",
        unsafe_allow_html=True,
    )
    select_minus, select_slider, select_plus = st.sidebar.columns([1, 8, 1])
    with select_minus:
        st.button(
            "◀",
            on_click=change_cursor,
            args=(
                True,
                True,
            ),
            disabled=st.session_state.slider_json <= 1,
            use_container_width=True,
        )
    with select_plus:
        st.button(
            "▶",
            on_click=change_cursor,
            args=(
                False,
                True,
            ),
            disabled=st.session_state.slider_json >= len(json_files),
            use_container_width=True,
        )
    with select_slider:
        st.slider(
            "Choose a JSON file to correct:",
            min_value=1,
            max_value=len(json_files),
            label_visibility="collapsed",
            key="slider_json",
            on_change=update_selected_entity,
        )


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
        search_json = json_search()
        json_selector(json_files)
        if search_json:
            path_name = "annotations/" + search_json
        else:
            path_name = (
                "annotations/"
                + json_files[st.session_state.slider_json - 1]
                + ".json"
            )
    else:
        path_name = "annotations/" + json_files[0] + ".json"
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
            </style>
        """,
        unsafe_allow_html=True,
    )


def user_interaction() -> None:
    """Control the streamlit application.

    Allows interaction between the user and the set of json files.
    """
    st.set_page_config(page_title="Entity Annotator", layout="wide")
    load_css()
    st.title("Entity Annotator")
    col_msg, _ = st.columns([2, 1])
    path = "annotations/"
    json_files = [
        file.split("/")[-1].split(".")[0] for file in glob.glob(path + "*.json")
    ]
    if json_files:
        # Display filters and select the json file to correct
        if "selected" not in st.session_state:
            st.session_state["selected"] = 1  # session_state for the selected entity
        path_name = display_filters(json_files)
        name_file = path_name.split("/")[-1]
        with open(path_name, "r", encoding="utf-8") as f:
            data_json = json.load(f)
        # Display if the json has dedicated text file
        display_have_text(name_file, col_msg, data_json)
        # Display editor tools and get the selected entity
        display_editor(data_json, col_msg)
        # Display information about the selected entity and percentage of entities
        display_infos_entities(data_json)
        # Display spacy visualizer
        display_ner(name_file, data_json, path_name, col_msg)
        # Save the json file automatically
        save_json(path_name, data_json)
    else:
        with col_msg:
            st.error("No JSON file found !", icon="🚨")


if __name__ == "__main__":
    user_interaction()

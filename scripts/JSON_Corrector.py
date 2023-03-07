import pandas as pd
import streamlit as st
import json
import spacy
from spacy.training import Example
from spacy import displacy
import random
import re


def display_ner():
    colors = {
        "TEMPERATURE": "#FF0000",
        "SOFTWARE": "#FFA500",
        "SIMULATION TIME": "#FD6C9E",
        "MODEL": "#00FFFF",
        "MOLECULE": "#FFFF00",
    }
    options = {"ents": [
        "TEMPERATURE",
        "SOFTWARE",
        "SIMULATION TIME",
        "MODEL",
        "MOLECULE"
    ],
        "colors": colors
    }
    nlp = spacy.blank("en")
    text, _ = json_data["annotations"][0]
    example = Example.from_dict(nlp.make_doc(text), json.loads(st.session_state["ent"]))
    ent_html = spacy.displacy.render(example.reference, style="ent", jupyter=False, options=options)
    st.markdown(ent_html, unsafe_allow_html=True)


def display_editor():
    col_editor, col_display = st.columns([1,1])
    with col_editor:
        st.session_state["ent"] = st.text_area("JSON Editor", st.session_state["ent"], height=600, key=random.randint(0, 1000))
    with col_display:
        display_ner()


st.set_page_config(page_title="JSON Corrector",layout='wide')
st.title("JSON Corrector")
f = open("../annotations/zenodo_838635.json", "r")
json_data = json.load(f)
f.close()
if "ent" not in st.session_state:
    st.session_state["ent"] = json.dumps(json_data["annotations"][0][1], indent=4)
    st.session_state["previous"] = ""
save = st.button("Save")
col_editor, col_display = st.columns([1,1])
with col_editor:
    st.session_state["previous"] = st.session_state["ent"]
    st.session_state["ent"] = st.text_area("JSON Editor", st.session_state["ent"], height=600)
with col_display:
    try :
        display_ner()
    except Exception as e :
        pass
if save :
    st.session_state["ent"] = re.sub("\n| ", "", st.session_state["ent"])
    json_data["annotations"][0][1] = st.session_state["ent"]
    f = open("../annotations/zenodo_838635.json", "w")
    f.write(json.dumps(json_data))
    f.close()

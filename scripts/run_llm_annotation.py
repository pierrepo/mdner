# run_llm_annotation.py

import os
import time
import json
import jinja2
import datetime
from loguru import logger
from openai import OpenAI
from groq import Groq, InternalServerError

# === Configuration ===

ANNOTATIONS_FOLDER = "../annotations/"
PROMPT_PATH = "../prompt_templates/"

LIST_PROMPTS = ["zero_shot", "one_shot", "few_shot"]

LIST_MODELS_GROQ = [
    "gemma2-9b-it",
    "mistral-saba-24b",
    "llama-3.3-70b-versatile",
    "qwen-qwq-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "deepseek-r1-distill-llama-70b"
]

LIST_MODELS_OPENAI = [
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-11-20",
    "o3-2025-04-16"
]

TAGS = ["MOL", "SOFTNAME", "SOFTVERS", "STIME", "TEMP", "FFM"]

API_TYPE = "openai"

NUMBER_OF_TEXTS_TO_ANNOTATE = 1


# === Setup ===

def get_api(api_type: str):
    if api_type == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif api_type == "groq":
        return Groq(api_key=os.environ.get("GROQ_API_KEY"))
    else:
        raise ValueError("Invalid API type. Choose 'openai' or 'groq'.")


client = get_api(API_TYPE)

date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

OUTPUT_BASE = f"../llm_outputs/{date_and_time}/"
OUTPUT_FOLDERS = {
    "annotations": os.path.join(OUTPUT_BASE, "annotations"),
    "stats": os.path.join(OUTPUT_BASE, "stats"),
    "images": os.path.join(OUTPUT_BASE, "images")
}

for folder in OUTPUT_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

LIST_MODELS = LIST_MODELS_OPENAI if API_TYPE == "openai" else LIST_MODELS_GROQ

for prompt in LIST_PROMPTS:
    prompt_folder = os.path.join(OUTPUT_FOLDERS["annotations"], prompt)
    os.makedirs(prompt_folder, exist_ok=True)
    for model in LIST_MODELS:
        os.makedirs(os.path.join(prompt_folder, model), exist_ok=True)


# === Helper Functions ===

def process_json_file(json_file: str) -> tuple:
    with open(json_file, "r") as f:
        data = json.load(f)
    annotation_entry = data["annotations"][0]
    return annotation_entry[0], annotation_entry[1]["entities"]


def load_and_render_prompt(template_path: str, text_to_annotate: str) -> str:
    with open(template_path, "r") as f:
        template_content = f.read()
    template = jinja2.Template(template_content)
    return template.render(text_to_annotate=text_to_annotate)


def chat_with_template(prompt: str, template_path: str, model: str, text_to_annotate: str) -> str:
    delay = 1
    max_retries = 5

    prompt = load_and_render_prompt(template_path, text_to_annotate)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
            )
            return response.choices[0].message.content
        except InternalServerError as err:
            if getattr(err, "status_code", None) in (503, 249) and attempt < max_retries:
                logger.warning(f"503 error for model {model}, retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                logger.error(f"Error from model {model}: {err}")
                raise


def save_response_as_json(response_text: str, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response_text, f, ensure_ascii=False, indent=2)


# === Main Processing Loop ===

def main():
    logger.info(f"Starting LLM annotation process at {date_and_time}")
    number_texts = 0

    for filename in os.listdir(ANNOTATIONS_FOLDER):
        if number_texts >= NUMBER_OF_TEXTS_TO_ANNOTATE:
            break

        if filename.endswith(".json") and filename.count("_") == 1:
            number_texts += 1
            logger.info(f"Processing file {number_texts}: {filename}...")
            input_path = os.path.join(ANNOTATIONS_FOLDER, filename)
            input_text, _ = process_json_file(input_path)

            for prompt in LIST_PROMPTS:
                logger.info(f"Testing prompt: {prompt} ------")
                prompt_folder = os.path.join(OUTPUT_FOLDERS["annotations"], prompt)

                for model in LIST_MODELS:
                    logger.info(f"Testing model: {model}")
                    output_model_folder = os.path.join(prompt_folder, model)

                    response = chat_with_template(
                        prompt=prompt,
                        template_path=os.path.join(PROMPT_PATH, f"{prompt}.txt"),
                        model=model,
                        text_to_annotate=input_text
                    )

                    output_path = os.path.join(output_model_folder, filename)
                    data = {
                        "model": model,
                        "text_to_annotate": input_text,
                        "response": response
                    }

                    save_response_as_json(data, output_path)

    logger.success("Annotation process completed.")


if __name__ == "__main__":
    main()

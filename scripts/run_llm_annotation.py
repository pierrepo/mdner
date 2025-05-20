# run_llm_annotation.py

import datetime
import json
import os
import time

import jinja2
from groq import Groq, InternalServerError, RateLimitError
from loguru import logger
from openai import OpenAI

# ======================================================================================
# Configuration
# ======================================================================================

# Folder where we have the json files to annotate
ANNOTATIONS_FOLDER = "../annotations/"

# Number of texts to annotate
NUMBER_OF_TEXTS_TO_ANNOTATE = 1

# Folder where the prompt templates are stored
PROMPT_PATH = "../prompt_templates/"

# Name of the prompts to then name the output folders
# LIST_PROMPTS = ["zero_shot", "one_shot", "few_shot"]
LIST_PROMPTS = ["few_shot_5", "few_shot_15", "few_shot_30"]

# INPUT: Determine which API to use
API_TYPE = input("Which API to use ('groq' or 'openai'): ")

# Models to test depending on the API key
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
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o-2024-11-20",
    "o4-mini-2025-04-16",
    "o3-2025-04-16",
    "o3-mini-2025-01-31",
]

# ======================================================================================
# Client and setup confirmation
# ======================================================================================

def get_api(api_type: str):
    """
    Returns the API client based on the provided API type.
    """
    if api_type == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif api_type == "groq":
        return Groq(api_key=os.environ.get("GROQ_API_KEY"))
    else:
        raise ValueError("Invalid API type. Choose 'openai' or 'groq'.")

# Initialize the API client
client = get_api(API_TYPE)

# log the current setup : api type, models, prompts, number of texts : confirm
logger.info(f"Using API: {API_TYPE}")
logger.info(f"Models: {LIST_MODELS_GROQ if API_TYPE == 'groq' else LIST_MODELS_OPENAI}")
logger.info(f"Prompts: {LIST_PROMPTS}")
logger.info(f"Number of texts to annotate: {NUMBER_OF_TEXTS_TO_ANNOTATE}")

# Is this the correct config? (yes/no)
confirm_setup = input("Is this the correct config? (yes/no): ").strip().lower()
if confirm_setup != "yes":
    logger.error("Configuration not confirmed. Exiting...")
    logger.error("Modify config in '../scripts/run_llm_annotation.py'")
    exit(1)

# ======================================================================================
# Create output folders with current date and time
# ======================================================================================

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

# ======================================================================================
# Helper Functions
# ======================================================================================

def process_json_file(json_file: str) -> tuple:
    """
    Process the JSON file and return the text to annotate and the entities.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    annotation_entry = data["annotations"][0]
    return annotation_entry[0], annotation_entry[1]["entities"]


def load_and_render_prompt(template_path: str, text_to_annotate: str) -> str:
    """
    Load the prompt template from the specified path and
    render it with the provided text.
    """
    with open(template_path, "r") as f:
        template_content = f.read()
    template = jinja2.Template(template_content)
    return template.render(text_to_annotate=text_to_annotate)


def chat_with_template(
        prompt: str,
        template_path: str,
        model: str,
        text_to_annotate: str
        ) -> str:
    """
    Chat with the model using the specified prompt and text to annotate.
    Handles rate limits and retries.
    """
    delay = 1
    max_retries = 10

    prompt = load_and_render_prompt(template_path, text_to_annotate)

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
            )
            content = response.choices[0].message.content
            usage = response.usage

            # Extract the three token counts
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens  = usage.total_tokens

            # Bundle them in a dict
            usage_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            return content, usage_dict

        except RateLimitError as err:
            status = getattr(err, "status_code", None)
            if status in (498, 499, 429) and attempt < max_retries:
                logger.warning(
                    f"{status} error for model {model}, retrying in {delay}s..."
                    )
                time.sleep(delay)
                delay *= 5
                continue
            else:
                logger.error(f"Error from model {model}: {err}")
                raise
        except InternalServerError as err:
            status = getattr(err, "status_code", None)
            if status in (503, 500, 502) and attempt < max_retries:
                logger.warning(
                    f"{status} error for model {model}, retrying in {delay}s..."
                    )
                time.sleep(delay)
                delay *= 5
                continue
            else:
                logger.error(f"Error from model {model}: {err}")
                raise


def save_response_as_json(response_text: str, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response_text, f, ensure_ascii=False, indent=2)


# ======================================================================================
# Main Logic
# ======================================================================================


def main():
    logger.info(f"Starting LLM annotation process at {date_and_time}")
    # Count the number of texts to annotate to respect the limit
    number_texts = 0

    for filename in os.listdir(ANNOTATIONS_FOLDER):
        if number_texts >= NUMBER_OF_TEXTS_TO_ANNOTATE:
            break

        # Check if the file is a JSON file and has the correct format "_"
        # Grab file
        if filename.endswith(".json") and filename.count("_") == 1:
            number_texts += 1
            logger.info(f"Processing file {number_texts}: {filename}...")
            input_path = os.path.join(ANNOTATIONS_FOLDER, filename)
            input_text, _ = process_json_file(input_path)

            # Grab prompt
            for prompt in LIST_PROMPTS:
                logger.info(f"Testing prompt: {prompt} ------")
                prompt_folder = os.path.join(OUTPUT_FOLDERS["annotations"], prompt)

                # Grab model
                for model in LIST_MODELS:
                    logger.info(f"Testing model: {model}")
                    output_model_folder = os.path.join(prompt_folder, model)

                    # Chat with the model
                    response, usage = chat_with_template(
                        prompt=prompt,
                        template_path=os.path.join(PROMPT_PATH, f"{prompt}.txt"),
                        model=model,
                        text_to_annotate=input_text
                    )

                    # Save the response
                    output_path = os.path.join(output_model_folder, filename)
                    data = {
                        "model": model,
                        "text_to_annotate": input_text,
                        "response": response,
                        "usage": usage,
                    }

                    save_response_as_json(data, output_path)

    logger.success("Annotation process completed.")


if __name__ == "__main__":
    main()

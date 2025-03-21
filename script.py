# requirements:
# google-generativeai
# wmill
# pandas
# pillow
# requests
import google.generativeai as genai
import pandas as pd
import json
import re
import io
import requests
from PIL import Image
from wmill import task

# Configure Gemini API key
genai.configure(api_key="YOUR_API_KEY")


def download_image_from_url(image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.verify()
        return response.content
    except Exception as e:
        raise ValueError(f"Failed to download/validate image: {str(e)}")


def analyze_floor_plan_from_image_bytes(image_bytes: bytes):
    model = genai.GenerativeModel("gemini-2.0-flash")

    image = Image.open(io.BytesIO(image_bytes))

    response = model.generate_content(
        [
            "Identify all rooms and their approximate areas in square meters from this floor plan image. "
            "Provide details in a structured way.",
            image,
        ]
    )
    if response and response.text:
        return response.text
    else:
        raise ValueError(
            "Empty response from Gemini API. Check API configuration or input image."
        )


def clean_json_text(raw_text):
    cleaned_text = re.sub(r"```json|```", "", raw_text).strip()
    return cleaned_text


def extract_structured_data(response_text):
    structured_model = genai.GenerativeModel("gemini-2.0-flash")
    structured_prompt = f"""{response_text}
    Output format:
    {{
        "rooms": [
            {{"room_name": "Living Area", "area": 15.0, "thinking": "Estimated based on image proportions."}},
            ...
        ]
    }} """

    structured_response = structured_model.generate_content(structured_prompt)
    cleaned_json = clean_json_text(structured_response.text)
    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON. Cleaned response was:\n" + cleaned_json)


def save_to_csv(data):
    df = pd.DataFrame(data["rooms"])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


@task()
def main(image_url: str):
    image_bytes = download_image_from_url(image_url)
    response_text = analyze_floor_plan_from_image_bytes(image_bytes)
    structured_data = extract_structured_data(response_text)
    csv_result = save_to_csv(structured_data)

    return {"structured_data": structured_data, "csv": csv_result}

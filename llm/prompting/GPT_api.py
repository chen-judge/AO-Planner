import openai
from openai import OpenAI
import base64
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

generation_key = "your_gpt_key"

client = OpenAI(api_key=generation_key)
openai.api_key = generation_key
client = OpenAI(
    api_key=generation_key,
    base_url="your_api_link"  # TODO: set your api link
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    """
    To avoid this: "BadRequestError: Error code: 400 - {'error': {'message': "You uploaded an unsupported image.
    Please make sure your image is below 20 MB in size and is of one the following formats: ['png', 'jpeg', 'gif', 'webp'].",
    'type': 'invalid_request_error', 'param': None, 'code': 'sanitizer_server_error'}}"
    """
    return client.chat.completions.create(**kwargs)


def gpt4v_infer(messages, model="gpt-4-vision-preview", max_tokens=600, response_format=None):
    """
    OpenAI 1.3.7
    """

    for message in messages[1:]:
        if isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "image_url":
                    image = content["image_url"]["url"]
                    with open(image, "rb") as image_file:
                        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    content["image_url"]["url"] = f"data:image/jpeg;base64,{image_base64}"

    if response_format:
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens, response_format=response_format)
    else:  # gpt-4-turbo does not support response_format
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens)

    answer = chat_message.choices[0].message.content
    tokens = chat_message.usage

    return answer, tokens

def make_image_content(images):
    user_content = []
    for image_dict in images:
        for key, image in image_dict.items():
            if image is not None:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"{key}:"
                    }
                )

                image_message = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{image}",
                        "detail": "low"
                    }
                }
                user_content.append(image_message)

    return user_content
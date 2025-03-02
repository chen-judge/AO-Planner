import PIL
import google.generativeai as genai
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

key_list = [
    "your_gemini_api_keys",
]

model = genai.GenerativeModel('gemini-1.5-pro-latest')


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return model.generate_content(**kwargs)


def gemini_infer(system, text, image_list, temperature=0):
    class Tokens:
        def __init__(self):
            self.prompt_tokens = 0
            self.completion_tokens = 0

    prompt_list = [system]

    if text is not None:
        prompt_list.append(text)

    for i, image in enumerate(image_list):
        if image is not None:
            prompt_list.append(f"Image {i}:")

            img = PIL.Image.open(image)
            prompt_list.append(img)

    api_key = random.choice(key_list)
    genai.configure(api_key=api_key)

    response = completion_with_backoff(
            contents=prompt_list,
            generation_config=genai.types.GenerationConfig(
            temperature=temperature,  # default 0.9
        )
    )

    response.resolve()  # acquire full text
    answer = response.text

    tokens = Tokens()
    tokens.prompt_tokens = 0
    tokens.completion_tokens = 0

    return answer, tokens


def gemini_interleaved_infer(messages, temperature=0):
    class Tokens:
        def __init__(self):
            self.prompt_tokens = 0
            self.completion_tokens = 0

    system = messages[0]["content"]
    user_content = messages[1]["content"]

    prompt_list = [system]

    for content in user_content:
        if content["type"] == "text":
            prompt_list.append(content["text"])
            print('Text: ', content["text"])

        elif content["type"] == "image_url":
            image = content["image_url"]["url"]
            print('Image: ', image)

            img = PIL.Image.open(image)
            prompt_list.append(img)
        else:
            raise NotImplemented


    api_key = random.choice(key_list)
    genai.configure(api_key=api_key)

    response = completion_with_backoff(
            contents=prompt_list,
            generation_config=genai.types.GenerationConfig(
            temperature=temperature,  # default 0.9
        )
    )

    response.resolve()  # acquire full text
    answer = response.text

    tokens = Tokens()
    tokens.prompt_tokens = 0
    tokens.completion_tokens = 0

    return answer, tokens



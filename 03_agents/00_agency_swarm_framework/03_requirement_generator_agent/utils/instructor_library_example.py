import instructor
from anthropic import AnthropicBedrock
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from PIL import Image
import io
import base64

# https://medium.com/@dminhk/unlocking-structured-outputs-with-amazon-bedrock-a-guide-to-leveraging-instructor-and-anthropic-abb76e4f6b20

env_loaded = load_dotenv(".env")
print(f"Env loaded: {env_loaded}")

HAIKU_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
SONNET_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

client = instructor.from_anthropic(
                                    AnthropicBedrock(
                                        aws_access_key=os.getenv("AWS_ACCESS_KEY"),
                                        aws_secret_key=os.getenv("AWS_SECRET_KEY"),
                                        aws_region=os.getenv("AWS_REGION"),
                                    )
                                )

def image_to_base64(image_path):
    # Open the image file
    with open(image_path, "rb") as image_file:
        # Read the image file
        image_data = image_file.read()
        # Encode the image data in base64
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return base64_encoded

# Simple example to extract user information from the message.
def example_1_simple_anthropic_bedrock():
    class User(BaseModel):
        name: str
        age: int

    # note that client.chat.completions.create will also work
    resp = client.messages.create(
        model=HAIKU_MODEL_ID,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.name == "Jason"
    assert resp.age == 25

    print(resp)
    # User(name='Jason', age=25)

# Image explanation example - 1
def example_2_image_explanation_antrhopic_bedrock():
    IMAGE_PATH = "D:/tmp/unstructured/pdfImages2/figure-6-5.jpg"

    # https://docs.anthropic.com/en/docs/build-with-claude/vision
    class ImageInfo (BaseModel):
        """
    Represents information about an image.
    """
        image_title: str = Field(
        description="A short title for the image.",
    ) 
        image_description: str = Field(
        description="A detailed description of the image.",
    )

    resp = client.messages.create(
            model=HAIKU_MODEL_ID,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_to_base64(IMAGE_PATH),
                            },
                        },
                        {
                            "type": "text",
                            "text": "You shall be provided with an image. Try to understand its content and provide a detailed description of the image along with a suitable title. Try to focus on the content present in the image and not the aesthetics of the image. \n## Response format:\nFormat your response as json with keys image_title and image_description"
                        }
                    ],
                }
            ],
            response_model=ImageInfo,
        )

    print(resp)
    print(resp.image_title)
    print(resp.image_description)

# Image explanation example - 2
def example_3_image_explanation_antrhopic_bedrock():
    IMAGE_PATH = "D:/tmp/unstructured/pdfImages2/table-18-2.jpeg"

    # https://docs.anthropic.com/en/docs/build-with-claude/vision
    class HTMLTable (BaseModel):
        """
        Represents table image as HTML table text.
        """
        html_table_syntax: str = Field(
        description="Image represented as HTML table syntax",
    ) 

    resp = client.messages.create(
            model=SONNET_MODEL_ID,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_to_base64(IMAGE_PATH),
                            },
                        },
                        {
                            "type": "text",
                            "text": "You shall be provided with an image that resembles a table. Convert the image representation of the table to HTML table. The output should be in HTML format with just including the <table> tag. Do not add any other text to the output other than the HTML <table> representation."
                        }
                    ],
                }
            ],
            response_model=HTMLTable,
        )

    print(resp)

example_2_image_explanation_antrhopic_bedrock()
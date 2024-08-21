"""
AzureOpenAIHelper class is a light-weight alternative to the existing frameworks such as langchain, llamaindex to invoke deployed models on azure.

This class is capable enough to support streaming mode and standard mode.

Required ENV vars:
AZURE_OPENAI_ENDPOINT=<YOUR_AZURE_OPENAI_ENDPOINT>
AZURE_OPENAI_API_KEY=<YOUR_AZURE_OPENAI_API_KEY>
AZURE_DEPLOYMENT_NAME=gpt-4o
AZURE_API_VERSION=2024-02-15-preview
"""
from utils import remove_trailing_forward_slash
from openai import AzureOpenAI
from config import settings
import base64
import tiktoken


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    elif "gpt-4o" in model:
        return num_tokens_from_messages(messages, model="gpt-4o")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if key in ["role", "name"]:
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
            if key == "content":
                if value[0]["type"] == "text":
                    num_tokens += len(encoding.encode(value[0]["type"]))
                    num_tokens += len(encoding.encode(value[0]["text"]))

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def fetch_image_description_from_azure_openai(b64_image_data):
    system_message = "You are an expert in analyzing images and provide descriptive content as per the user requirement."
    user_message = "You shall be provided with an image. Try to understand its content and provide a detailed description of the image along with a suitable title. Try to focus on the content present in the image and not the aesthetics of the image. \n## Response format:\nFormat your response as below:\nImage Title:\nImage Description:\n"

    azure_openai_api = AzureOpenAIHelper(system_message=system_message)

    response = azure_openai_api.invoke(message=user_message, b64_img_content=b64_image_data)
    return response

class AzureOpenAIHelper:    
    def __init__(self, system_message=None, temperature=0, max_tokens=1024, top_p=1, max_conversations=5):	
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.__client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,  
            api_version=settings.AZURE_API_VERSION,
            azure_endpoint = remove_trailing_forward_slash(settings.AZURE_OPENAI_ENDPOINT),
        )
        self.__system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful Assistant." if system_message is None else system_message
                }
            ]
        }
        self.__conversation_history = []
        self.max_conversations = max_conversations

    def __add_user_message(self, message, b64_img_content=None): 
        
        if not message and not b64_img_content:
            raise ValueError("message cannot be empty when b64_img_content is not provided.")

        content = []
        # if b64_img_content is provided, then its an image followed with/without a question.
        # Image with a followup question.
        if b64_img_content and message:
            content.extend([{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img_content}"
                }
            }, {
                    "type": "text",
                    "text": message
            }])
        # Image without a followup question.
        elif b64_img_content:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img_content}"
                }
            })
        # Normal text based question.
        else:
            content.append({
                "type": "text",
                "text": message
            })
        
        self.__conversation_history.append({
            "role": "user",
            "content": content
        })
    
    def __add_assistant_message(self, message): 
        self.__conversation_history.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": message
                }
            ]
        })
    
    def __prepare_payload(self, stream=False):
        # Append system message to the top. This is done to support truncating the conversation history.
        payload = {
            "model": settings.AZURE_DEPLOYMENT_NAME,
            "messages": [self.__system_message] + self.__conversation_history,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": stream
        }        
        return payload
    
    def invoke(self, message, b64_img_content=None, debug=False):
        self.__add_user_message(message=message, b64_img_content=b64_img_content)
        payload = self.__prepare_payload()
        # Send request
        if debug:
            print(f"Sending request: {payload}")
        try:
           response =self.__client.chat.completions.create(**payload)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to make the request. Error: {e}")
        assistant_message = response.choices[0].message.content
        if debug:
            print(f"Prompt Tokens: {response.usage.prompt_tokens} | Completion Tokens: {response.usage.completion_tokens} | "
                f"Total Tokens: {response.usage.total_tokens}")
        self.__add_assistant_message(message=assistant_message)
        self.__truncate_conversation_history()
        return assistant_message
    
    @property
    def conversation_history(self):
        return self.__conversation_history
    

    def __truncate_conversation_history(self):
        if int(len(self.conversation_history)/2) > self.max_conversations:
            del self.__conversation_history[:2]

    def get_usage_tokens(self):
        """
        Calculate the number of tokens from the conversation history.
        Note that any conversation removed from the history will be ignored during the calculation.
        Also, the images are not taken into consideration during token calculation.

        Please note that this is an approximation and not exact token count.
        """
        return num_tokens_from_messages(self.__conversation_history)


    def stream(self, message, b64_img_content=None, debug=False):
        self.__add_user_message(message=message, b64_img_content=b64_img_content)
        payload = self.__prepare_payload(stream=True)
        chunk_array = []
        # Send request
        if debug:
            print(f"Sending request: {payload}")
        try:
           response_stream =self.__client.chat.completions.create(**payload)
           for chunk in response_stream:
               if chunk.choices and chunk.choices[0].delta.content:
                chunk_array.append(chunk.choices[0].delta.content)
                yield chunk.choices[0].delta.content
        except RuntimeError as e:
            raise RuntimeError(f"Failed to make the request. Error: {e}")
        
        assistant_message = "".join(chunk_array)
        self.__add_assistant_message(message=assistant_message)
        self.__truncate_conversation_history()

    def clear_conversation_history(self):
        self.__conversation_history = []

# Test the implementation
if __name__ == "__main__":

    # Enable debug logging
    import logging
    logging.basicConfig(level=logging.INFO)


    azure_openai_api = AzureOpenAIHelper()
    while True:
        message = input("\nEnter user message (Type \"exit\" to break the conversation): ")
        if message == "exit":
            azure_openai_api.clear_conversation_history()
            break
        
        IMAGE_PATH = "C:/Users/20092/Downloads/maxresdefault.jpg"
        b64_img_content = base64.b64encode(open(IMAGE_PATH, "rb").read()).decode("utf-8")

        # Invoke Option - TEXT ONLY
        response = azure_openai_api.invoke(message=message, b64_img_content=None, debug=True)
        print("**** Response: ****")
        print(response)

        # Invoke Option - TEXT + IMAGE
        # response = azure_openai_api.invoke(message=message, b64_img_content=b64_img_content)
        # print("**** Response: ****")
        # print(response)

        # Invoke Option - IMAGE ONLY
        # response = azure_openai_api.invoke(message=None, b64_img_content=b64_img_content)
        # print("**** Response: ****")
        # print(response)

        # Streaming Option - TEXT ONLY
        # response = azure_openai_api.stream(message=message, b64_img_content=None, debug=True)
        # print("**** Response: ****")
        # for chunk in response:
        #     print(chunk, end="", flush=True)
        # 
        # print(f"TOKENS: {azure_openai_api.get_usage_tokens()}")

        # Streaming Option - TEXT + IMAGE
        # response = azure_openai_api.stream(message=message, b64_img_content=b64_img_content)
        # print("**** Response: ****")
        # for chunk in response:
        #     print(chunk, end="", flush=True)

        # Streaming Option - IMAGE ONLY
        # response = azure_openai_api.stream(message=None, b64_img_content=b64_img_content)
        # print("**** Response: ****")
        # for chunk in response:
        #     print(chunk, end="", flush=True)

        # print("\n**** Conversation history: ****")
        # print(f"Number of conversation turns happened: {len(azure_openai_api.conversation_history)/2}")
        # print(azure_openai_api.conversation_history)
    
    # print("*** CONVERSATION HISTORY CLEARED ***")
    # print(azure_openai_api.conversation_history)

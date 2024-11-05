## Activities: 

### Data Ingestion:
# 1. Read PDF by page and convert each page into an image for LLM Summarization. (Detailed Summary with multiple chunks.)
# 2. Summarize each image using LLM.
# 3. Store the summarized content in vector store with appropriate metadata (Id, Filename, Filepath, PageNumber, ImagePath)

### Retrieval:
# 1. Query the vector store and retrieve the top K chunks. 
# 2. Rerank the retrieved chunks based on their relevance to the question.
# 3. Extract images for the reranked chunks.

### Response Generation
### Option 1:
# 1. Send the retrieved context to the LLM with appropriate prompt to answer the question.
### Option 2 (Advanced):
# 1. Send the retrieved context along with the image to the LLM with appropriate prompt to answer the question.

from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel, Field
import base64
import os
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import AzureChatOpenAI
import pickle
import time

env_loaded = load_dotenv()

print(f"Env loaded: {env_loaded}")

pdf_file = "./data/jio-financial-services-annual-report-2023-2024-small.pdf"
images_dir = "./data/images"
temp_dir = "./temp"


############################################## Data Ingestion - START ####################################################

class PropositionalChunk(BaseModel):
    """Propositional Chunk created from the image."""
    image_path: str = Field(description="Path to the image.")
    chunk_number: str = Field(description="Sequence number given to the chunk.")
    propositional_chunk: str = Field(description="Propositional chunk from the image.")

class PropositionalChunks(BaseModel):
    """Propositional Chunk created from the image."""
    image_path: str = Field(description="Path to the image for which the propositional chunks were created.")
    """Consolidated Propositional Chunks created from the image."""
    propositional_chunks: List[PropositionalChunk] = Field(description="List of Propositional chunks from the image.")



########################################################################################################################

#### 1. Read PDF by page and convert each page into an image for LLM Summarization. (Detailed Summary with multiple chunks.)

def convert_pdf_to_images():
    import fitz

    os.makedirs(images_dir, exist_ok=True)

    doc = fitz.open(pdf_file)

    for page_number in range(len(doc)):  # Iterate through the pages
        page = doc.load_page(page_number)  # Load the page
        pix = page.get_pixmap(dpi=150)  # Render the page to a pixmap
        pix.save(f'./data/images/page-{page_number + 1}.png')  # Save the pixmap as a PNG file

    doc.close()
# convert_pdf_to_images()


########################################################################################################################



########################################################################################################################

#### 2. Summarize each image using LLM.

def convert_images_to_propositional_chunks():
    model_id = "gpt-4o-mini"

    azure_openai_model = AzureChatOpenAI(model=model_id, max_tokens=8192)

    image_summarization_prompt = """
    You are an expert in understanding the provided image and convert it into a detailed textual representation. Make sure that you capture all aspects of the image and write detailed description of the image content.  Ensure that the textual representation of the image is as comprehensive as possible.

    Image path in the file system: {image_path}

    Format your detailed response into multiple chunks with each chunk holding 200-250 words. Each chunk should be in the form of proposition that should stand on its own. These chunks shall be stored in a vector database to perform a similarity search so prepare it accordingly.
    """

    # Iterate through the images and generate descriptions.
    image_description = []
    batch_messages = []
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        
        print("Processing image:", image_path)
        
        # Read the image file and encode it to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Summarize the image using LLM
        messages = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
                },
                {"type": "text", "text": image_summarization_prompt.format(image_path=image_path)},
            ],
        )
        batch_messages.append([messages])

    batch_response = azure_openai_model.with_structured_output(PropositionalChunks).batch(batch_messages, config={"max_concurrency": 5})

    for response in batch_response:
        image_description.append({
            "image_path": response.image_path,
            "propositional_chunks": response.propositional_chunks
        })
    # Serialize the list to a file
    pickle_file = f"{temp_dir}/image_description_{time.time_ns()}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(image_description, file)
    
    return pickle_file
# image_description_pickle_file = convert_images_to_propositional_chunks()
# print(f"Image description pickle file created: {image_description_pickle_file}")

########################################################################################################################



########################################################################################################################

#### 3. Store the summarized content in vector store with appropriate metadata (Id, Filename, Filepath, PageNumber, ImagePath)

# Deserialize the list from the file
file_path = f"{temp_dir}/image_description_1730829480476374600.pkl"
with open(file_path, 'rb') as file:
    loaded_image_description = pickle.load(file)

for image_description in loaded_image_description:
    print("*" * 30)
    print("Image path:", image_description["image_path"])
    for chunk in image_description["propositional_chunks"]:
        print("*" * 20)
        print("Chunk number:", chunk.chunk_number)
        print("Image path:", chunk.image_path)
        print("Chunk content:")
        print(chunk.propositional_chunk)

########################################################################################################################

############################################## Data Ingestion - END ####################################################

import os
from langchain_core.messages import HumanMessage
from langchain_cohere import CohereRerank
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
load_dotenv()

# https://github.com/Azure/azureml-examples/tree/main/sdk/python/foundation-models/cohere
# https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/cohere/rerank-3-langchain.ipynb

co = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
query = "What is the capital of the United States?"
docs = [
    "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
    "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
    "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. The President of the USA and many major national government offices are in the territory. This makes it the political center of the United States of America.",
    "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states. The federal government (including the United States military) also uses capital punishment.",
]
results = co.rerank(query=query, documents=docs)
print(results)
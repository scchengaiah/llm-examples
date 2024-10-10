# Using Custom Impln of Azure API. Refer to mindsearch\agent\models.py
python -m mindsearch.app --lang en --model_format azure_gpt4 --search_engine DuckDuckGoSearch

# Using Standard OpenAI API. Refer to mindsearch\agent\models.py
python -m mindsearch.app --lang en --model_format gpt4 --search_engine DuckDuckGoSearch
import json
import os
import requests
from agency_swarm.tools import BaseTool
from pydantic import Field

class SearchInternet(BaseTool):
    """
    Tool for searching the internet about a given topic and returning relevant results. Further scraping of the website shall be performed by the ScrapeAndSummarizeWebsite tool.
    """
    query: str = Field(..., description="The search query for the internet search.")

    def run(self):
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": self.query})
        headers = {
            'X-API-KEY': os.environ['SERPER_API_KEY'],
            'content-type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        if 'organic' not in response.json():
            return "Sorry, I couldn't find anything about that, there could be an error with your Serper API key."
        else:
            results = response.json()['organic']
            string = []
            for result in results[:top_result_to_return]:
                try:
                    string.append('\n'.join([
                        f"Title: {result['title']}",
                        f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}",
                        "\n-----------------"
                    ]))
                except KeyError:
                    continue

            return '\n'.join(string)
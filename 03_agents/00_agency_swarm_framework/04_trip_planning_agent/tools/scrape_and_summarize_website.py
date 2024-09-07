import json
import os
import requests
from agency_swarm.tools import BaseTool
from pydantic import Field
from unstructured.partition.html import partition_html
from agency_swarm import get_openai_client
from textwrap import dedent
from playwright.sync_api import sync_playwright

class ScrapeAndSummarizeWebsite(BaseTool):
    """
    Tool for scraping and summarizing website content. The website shall be identified from the SearchInternet tool.
    """
    website: str = Field(..., description="The URL of the website to scrape and summarize.")

    def run(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.website)
            page_content = page.content()
            browser.close()

        elements = partition_html(text=page_content)
        content = "\n\n".join([str(el) for el in elements])
        content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
        summaries = []
        client = get_openai_client()

        for chunk in content:
            task_description = dedent(f"""
                Analyze and summarize the content below, make sure to include the most relevant information in the summary, return only the summary nothing else.

                CONTENT
                ----------
                {chunk}
            """)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Principal Researcher at a big company and you need to do research about a given topic."},
                    {"role": "user", "content": task_description}
                ],
            )
            summary = completion.choices[0].message.content
            summaries.append(summary)

        return "\n\n".join(summaries)
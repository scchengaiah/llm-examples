from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

markdown_document = "# Intro \n\n    ## History \n\n Markdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9] \n\n Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files. \n\n ## Rise and divergence \n\n As Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for \n\n additional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks. \n\n #### Standardization \n\n From 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort. \n\n ## Implementations \n\n Implementations of Markdown are available for over a dozen programming languages."
markdown_document = """Mahatma Gandhi, born Mohandas Karamchand Gandhi on October 2, 1869, in Porbandar, India, emerged as a pivotal figure in the struggle for Indian independence from British rule. Known for his unwavering commitment to nonviolent resistance, Gandhi developed and popularized the concept of Satyagraha, which translates to "truth force" or "soul force." This philosophy advocated for civil disobedience and passive resistance as powerful tools for social and political change. Gandhi's leadership was instrumental during various movements, such as the Non-Cooperation Movement, the Salt March, and the Quit India Movement, which collectively galvanized millions of Indians across different strata of society to challenge the British colonial government. His ability to inspire mass participation while maintaining a strict adherence to nonviolence earned him not only national reverence but also global recognition as a champion of peace and human rights.

Beyond his political endeavors, Gandhi's influence extended into social and cultural realms, where he championed causes such as the eradication of untouchability, the promotion of rural self-reliance, and the upliftment of women. He envisioned an India that was self-sufficient, harmonious, and free from social injustices. Gandhi's simple lifestyle and ascetic practices, including his use of the spinning wheel (charkha) as a symbol of economic independence, resonated deeply with the common people and became emblematic of his philosophy. His writings and speeches continue to inspire movements for civil rights and freedom across the world. Tragically, Gandhi's life was cut short when he was assassinated on January 30, 1948, but his legacy endures, embodying the principles of nonviolence, justice, and the relentless pursuit of truth."""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)

# Char-level splits
from langchain_text_splitters import RecursiveCharacterTextSplitter

chunk_size = 150
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Split
splits = text_splitter.split_documents(md_header_splits)

for doc in splits:
    print("**********************************")
    print("***** METADATA *****")
    print(doc.metadata)
    print("***** CONTENT *****")
    print(doc.page_content)
    print("\n")

from anthropic import Anthropic

max_token_count = 12000 #property of Claude 2

client = Anthropic()

def count_tokens(text):
    return client.count_tokens(text)

def get_chunks(full_text, OVERLAP=True, DEBUG=False):
    '''
    This will take a text and return an array with sliced chunks of the text in optimal sizing for summarization.  Note that by default, this does include overlaping text in each chunk.
    Overlaping allows more cohesion between text, and should only be turned off when trying to count specific numbers and no duplicated text is a requirment.

    We could just drop text up to the maximum context window of our model, but that actually doesn't work very well.
    Part of the reason for this is because no matter the input length, the output length is about the same.
    For example, if you drop in a paragraph or 10 pages, you get about a paragraph in response.
    To mitigate this, we create chunks using the lesser of two values: 25% of the total token count or 2k tokens.
    We'll also overlap our chunks by about a paragraph of text or so, in order to provide continuity between chunks.
    (Logic taken from https://gist.github.com/Donavan/4fdb489a467efdc1faac0077a151407a)
    '''
    # DEBUG = False  # debugging at this level is usually not very helpful.

    # Following testing, it was found that chunks should be 2000 tokens, or 25% of the doc, whichever is shorter.
    # max chunk size in tokens
    chunk_length_tokens = 2000
    # chunk length may be shortened later for shorter docs.

    # a paragraph is about 200 words, which is about 260 tokens on average
    # we'll overlap our chunks by a paragraph to provide cohesion to the final summaries.
    overlap_tokens = 260
    if not OVERLAP: overlap_tokens = 0

    # anything this short doesn't need to be chunked further.
    min_chunk_length = 260 + overlap_tokens * 2

    # grab basic info about the text to be chunked.
    char_count = len(full_text)
    word_count = len(full_text.split(" "))  # rough estimate
    token_count = count_tokens(full_text)
    token_per_charater = token_count / char_count

    # don't chunk tiny texts
    if token_count <= min_chunk_length:
        if DEBUG: print("Text is too small to be chunked further")
        return [full_text]

    if DEBUG:
        print("Chunk DEBUG mode is on, information about the text and chunking will be printed out.")
        print("Estimated character count:", char_count)
        print("Estimated word count:", word_count)
        print("Estimated token count:", token_count)
        print("Estimated tokens per character:", token_per_charater)

        print("Full text tokens: ", count_tokens(full_text))
        print("How many times bigger than max context window: ", round(count_tokens(full_text) / max_token_count, 2))

    # if the text is shorter, use smaller chunks
    if (token_count / 4 < chunk_length_tokens):
        overlap_tokens = int((overlap_tokens / chunk_length_tokens) * int(token_count / 4))
        chunk_length_tokens = int(token_count / 4)

        if DEBUG:
            print("Short doc detected:")
            print("New chunk length:", chunk_length_tokens)
            print("New overlap length:", overlap_tokens)

    # convert to charaters for easy slicing using our approximate tokens per character for this text.
    overlap_chars = int(overlap_tokens / token_per_charater)
    chunk_length_chars = int(chunk_length_tokens / token_per_charater)

    # itterate and create the chunks from the full text.
    chunks = []
    start_chunk = 0
    end_chunk = chunk_length_chars + overlap_chars

    last_chunk = False
    while not last_chunk:
        # the last chunk may not be the full length.
        if (end_chunk >= char_count):
            end_chunk = char_count
            last_chunk = True
        chunks.append(full_text[start_chunk:end_chunk])

        # move our slice location
        if start_chunk == 0:
            start_chunk += chunk_length_chars - overlap_chars
        else:
            start_chunk += chunk_length_chars

        end_chunk = start_chunk + chunk_length_chars + 2 * overlap_chars

    if DEBUG: print("Created %s chunks." % len(chunks))
    return chunks






if __name__ == '__main__':
    text_to_open_short = '../sample texts/artificial_intelligence_essay.txt'
    with open(text_to_open_short) as f:
        full_text = f.read()
    print(len(full_text))
    chunks = get_chunks(full_text, OVERLAP=True, DEBUG=True)
    print(chunks)
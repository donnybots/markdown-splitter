import re

import mistune.renderers
import mistune.renderers.markdown
from transformers import GPT2TokenizerFast

import mistune
from mistune.renderers.markdown import MarkdownRenderer
from pprint import pprint

def parse_markdown_into_blocks(markdown_text):
    """
    Parses the markdown text into a list of blocks (e.g., headings, paragraphs, code blocks).
    """
    parser = mistune.create_markdown(renderer='ast')
    ast = parser(markdown_text)
    blocks = []

    pprint(ast)

    def extract_text(node):
        if node['type'] == 'text':
            return node['raw']
        elif node['type'] == 'heading':
            level = node['attrs']['level']
            text = ''.join([extract_text(child) for child in node['children']])
            return f"{'#' * level} {text}\n"
        elif node['type'] == 'paragraph':
            text = ''.join([extract_text(child) for child in node['children']])
            return f"{text}\n\n"
        elif 'children' in node:
            return ''.join([extract_text(child) for child in node['children']])
        return ''

    for node in ast:
        block_text = extract_text(node)
        # Reconstruct the markdown from the AST node
        render_markdown = mistune.create_markdown(renderer=MarkdownRenderer())
        rendered_text = render_markdown(block_text)
        blocks.append(rendered_text)

    return blocks

def split_large_block(block_text, tokenizer, max_tokens=200):
    """
    Splits a large block into smaller chunks without breaking words.
    """
    words = block_text.split()
    chunks = []
    current_chunk = ''
    current_token_count = 0

    for word in words:
        word_token_count = len(tokenizer.encode(word + ' '))
        if current_token_count + word_token_count <= max_tokens:
            current_chunk += word + ' '
            current_token_count += word_token_count
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + ' '
            current_token_count = word_token_count

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def split_markdown_by_tokens(markdown_text, max_tokens=200, tokenizer=None):
    """
    Splits the markdown text into chunks without breaking formatting.

    Args:
        markdown_text (str): The markdown content to split.
        max_tokens (int): The maximum number of tokens per chunk.

    Returns:
        List[str]: A list of markdown chunks.
    """
    if not isinstance(markdown_text, str):
        raise TypeError("markdown_text must be a string.")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")

    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    blocks = parse_markdown_into_blocks(markdown_text)
    chunks = []
    current_chunk = ''
    current_token_count = 0

    for block in blocks:
        block_token_count = len(tokenizer.encode(block))

        if current_token_count + block_token_count <= max_tokens:
            current_chunk += block
            current_token_count += block_token_count
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Start a new chunk
            current_chunk = block
            current_token_count = block_token_count

            # If a single block exceeds max_tokens, we need to split it further
            if current_token_count > max_tokens:
                sub_blocks = split_large_block(block, tokenizer, max_tokens)
                chunks.extend(sub_blocks)
                current_chunk = ''
                current_token_count = 0

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


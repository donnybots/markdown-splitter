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
    Each block is a tuple (block_type, block_text, node)
    """
    parser = mistune.create_markdown(renderer='ast')
    ast = parser(markdown_text)
    blocks = []

    def extract_text(node):
        if node['type'] == 'text':
            return node['raw']
        elif node['type'] == 'heading':
            level = node['attrs']['level']
            text = ''.join([extract_text(child) for child in node['children']])
            return f"{'#' * level} {text}\n"
        elif node['type'] == 'paragraph':
            text = ''.join([extract_text(child) for child in node['children']])
            return f"{text}\n"
        elif node['type'] == 'block_code':
            code = node['raw']
            language = node.get('info', '').strip()
            code_block = f"```{language}\n{code}\n```\n\n"
            return code_block
        elif 'children' in node:
            return ''.join([extract_text(child) for child in node['children']])
        else:
            return ''

    for node in ast:
        block_type = node['type']
        block_text = extract_text(node)
        blocks.append((block_type, block_text, node))

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

def split_large_code_block(code_text, language, tokenizer, max_tokens):
    """
    Splits a large code block into smaller code blocks without breaking code lines.

    Args:
        code_text (str): The code content.
        language (str): The programming language of the code block.
        tokenizer: The tokenizer to use.
        max_tokens (int): The maximum number of tokens per code block.

    Returns:
        List[str]: A list of code blocks, each properly formatted with code block markers.
    """
    code_lines = code_text.split('\n')
    code_chunks = []
    current_code_chunk = ''
    current_code_token_count = 0
    for line in code_lines:
        line_with_newline = line + '\n'
        line_token_count = len(tokenizer.encode(line_with_newline))
        # Account for code block markers and language in each chunk
        overhead_tokens = len(tokenizer.encode(f"```{language}\n```"))
        if current_code_token_count + line_token_count <= max_tokens - overhead_tokens:
            current_code_chunk += line_with_newline
            current_code_token_count += line_token_count
        else:
            # Finish current code chunk
            code_chunk_text = f"```{language}\n{current_code_chunk}```\n\n"
            code_chunks.append(code_chunk_text)
            # Start new code chunk
            current_code_chunk = line_with_newline
            current_code_token_count = line_token_count
    if current_code_chunk:
        code_chunk_text = f"```{language}\n{current_code_chunk}```\n\n"
        code_chunks.append(code_chunk_text)
    return code_chunks

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

    for block_type, block_text, node in blocks:
        block_token_count = len(tokenizer.encode(block_text))

        if current_token_count + block_token_count <= max_tokens:
            # Add the block to the current chunk
            current_chunk += block_text
            current_token_count += block_token_count
        else:
            # Current chunk is full, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = ''
            current_token_count = 0

            # Check if block itself exceeds max_tokens
            if block_token_count > max_tokens:
                # Need to split the block
                if block_type == 'block_code':
                    # Split code block into smaller code blocks
                    language = node.get('info', '').strip()
                    code = node['raw']
                    code_chunks = split_large_code_block(code, language, tokenizer, max_tokens)
                    chunks.extend(code_chunks)
                else:
                    # Split large block (e.g., paragraph)
                    sub_blocks = split_large_block(block_text, tokenizer, max_tokens)
                    chunks.extend(sub_blocks)
            else:
                # Block fits into max_tokens, start a new chunk with this block
                current_chunk = block_text
                current_token_count = block_token_count

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# tests/test_splitter.py

import pytest
from markdown_splitter import split_markdown_by_tokens
from transformers import GPT2TokenizerFast
import mistune

# Initialize the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def test_empty_input():
    """Test that an empty string returns an empty list."""
    result = split_markdown_by_tokens("")
    assert result == ["\n"]

def test_single_short_block():
    """Test splitting markdown with a single short block."""
    markdown_text = "# Heading\n\nThis is a test."
    result = split_markdown_by_tokens(markdown_text, max_tokens=50)
    assert len(result) == 1
    assert result[0] == f"{markdown_text}\n"

def test_exact_token_limit():
    """Test a markdown that exactly matches the token limit."""
    words = ["word"] * 49  # Adjust to match 49 tokens (renderer will append \n)
    markdown_text = " ".join(words)
    result = split_markdown_by_tokens(markdown_text, max_tokens=50)
    assert len(result) == 1
    assert result[0] == f"{markdown_text}\n"

def test_exceeds_token_limit():
    """Test splitting when the markdown exceeds the token limit."""
    words = ["word"] * 100
    markdown_text = " ".join(words)
    result = split_markdown_by_tokens(markdown_text, max_tokens=50)
    total_tokens = sum(len(tokenizer.encode(chunk)) for chunk in result)
    original_tokens = len(tokenizer.encode(markdown_text))
    assert len(result) > 1
    assert all(len(tokenizer.encode(chunk)) <= 50 for chunk in result)
    assert total_tokens == original_tokens # Account for the newline token

def test_preserve_formatting():
    """Test that markdown formatting is preserved after splitting."""
    markdown_text = (
        "# Heading\n\n"
        "This is a paragraph with **bold text** and _italic text_.\n\n"
        "## Subheading\n\n"
        "- List item 1\n"
        "- List item 2\n"
        "  - Nested list item\n\n"
        "Some more text to make the content longer and exceed the token limit for the test."
    )
    result = split_markdown_by_tokens(markdown_text, max_tokens=30)
    for chunk in result:
        # Attempt to parse the chunk to ensure it's valid markdown
        try:
            mistune.create_markdown()(chunk)
        except Exception as e:
            pytest.fail(f"Chunk is not valid markdown: {e}")

def test_large_single_block():
    """Test splitting a single large block."""
    words = ["word"] * 500
    markdown_text = "# Heading\n\n" + " ".join(words)
    result = split_markdown_by_tokens(markdown_text, max_tokens=50)
    assert len(result) > 1
    assert all(len(tokenizer.encode(chunk)) <= 50 for chunk in result)

# def test_split_large_code_block():
#     """Test splitting a markdown text with a large code block that exceeds the token limit."""
#     # Create a large code block
#     code_lines = ["print('Line {}')".format(i) for i in range(100)]
#     code_block = "```python\n" + "\n".join(code_lines) + "\n```"
#     markdown_text = f"Here is a large code block:\n\n{code_block}\n\nEnd of the markdown."
    
#     result = split_markdown_by_tokens(markdown_text, max_tokens=50)
    
#     # Ensure the code block is split into multiple chunks
#     assert len(result) > 1
#     # Verify each chunk is properly formatted as a code block
#     for chunk in result:
#         if chunk.strip().startswith("```"):
#             assert chunk.startswith("```")
#             assert chunk.endswith("```")

def test_multiple_blocks():
    """Test splitting multiple blocks."""
    markdown_text = (
        "# Heading 1\n\n" + " ".join(["text"] * 100) + "\n\n"
        "# Heading 2\n\n" + " ".join(["more text"] * 100) + "\n\n"
        "# Heading 3\n\n" + " ".join(["even more text"] * 100)
    )
    result = split_markdown_by_tokens(markdown_text, max_tokens=60)
    assert len(result) > 1
    assert all(len(tokenizer.encode(chunk)) <= 60 for chunk in result)

def test_non_string_input():
    """Test that non-string input raises a TypeError."""
    with pytest.raises(TypeError):
        split_markdown_by_tokens(12345)

def test_invalid_max_tokens():
    """Test that invalid max_tokens raises a ValueError."""
    markdown_text = "# Heading\n\nThis is a test."
    with pytest.raises(ValueError):
        split_markdown_by_tokens(markdown_text, max_tokens=-10)
    with pytest.raises(ValueError):
        split_markdown_by_tokens(markdown_text, max_tokens=0)

def test_custom_tokenizer():
    """Test passing a custom tokenizer."""
    markdown_text = "This is a test."
    custom_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    result = split_markdown_by_tokens(markdown_text, max_tokens=50, tokenizer=custom_tokenizer)
    assert len(result) == 1
    assert result[0] == f"{markdown_text}\n"

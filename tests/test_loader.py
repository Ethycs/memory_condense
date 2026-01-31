from pathlib import Path

from memory_condense.loader import load_conversation, load_directory, parse_md, parse_txt


def test_parse_txt_basic():
    text = (
        "User:\n"
        "Hello world\n"
        "\n"
        "Claude:\n"
        " Hi there! How can I help?\n"
        "\n"
        "User:\n"
        "Tell me about Python\n"
    )
    turns = parse_txt(text)
    assert len(turns) == 3
    assert turns[0] == ("user", "Hello world")
    assert turns[1] == ("assistant", "Hi there! How can I help?")
    assert turns[2] == ("user", "Tell me about Python")


def test_parse_txt_multiline():
    text = (
        "User:\n"
        "First line\n"
        "Second line\n"
        "\n"
        "Claude:\n"
        " Response line one\n"
        " Response line two\n"
    )
    turns = parse_txt(text)
    assert len(turns) == 2
    assert "First line" in turns[0][1]
    assert "Second line" in turns[0][1]
    assert "Response line one" in turns[1][1]


def test_parse_txt_skips_empty_bodies():
    text = "User:\n\nClaude:\n \n\nUser:\nActual content\n"
    turns = parse_txt(text)
    assert len(turns) == 1
    assert turns[0][1] == "Actual content"


def test_parse_md_basic():
    text = (
        "**User:**\n"
        "\n"
        "Hello world\n"
        "\n"
        "**Assistant:**\n"
        "\n"
        "Hi there! How can I help?\n"
    )
    turns = parse_md(text)
    assert len(turns) == 2
    assert turns[0] == ("user", "Hello world")
    assert turns[1] == ("assistant", "Hi there! How can I help?")


def test_parse_md_skips_empty_assistant():
    text = (
        "**Assistant:**\n"
        "\n"
        "\n"
        "\n"
        "**Assistant:**\n"
        "\n"
        "Actual response here\n"
    )
    turns = parse_md(text)
    assert len(turns) == 1
    assert turns[0] == ("assistant", "Actual response here")


def test_load_conversation_txt(tmp_path: Path):
    f = tmp_path / "chat.txt"
    f.write_text("User:\nHello\n\nClaude:\n Reply\n", encoding="utf-8")
    turns = load_conversation(f)
    assert len(turns) == 2
    assert turns[0][0] == "user"
    assert turns[1][0] == "assistant"


def test_load_conversation_md(tmp_path: Path):
    f = tmp_path / "chat.md"
    f.write_text("**User:**\nHello\n\n**Assistant:**\nReply\n", encoding="utf-8")
    turns = load_conversation(f)
    assert len(turns) == 2


def test_load_directory(tmp_path: Path):
    (tmp_path / "a.txt").write_text("User:\nHello\n\nClaude:\n Hi\n")
    (tmp_path / "b.md").write_text("**User:**\nHey\n\n**Assistant:**\nHi\n")
    (tmp_path / "c.json").write_text("{}")  # should be skipped

    result = load_directory(tmp_path)
    assert "a.txt" in result
    assert "b.md" in result
    assert "c.json" not in result
    assert len(result["a.txt"]) == 2
    assert len(result["b.md"]) == 2


def test_load_real_txt_format():
    """Test against the actual format seen in the user's files."""
    text = (
        "User:\n"
        "Can I use a transformer to calculate shannon entropy\n"
        "\n"
        "Claude:\n"
        " You can use a transformer to calculate Shannon entropy, "
        "though it's not the most straightforward application.\n"
        "\n"
        "User:\n"
        "basically I want to calculate conditional probability\n"
    )
    turns = parse_txt(text)
    assert len(turns) == 3
    assert turns[0][0] == "user"
    assert "shannon entropy" in turns[0][1].lower()
    assert turns[1][0] == "assistant"
    assert turns[2][0] == "user"


def test_load_real_md_format():
    """Test against the actual format seen in the user's .md files."""
    text = (
        "**Assistant:**\n"
        "\n"
        "\n"
        "\n"
        "**Assistant:**\n"
        "\n"
        "The phrase **genericity** is about singularity theory.\n"
        "\n"
        "**User:**\n"
        "\n"
        "Tell me more about Whitney stratification\n"
    )
    turns = parse_md(text)
    assert len(turns) == 2
    assert turns[0][0] == "assistant"
    assert "genericity" in turns[0][1]
    assert turns[1][0] == "user"

from llm import _clean_env_value


def test_clean_env_value_strips_whitespace_quotes_and_newlines():
    assert _clean_env_value(' \r\n"  example-key  "\r\n') == "example-key"


def test_clean_env_value_handles_single_quotes():
    assert _clean_env_value("'example-key'") == "example-key"

from src.web.security import is_allowed_web_origin


def test_browser_mutations_only_allow_loopback_origins():
    assert is_allowed_web_origin("http://127.0.0.1:8000")
    assert is_allowed_web_origin("http://localhost:8000")
    assert not is_allowed_web_origin("https://attacker.example")
    assert not is_allowed_web_origin("null")

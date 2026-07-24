from src.fixers.alt_text_generator import AltTextGenerator
from src.fixers.image_optimizer import ImageOptimizer
from src.fixers.jsonld_generator import JsonLdGenerator
from src.fixers.link_fixer import LinkFixer
from src.fixers.robots_txt_fixer import RobotsTxtFixer


def test_judgment_based_fixes_require_review():
    assert AltTextGenerator.fix_type == "semi_auto"
    assert ImageOptimizer.fix_type == "semi_auto"
    assert JsonLdGenerator.fix_type == "semi_auto"
    assert LinkFixer.fix_type == "semi_auto"
    assert RobotsTxtFixer.fix_type == "semi_auto"

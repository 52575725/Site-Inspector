from src.fixers.alt_text_generator import AltTextGenerator
from src.fixers.canonical_fixer import CanonicalFixer
from src.fixers.hreflang_fixer import HreflangFixer
from src.fixers.image_optimizer import ImageOptimizer
from src.fixers.jsonld_generator import JsonLdGenerator
from src.fixers.link_fixer import LinkFixer
from src.fixers.robots_txt_fixer import RobotsTxtFixer
from src.fixers.sitemap_fixer import SitemapFixer


def test_judgment_based_fixes_require_review():
    assert AltTextGenerator.fix_type == "semi_auto"
    assert ImageOptimizer.fix_type == "semi_auto"
    assert JsonLdGenerator.fix_type == "semi_auto"
    assert LinkFixer.fix_type == "semi_auto"
    assert RobotsTxtFixer.fix_type == "semi_auto"
    assert CanonicalFixer.fix_type == "semi_auto"
    assert HreflangFixer.fix_type == "semi_auto"
    assert SitemapFixer.fix_type == "semi_auto"

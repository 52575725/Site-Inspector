from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Sequence

from src.storage.models import Fix, Issue, Scan

# ── Issue knowledge base ─────────────────────────────────────────────
# Each entry explains: what the problem is, why it matters, how to fix it,
# and what improvement to expect after fixing.

ISSUE_KB: dict[str, dict[str, str]] = {
    # ── SEO ──
    "missing_title": {
        "what": "页面缺少 <title> 标签",
        "impact": "搜索引擎无法识别页面主题，搜索结果中会显示网址而不是页面标题，严重影响点击率和排名",
        "how": "给每个页面添加唯一的描述性标题，30-60 字符，包含核心关键词",
        "outcome": "Google 搜索结果显示完整标题，预计 CTR 提升 5-15%，核心关键词排名有望上升 1-3 位",
    },
    "title_too_short": {
        "what": "页面标题过短（少于 30 字符）",
        "impact": "标题信息量不足，搜索引擎可能无法充分理解页面内容，降低长尾关键词排名机会",
        "how": "扩展标题至 30-60 字符，包含主要关键词和品牌名",
        "outcome": "更多长尾关键词覆盖，搜索曝光机会增加",
    },
    "title_too_long": {
        "what": "页面标题超过 60 字符",
        "impact": "搜索结果显示时会被截断（显示'...'），用户看到的标题不完整，降低点击意愿",
        "how": "精简标题到 60 字符以内，保留核心关键词",
        "outcome": "搜索结果中标题完整展示，用户更容易理解页面内容，CTR 预计提升",
    },
    "missing_meta_description": {
        "what": "缺少 meta description 标签",
        "impact": "Google 会自行从页面提取摘要，质量不可控；搜索结果中的描述可能包含无关信息或 JS 代码片段",
        "how": "为每个页面写 120-160 字符的描述文案，包含关键词和价值主张，引导用户点击",
        "outcome": "搜索结果显示可控的高质量描述，CTR 预计提升 3-8%",
    },
    "meta_description_too_short": {
        "what": "meta description 过短（少于 120 字符）",
        "impact": "描述信息不够充实，浪费了搜索结果中宝贵的展示空间，说服用户点击的机会减少",
        "how": "将描述扩展至 120-160 字符，补充产品特点、服务优势或行动号召",
        "outcome": "搜索结果展示更完整，用户可以在搜索页就了解页面价值",
    },
    "meta_description_too_long": {
        "what": "meta description 超过 160 字符",
        "impact": "搜索结果中会被截断，关键信息可能无法完整展示给用户",
        "how": "精简描述到 120-160 字符，把最重要的信息放在前面",
        "outcome": "搜索结果描述完整展示，关键卖点对用户可见",
    },
    "missing_h1": {
        "what": "页面缺少 H1 标题标签",
        "impact": "H1 是页面主题的最强信号，缺少 H1 会使搜索引擎难以确定页面核心主题，影响排名",
        "how": "添加一个 H1 标签，包含页面最主要的关键词，每个页面只保留一个 H1",
        "outcome": "搜索引擎能准确理解页面主题，核心关键词排名更稳定",
    },
    "multiple_h1": {
        "what": "页面有多个 H1 标签",
        "impact": "多个 H1 会分散主题信号，搜索引擎无法确定哪个是核心主题，削弱 SEO 效果",
        "how": "保留一个 H1 作为主标题，其余改为 H2 或 H3",
        "outcome": "页面主题信号清晰，搜索引擎更容易给精准排名",
    },
    "h_tag_skip": {
        "what": "标题层级跳跃（如 H1 直接到 H3，缺少 H2）",
        "impact": "破坏文档结构和语义层次，影响搜索引擎理解内容结构；对屏幕阅读器用户不友好",
        "how": "按层级使用标题标签：H1 → H2 → H3，不要跳级",
        "outcome": "文档结构清晰，搜索引擎和辅助工具都能正确理解页面层次",
    },
    "missing_canonical": {
        "what": "缺少 canonical 标签",
        "impact": "如果有多个 URL 指向相同内容（如带/不带斜杠、带参数），Google 会认为存在重复内容，分散页面权重",
        "how": "添加 <link rel='canonical' href='...'> 指向页面的规范 URL",
        "outcome": "页面权重集中，避免重复内容导致的排名稀释",
    },
    "missing_hreflang": {
        "what": "多语言页面缺少 hreflang 标签",
        "impact": "搜索引擎可能将日语页面展示给英文用户（反之亦然），搜索结果语言不匹配导致高跳出率",
        "how": "每个语言页面添加 hreflang 标签，标注 en/ja/x-default",
        "outcome": "日本用户看到日语页面，英文用户看到英文页面，语言匹配后跳出率降低，停留时间增加",
    },
    "incomplete_hreflang": {
        "what": "hreflang 标签不完整，缺少某些语言版本的对应关系",
        "impact": "搜索引擎可能无法识别页面间的双语对应关系，两国用户可能都看不到最适合的页面版本",
        "how": "确保每个页面都标注所有语言版本（en、ja）和 x-default（默认语言）",
        "outcome": "双语 SEO 完整生效，JP 域名在日本搜索中的曝光率提升",
    },
    "missing_og_tags": {
        "what": "缺少 Open Graph 标签",
        "impact": "当用户在社交媒体、LINE、WhatsApp 中分享链接时，无法显示标题、描述和图片预览卡片，分享效果大打折扣",
        "how": "添加 og:title, og:description, og:image, og:url, og:type 标签",
        "outcome": "社交媒体分享时自动生成精美预览卡片，提升品牌形象和点击率",
    },
    "missing_jsonld": {
        "what": "页面缺少结构化数据（JSON-LD）",
        "impact": "Google 无法在搜索结果中展示富文本摘要（如星级、价格、FAQ、面包屑），错失额外曝光机会",
        "how": "根据页面类型添加相应的 JSON-LD：Organization、Product、Article、BreadcrumbList",
        "outcome": "搜索结果可能出现富文本摘要，视觉上更突出，CTR 预计提升 10-20%",
    },
    "missing_viewport_meta": {
        "what": "缺少 viewport meta 标签",
        "impact": "移动设备上页面可能以桌面宽度缩放显示，文字极小、需要缩放操作，Google 移动优先索引会降低排名",
        "how": "在 <head> 中添加 <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "outcome": "页面在手机上正常缩放显示，通过 Google 移动友好测试，移动排名不受影响",
    },
    "missing_charset": {
        "what": "缺少字符编码声明",
        "impact": "浏览器可能错误猜测编码导致中文出现乱码，用户体验极差",
        "how": "在 <head> 最前面添加 <meta charset='UTF-8'>",
        "outcome": "所有浏览器正确显示中文字符，避免乱码",
    },
    "excessive_inline_styles": {
        "what": "页面内联样式过多",
        "impact": "增加 HTML 体积，减慢页面加载速度；样式无法被浏览器缓存，每次访问都要重新下载",
        "how": "将重复的内联样式提取到外部 CSS 文件，使用 class 代替 style 属性",
        "outcome": "HTML 体积减小 20-40%，页面加载更快（尤其对重复访问的用户）",
    },

    # ── Mobile ──
    "small_font_size": {
        "what": "移动端文字过小（小于 16px）",
        "impact": "用户在手机上需要手动缩放才能阅读，iOS Safari 会在小字体上强制缩放导致布局错乱",
        "how": "通过 CSS media query 设置移动端最小字号为 16px，调整相关元素样式",
        "outcome": "移动端阅读体验舒适，不需要缩放操作，用户停留时间可能增加 20%",
    },
    "small_touch_targets": {
        "what": "可点击元素尺寸过小（小于 48px）",
        "impact": "手指难以精准点击，用户容易点到错误的链接或按钮，操作错误率高",
        "how": "通过 CSS 设置链接和按钮的最小触摸区域为 48x48px，增大点击热区",
        "outcome": "移动端误触率降低，用户导航体验流畅",
    },
    "horizontal_scroll": {
        "what": "页面在移动端出现水平滚动条",
        "impact": "用户需要左右滑动才能看到完整内容，这是移动体验中最让人反感的问题之一",
        "how": "为溢出元素添加 max-width: 100%，设置 overflow-x: hidden 兜底",
        "outcome": "页面自适应手机屏幕宽度，无需左右滑动",
    },

    # ── Accessibility ──
    "missing_alt_text": {
        "what": "图片缺少 alt 属性（替代文本）",
        "impact": "视障用户使用屏幕阅读器时无法了解图片内容；图片加载失败时无后备信息显示；Google 图片搜索缺少索引依据",
        "how": "为每张图片添加描述性的 alt 文本，准确说明图片内容",
        "outcome": "满足 WCAG 无障碍标准；Google 图片搜索可为网站导流；视障用户也能完整浏览",
    },
    "empty_alt_text": {
        "what": "图片 alt 属性为空",
        "impact": "屏幕阅读器会跳过该图片；如果图片不是纯装饰性的，视障用户就失去了内容信息",
        "how": "非装饰性图片补上有意义的描述文本；纯装饰性图片添加 role='presentation'",
        "outcome": "无障碍合规，所有用户都能通过不同方式获取图片信息",
    },
    "missing_lang_attribute": {
        "what": "<html> 标签缺少 lang 属性",
        "impact": "屏幕阅读器无法自动切换正确的发音引擎，中文页面可能被用英文语音朗读。搜索引擎也依赖此属性判断页面语言",
        "how": "添加 lang 属性，英文页面用 'en'，日文页面用 'ja'",
        "outcome": "屏幕阅读器自动匹配正确语言发音，搜索引擎正确识别语言版本",
    },
    "missing_form_label": {
        "what": "表单输入框缺少关联的 <label> 标签",
        "impact": "用户点击标签文字无法聚焦到输入框；屏幕阅读器无法告知用户此输入框的用途",
        "how": "为每个 input/select/textarea 添加 <label> 并使用 for 属性关联",
        "outcome": "表单操作更便捷，无障碍合规",
    },
    "missing_iframe_title": {
        "what": "iframe 缺少 title 属性",
        "impact": "屏幕阅读器用户不知道嵌入内容是什么，无法判断是否需要浏览该内容",
        "how": "为每个 iframe 添加描述性的 title 属性",
        "outcome": "嵌入内容对辅助技术用户可识别",
    },

    # ── Content Quality ──
    "thin_content": {
        "what": "页面内容量不足（少于 300 词）",
        "impact": "Google 将内容稀少的页面视为'低质量页面'，可能在索引中降权或不收录，尤其影响日语版页面",
        "how": "扩充页面文字内容至 300 词以上，添加产品细节、FAQ、客户案例等有实际价值的信息",
        "outcome": "页面被 Google 视为有实质内容，索引保留率提高，长尾关键词有机会获得排名",
    },
    "low_readability": {
        "what": "文本可读性差（句子过长或词汇过于复杂）",
        "impact": "用户难以快速理解内容，跳出率增加。对日语页面来说，过于复杂的敬语或长句会让客户困惑",
        "how": "缩短长句，使用更简洁的表达，适当分段和添加列表",
        "outcome": "内容更易于阅读和消化，用户在页面停留时间增长",
    },
    "duplicate_content": {
        "what": "页面内容与站内其他页面高度重复",
        "impact": "Google 会选一个页面索引而忽略其他，重复页面的排名被稀释甚至被标记为低质量",
        "how": "重写重复内容，每页突出独特的价值信息；或使用 canonical 指向规范版本",
        "outcome": "每页都有独特价值，所有页面均可被索引排名",
    },

    # ── Performance ──
    "slow_page": {
        "what": "页面加载速度慢",
        "impact": "加载超过 3 秒将流失 53% 的移动用户；Google Core Web Vitals 不达标会降低搜索排名",
        "how": "优化图片（压缩 + WebP 格式）、减少 JS/CSS 体积、启用浏览器缓存、使用 CDN",
        "outcome": "LCP 降至 2.5 秒以下，通过 Core Web Vitals 评估，搜索排名不受速度拖累",
    },
    "large_page": {
        "what": "页面体积过大",
        "impact": "在移动网络下加载缓慢，消耗用户流量，Google 页面体验评分下降",
        "how": "压缩图片、移除未使用的 CSS/JS、启用 GZIP 压缩",
        "outcome": "页面体积减小 30-50%，移动端加载速度显著提升",
    },

    # ── Broken Links ──
    "mixed_content": {
        "what": "HTTPS 页面中加载了 HTTP 资源（图片/样式/脚本）",
        "impact": "浏览器会阻止加载混合内容（显示'不安全'警告），用户看到锁图标破裂，影响信任感和 SEO",
        "how": "将所有 HTTP 资源链接替换为 HTTPS，或使用协议相对 URL（//开头）",
        "outcome": "浏览器显示完整的绿色锁图标，用户信任度和 SEO 均不受影响",
    },
    "broken_link": {
        "what": "页面存在已失效的链接（404 或 500 错误）",
        "impact": "用户点击后到达错误页面，导致挫败感和跳出；搜索引擎也会降低含有死链的页面评分",
        "how": "替换或删除失效链接，对已移除的页面设置 301 重定向",
        "outcome": "所有链接可达，用户体验完整，搜索引擎爬虫不会遇到死胡同",
    },
    "redirect_chain": {
        "what": "链接经过多次 301/302 跳转才到达目标页面",
        "impact": "每次跳转增加 200-500ms 延迟，超过 3 跳的链接 Google 可能放弃追踪",
        "how": "将跳转链缩短为一步直达，更新所有引用为最终目标 URL",
        "outcome": "页面访问速度提升，爬虫抓取效率更高",
    },

    # ── Sitemap ──
    "sitemap_missing": {
        "what": "网站缺少 sitemap.xml 站点地图文件",
        "impact": "搜索引擎爬虫无法高效发现所有页面，新页面可能延迟数周才被收录；无法通过 Search Console 提交索引请求",
        "how": "生成标准的 XML 站点地图，包含所有页面的 URL、最后修改时间和优先级，提交到 Google Search Console",
        "outcome": "Google 能快速发现和索引所有页面，新内容收录时间从天级缩短到小时级",
    },
    "sitemap_dead_url": {
        "what": "站点地图中包含已失效的 URL（404 页面）",
        "impact": "Google 爬虫访问死链浪费抓取配额（crawl budget），且多次遇到死链会降低站点质量评分",
        "how": "从站点地图中移除失效 URL，确保只保留返回 200 状态码的有效页面",
        "outcome": "爬虫抓取效率提升，所有抓取配额用于有效页面，站点质量信号改善",
    },
    "sitemap_missing_url": {
        "what": "网站页面未包含在站点地图中",
        "impact": "该页面可能无法被搜索引擎发现和索引，尤其对于没有内部链接指向的孤立页面影响更大",
        "how": "将缺失的页面 URL 添加到 sitemap.xml，标注正确的 lastmod 和优先级",
        "outcome": "所有页面都被 Google 发现，网站索引覆盖率接近 100%",
    },
    "sitemap_stale_lastmod": {
        "what": "站点地图中 lastmod 日期过旧（超过 30 天未更新）",
        "impact": "Google 可能认为这些页面内容已过时，降低抓取频率；更新了内容但 Google 不知道，延迟索引更新",
        "how": "每次修改页面后更新 sitemap 中的 lastmod 时间戳为实际修改日期",
        "outcome": "Google 及时知道内容更新并重新抓取，索引内容与实际页面保持同步",
    },
    "sitemap_missing_hreflang": {
        "what": "站点地图中缺少 xhtml:link hreflang 语言标注",
        "impact": "Google 无法通过站点地图识别双语页面关系，可能不如 hreflang HTML 标签效果好",
        "how": "在 sitemap.xml 的每个 <url> 中添加 xhtml:link 标注对应的语言版本 URL",
        "outcome": "Google 从站点地图层面就了解语言版本关系，双语 SEO 效果更稳定",
    },

    # ── Structured Data ──
    "schema_missing_type": {
        "what": "页面缺少必要的结构化数据类型（如产品页缺少 Product schema）",
        "impact": "Google 无法为这类页面生成特定的富文本搜索结果（如产品价格、星级、面包屑导航），错失视觉化搜索展示机会",
        "how": "根据页面类型添加对应的 Schema.org JSON-LD：Organization（所有页）、BreadcrumbList（所有页）、Product（产品页）、Article（文章页）",
        "outcome": "搜索结果中可能出现富文本摘要，视觉上更加突出，CTR 预计提升 10-20%",
    },
    "schema_missing_field": {
        "what": "JSON-LD 结构化数据中缺少必需字段",
        "impact": "Google 可能因数据不完整而拒绝展示富文本摘要，或仅部分展示，效果打折扣",
        "how": "根据 Schema.org 规范补全所有必需字段（如 Organization 必须包含 name 和 url）",
        "outcome": "结构化数据完整有效，Google 能顺利生成富文本摘要",
    },
    "schema_invalid_value": {
        "what": "JSON-LD 结构化数据中某个字段的值格式不正确或无效",
        "impact": "Google 结构化数据测试工具会报错，富文本摘要可能不显示或显示异常",
        "how": "修正无效字段值（如错误的 URL 格式、无效的 email 地址等）",
        "outcome": "通过 Google 富文本测试工具验证，无格式错误",
    },
    "schema_duplicate": {
        "what": "同一个 Schema.org 类型在页面中出现了多次",
        "impact": "Google 可能混淆不知道哪个才是正确数据，多个声明可能导致富文本摘要无法正常展示",
        "how": "检查并保留一个正确的 JSON-LD 声明，删除重复的声明",
        "outcome": "数据结构清晰，Google 能准确解析页面结构化信息",
    },

    # ── Content Gap ──
    "content_gap_section": {
        "what": "日语页面缺少英文版中存在的某个内容区块",
        "impact": "日本用户看到的页面信息不完整，可能缺少产品介绍、FAQ 或关键功能说明，影响转化率",
        "how": "将英文版相应区块翻译并添加到日语页面中，确保双语内容一致性",
        "outcome": "日语用户体验与英文版一致，信息完整，日本市场的咨询和转化率预计提升",
    },
    "content_gap_word_count": {
        "what": "日语页面的内容量远少于对应的英文页面（少于 50%）",
        "impact": "日语页面在 Google 日本搜索中可能被视为'内容单薄'，排名降低；日本用户获得的信息远少于英文用户",
        "how": "扩充日语页面内容至英文版的 60-80% 水平，涵盖所有关键信息和产品细节",
        "outcome": "日语页面内容充实，在日本市场搜索排名更稳定，用户信任度提升",
    },
    "content_gap_links": {
        "what": "日语页面内部链接数量远少于英文页面",
        "impact": "日语用户在页面上可导航的路径更少，站内链接结构不均衡，日文页面的 PageRank 分配可能不足",
        "how": "在日语页面中添加与英文版同等的导航链接和 CTA 按钮",
        "outcome": "日语站点的内部链接结构与英文版一致，用户导航流畅，SEO 权重分配均衡",
    },

    # ── Twitter Card & Image SEO ──
    "missing_twitter_cards": {
        "what": "页面缺少 Twitter Card 标签（twitter:card, twitter:title, twitter:description, twitter:image）",
        "impact": "在 Twitter/X 平台上分享链接时无法生成预览卡片，只能显示纯文字链接，品牌形象和点击率都会受影响",
        "how": "添加 Twitter Card meta 标签（通常与 OG 标签内容一致），至少设置 twitter:card 为 summary_large_image",
        "outcome": "在 Twitter/X 上分享可生成大图预览卡片，社交分享效果与 Facebook/LINE 一致",
    },
    "missing_og_image": {
        "what": "页面缺少 og:image 标签",
        "impact": "在社交媒体分享时没有预览图片，只有标题和文字，视觉吸引力大幅降低，严重影响分享点击率",
        "how": "为每个页面设置 og:image 标签，指向代表该页面内容的图片（推荐 1200×630px）",
        "outcome": "社交媒体分享卡片包含精美图片，品牌视觉一致性提升，分享点击率预计提升 30-50%",
    },
    "image_missing_alt": {
        "what": "图片缺少 alt 属性",
        "impact": "Google 图片搜索无法理解图片内容（错失流量来源）；视障用户无法通过屏幕阅读器了解图片信息",
        "how": "为每张有意义的图片添加描述性 alt 文本，包含相关关键词但不堆砌",
        "outcome": "图片出现在 Google 图片搜索结果中，为网站带来额外流量；满足无障碍合规要求",
    },
    "image_empty_alt": {
        "what": "图片 alt 属性为空字符串",
        "impact": "如果图片不是纯装饰性的，Google 和屏幕阅读器都会完全忽略该图片，意味着内容丢失",
        "how": "对于有信息内容的图片补上描述性 alt 文本；仅对纯装饰性图片保留空 alt 并添加 role='presentation'",
        "outcome": "所有内容图片都能被搜索引擎和辅助技术正确理解",
    },

    # ── Internal Links ──
    "internal_orphan_page": {
        "what": "页面没有任何站内其他页面链接指向它（孤立页面）",
        "impact": "搜索引擎爬虫可能无法发现和索引该页面；即使被索引，没有内部链接的页面权重极低，几乎不可能获得排名",
        "how": "从相关页面（如导航、博客相关推荐、产品列表）添加指向该页面的链接",
        "outcome": "页面能被搜索引擎发现和索引，通过内部链接获得权重传递，有排名可能性",
    },
    "internal_deep_page": {
        "what": "页面距离首页超过 3 次点击",
        "impact": "Google 认为深层页面的重要性较低，可能会降低抓取频率或完全不索引深层页面",
        "how": "优化网站导航结构，确保重要页面在 3 次点击内可达；添加面包屑导航和全局导航链接",
        "outcome": "所有页面在 3 次点击内可达，爬虫抓取效率提升，深层页面也能获得索引和排名",
    },

    # ── Generic fallback ──
    "_wcag": {
        "what": "页面存在 WCAG 无障碍标准违规",
        "impact": "可能不符合某些地区的法律要求（如 ADA 合规），且对残障用户不友好",
        "how": "根据具体 WCAG 规则修复对应问题，常见包括对比度不足、缺少 ARIA 标签等",
        "outcome": "网页对所有人可访问，满足无障碍合规要求",
    },
}

_FALLBACK_KB = {
    "what": "检测到网站优化问题",
    "impact": "可能影响搜索引擎排名或用户体验",
    "how": "建议人工审查该问题并根据具体情况修复",
    "outcome": "修复后将改善网站整体质量和用户体验",
}


def explain_issue(category: str, description: str = "") -> dict[str, str]:
    """Look up human-friendly explanation for an issue category."""
    if category in ISSUE_KB:
        return ISSUE_KB[category]
    # Fuzzy match for wcag_* categories
    if category.startswith("wcag_"):
        return ISSUE_KB["_wcag"]
    return _FALLBACK_KB


# ── Report template ───────────────────────────────────────────────────

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>网站巡检日报 - {{ date }} | {{ target_name }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, "Microsoft YaHei", "PingFang SC", sans-serif;
               max-width: 900px; margin: 0 auto; padding: 20px;
               color: #1a1a2e; line-height: 1.7; background: #fafbfc; }
        h1 { font-size: 24px; color: #0f3460; border-bottom: 3px solid #0f3460;
             padding-bottom: 12px; margin-bottom: 8px; }
        .meta { color: #6c757d; font-size: 13px; margin-bottom: 24px; }

        /* Score cards */
        .score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                      gap: 12px; margin-bottom: 32px; }
        .score-card { border-radius: 10px; padding: 20px 16px; text-align: center; }
        .score-card.good { background: #d4edda; border: 1px solid #c3e6cb; }
        .score-card.warn { background: #fff3cd; border: 1px solid #ffeaa7; }
        .score-card.bad  { background: #f8d7da; border: 1px solid #f5c6cb; }
        .score-card .value { font-size: 42px; font-weight: 700; line-height: 1.1; }
        .score-card .label { font-size: 13px; margin-top: 4px; color: #495057; }
        .score-card .hint { font-size: 11px; margin-top: 2px; color: #6c757d; }

        /* Section */
        h2 { font-size: 20px; color: #16213e; margin: 32px 0 16px; padding-bottom: 6px;
             border-bottom: 1px solid #dee2e6; }
        h3 { font-size: 16px; color: #e8590c; margin: 20px 0 8px; }
        h3.green { color: #2b8a3e; }

        /* Fix list */
        .fix-list { list-style: none; }
        .fix-list li { padding: 6px 12px; margin: 3px 0; border-radius: 4px;
                       font-size: 14px; background: #e7f5e8; }
        .fix-list li.semi { background: #fff3cd; }

        /* Issue cards */
        .issue-card { background: #fff; border: 1px solid #e9ecef; border-radius: 8px;
                      margin: 16px 0; overflow: hidden;
                      box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
        .issue-card .header { padding: 12px 16px; display: flex; align-items: center;
                              gap: 12px; }
        .issue-card .tier { display: inline-block; padding: 2px 10px; border-radius: 12px;
                            font-size: 12px; font-weight: 700; color: #fff;
                            min-width: 32px; text-align: center; }
        .tier.P0 { background: #dc3545; }
        .tier.P1 { background: #fd7e14; }
        .tier.P2 { background: #ffc107; color: #212529; }
        .tier.P3 { background: #6c757d; }
        .issue-card .category { font-size: 13px; color: #868e96; }
        .issue-card .page-link { font-size: 13px; margin-left: auto; }
        .issue-card .page-link a { color: #0f3460; text-decoration: none; }
        .issue-card .body { padding: 0 16px 16px; }
        .issue-card .row { margin-bottom: 10px; }
        .issue-card .row-label { font-size: 12px; font-weight: 600; color: #495057;
                                 text-transform: uppercase; letter-spacing: 0.5px; }
        .issue-card .row-text { font-size: 14px; color: #212529; margin-top: 2px; }

        /* Summary bar */
        .summary-bar { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
        .summary-item { background: #fff; border: 1px solid #dee2e6; border-radius: 8px;
                        padding: 14px 18px; flex: 1; min-width: 120px; text-align: center; }
        .summary-item .num { font-size: 28px; font-weight: 700; color: #0f3460; }
        .summary-item .desc { font-size: 12px; color: #868e96; }

        .footer { margin-top: 40px; font-size: 12px; color: #adb5bd; text-align: center;
                  border-top: 1px solid #dee2e6; padding-top: 16px; }

        @media (max-width: 600px) {
            body { padding: 12px; }
            .score-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>

<h1>网站智能巡检日报</h1>
<p class="meta">
    <strong>{{ target_name }}</strong> ({{ target_url }}) — {{ date }}
    &nbsp;|&nbsp; 扫描 {{ pages_crawled }} 个页面，发现 {{ total_issues }} 个问题，自动修复 {{ fixes|length }} 项
</p>

<!-- Summary bar -->
<div class="summary-bar">
    <div class="summary-item"><div class="num">{{ pages_crawled }}</div><div class="desc">已扫描页面</div></div>
    <div class="summary-item"><div class="num">{{ total_issues }}</div><div class="desc">发现问题</div></div>
    <div class="summary-item"><div class="num">{{ fixes|length }}</div><div class="desc">自动修复</div></div>
    <div class="summary-item"><div class="num">{{ p0_count }}</div><div class="desc">P0 严重问题</div></div>
    <div class="summary-item"><div class="num">{{ p1_count }}</div><div class="desc">P1 中等问题</div></div>
</div>

<!-- Health scores -->
<h2>各维度健康评分</h2>
<div class="score-grid">
    {% for name, data in scores.items() %}
    <div class="score-card {{ data.css_class }}">
        <div class="value">{{ data.score }}</div>
        <div class="label">{{ data.label }}</div>
        <div class="hint">{{ data.issue_count }} 个问题</div>
    </div>
    {% endfor %}
</div>

<!-- Fix proposals and execution results -->
<h2>修复建议与执行结果 ({{ fixes|length }})</h2>
{% if fixes %}
    <ul class="fix-list">
        {% for fix in fixes %}
        <li class="{{ 'semi' if fix.fix_type == 'semi_auto' else '' }}">
            <strong>{{ "半自动" if fix.fix_type == "semi_auto" else "全自动" }}</strong>
            &mdash; {{ fix.fixer_label }} [{{ fix.status }}]：
            <code>{{ fix.file_path }}</code>
            {% if fix.git_pr_url %} <a href="{{ fix.git_pr_url }}">查看 PR</a>{% endif %}
        </li>
        {% endfor %}
    </ul>
{% else %}
    <p style="color:#868e96">本日无自动修复</p>
{% endif %}

<!-- Top issues with explanations -->
<h2>重点问题详解</h2>
<p style="font-size:13px;color:#868e96;margin-bottom:16px">
    以下列出本次扫描发现的重要问题（P0/P1），每个问题附有影响分析和优化建议。
</p>

{% if explained_issues %}
    {% for item in explained_issues %}
    <div class="issue-card">
        <div class="header">
            <span class="tier {{ item.issue.priority_tier }}">{{ item.issue.priority_tier }}</span>
            <span class="category">{{ item.issue.category }}</span>
            <span class="page-link">
                <a href="{{ item.issue.url }}" target="_blank">
                    {{ item.issue.url.split('/')[-2] or '/' if '/' in item.issue.url[-10:] else item.issue.url.split('/')[-1] or '/' }}
                    &rarr;
                </a>
            </span>
        </div>
        <div class="body">
            <div class="row">
                <div class="row-label">问题描述</div>
                <div class="row-text">{{ item.issue.description[:200] }}</div>
            </div>
            <div class="row">
                <div class="row-label">这是什么问题？</div>
                <div class="row-text">{{ item.explanation.what }}</div>
            </div>
            <div class="row">
                <div class="row-label">会造成什么影响？</div>
                <div class="row-text" style="color:#c92a2a">{{ item.explanation.impact }}</div>
            </div>
            <div class="row">
                <div class="row-label">如何优化？</div>
                <div class="row-text" style="color:#0f3460">{{ item.explanation.how }}</div>
            </div>
            <div class="row">
                <div class="row-label">预期效果</div>
                <div class="row-text" style="color:#2b8a3e">{{ item.explanation.outcome }}</div>
            </div>
        </div>
    </div>
    {% endfor %}
{% else %}
    <p style="color:#868e96">暂无需要重点关注的问题 &#x2705;</p>
{% endif %}

<div class="footer">
    由 Site Inspector 在 {{ generated_at }} 自动生成 &nbsp;|&nbsp; {{ target_name }}
</div>

</body>
</html>"""

# Friendly labels for dimensions
DIMENSION_LABELS = {
    "seo": "SEO 基础优化",
    "accessibility": "无障碍访问",
    "mobile": "移动端适配",
    "performance": "页面性能",
    "content_quality": "内容质量",
    "broken_links": "链接检测",
    "sitemap": "站点地图",
    "structured_data": "结构化数据",
    "content_gap": "双语内容差异",
}

FIXER_LABELS = {
    "meta_fixer": "Meta 标签修复",
    "jsonld_generator": "结构化数据生成",
    "alt_text_generator": "图片 Alt 文本生成",
    "link_fixer": "链接修复",
    "hreflang_fixer": "Hreflang 多语言标签修复",
    "htag_restructurer": "标题层级重构",
    "content_rewriter": "内容质量优化",
    "mobile_css_fixer": "移动端 CSS 修复",
    "og_image_fixer": "OG 图片 & Twitter Card 修复",
    "sitemap_fixer": "站点地图修复",
    "breadcrumb_fixer": "面包屑导航数据生成",
}


class DailyReportGenerator:
    """Generate human-friendly daily inspection report."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, target_name: str, scan: Scan,
                 issues: Sequence[Issue], fixes: Sequence[Fix],
                 target_url: str = "") -> Path:
        from jinja2 import Environment

        scores = self._calculate_scores(issues)
        p0 = sum(1 for i in issues if i.priority_tier == "P0")
        p1 = sum(1 for i in issues if i.priority_tier == "P1")

        # Deduplicate issues by category before explaining (avoid repeating same fix)
        explained = self._build_explanations(issues)

        # Enrich fix data for display
        fix_data = []
        for f in fixes:
            fix_data.append({
                "fix_type": f.fix_type,
                "fixer_label": FIXER_LABELS.get(f.fixer, f.fixer),
                "file_path": f.file_path,
                "git_pr_url": f.git_pr_url,
                "status": f.status,
            })

        template = Environment(autoescape=True).from_string(TEMPLATE)
        html = template.render(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            target_name=target_name,
            target_url=target_url,
            pages_crawled=scan.pages_crawled,
            total_issues=scan.total_issues_found,
            p0_count=p0,
            p1_count=p1,
            scores=scores,
            fixes=fix_data,
            explained_issues=explained,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

        filename = f"daily-report-{scan.id}-{datetime.utcnow():%Y%m%d}.html"
        filepath = self.output_dir / filename
        filepath.write_text(html, encoding="utf-8")
        return filepath

    def generate_console(self, target_name: str, scan: Scan,
                         issues: Sequence[Issue], fixes: Sequence[Fix]) -> str:
        lines = [
            f"=== Site Inspection Daily Report ===",
            f"Target: {target_name}",
            f"Date: {datetime.utcnow():%Y-%m-%d}",
            f"Pages crawled: {scan.pages_crawled}",
            f"Issues found: {scan.total_issues_found}",
            "",
            "--- Health Scores ---",
        ]
        by_inspector = {}
        for issue in issues:
            by_inspector.setdefault(issue.inspector, []).append(issue)
        for insp, items in sorted(by_inspector.items()):
            p0_count = sum(1 for i in items if i.priority_tier == "P0")
            p1_count = sum(1 for i in items if i.priority_tier == "P1")
            label = DIMENSION_LABELS.get(insp, insp)
            lines.append(f"  {label}: {len(items)} issues ({p0_count} P0, {p1_count} P1)")

        if fixes:
            lines.append(f"\n--- Fix Proposals and Results ({len(fixes)}) ---")
            for f in fixes:
                label = FIXER_LABELS.get(f.fixer, f.fixer)
                lines.append(
                    f"  [{f.status}/{f.fix_type}] {label}: {f.file_path or '?'}"
                )

        return "\n".join(lines)

    @staticmethod
    def _build_explanations(issues: Sequence[Issue]) -> list[dict]:
        """Build explained issue list, one per unique category+URL combination."""
        seen = set()
        top = []
        for issue in issues:
            if issue.priority_tier not in ("P0", "P1"):
                continue
            key = (issue.category, issue.url)
            if key in seen:
                continue
            seen.add(key)
            top.append({
                "issue": issue,
                "explanation": explain_issue(issue.category, issue.description or ""),
            })
        # Limit to 30 explained items
        return top[:30]

    PRIORITY_WEIGHTS = {"P0": 8, "P1": 4, "P2": 2, "P3": 1}

    @staticmethod
    def _calculate_scores(issues: Sequence[Issue]) -> dict:
        by_inspector: dict[str, dict[str, int]] = {}
        for issue in issues:
            by_inspector.setdefault(issue.inspector, {}).setdefault(issue.priority_tier, 0)
            by_inspector[issue.inspector][issue.priority_tier] += 1

        def css_for(score: int) -> str:
            if score >= 70:
                return "good"
            if score >= 40:
                return "warn"
            return "bad"

        result = {}
        for name, tier_counts in by_inspector.items():
            total = sum(tier_counts.values())
            penalty = sum(
                DailyReportGenerator.PRIORITY_WEIGHTS.get(tier, 1) * math.log2(1 + count)
                for tier, count in tier_counts.items()
            )
            s = max(5, round(100 - penalty))
            result[name] = {
                "score": s,
                "css_class": css_for(s),
                "issue_count": total,
                "label": DIMENSION_LABELS.get(name, name),
            }
        return result

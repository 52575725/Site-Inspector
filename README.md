# Site Inspector

Site Inspector is a Python 3.12+ website inspection and controlled SEO
remediation agent. It crawls a configured site, prioritizes findings, proposes
or applies eligible fixes, validates modified HTML, and can open a pull request
for review.

The project is designed for controlled repository workflows. Repository writes
are disabled in the web application by default, and only deterministic
`fully_auto` fixers may write without explicit review.

## Capabilities

- Technical SEO: titles, descriptions, canonical URLs, hreflang, sitemaps,
  robots.txt, headings, links, crawl budget, JavaScript SEO, and mobile checks.
- Content analysis: quality, freshness, E-E-A-T signals, keyword overlap,
  content gaps, cannibalization, and competitor gaps.
- Automatic article research: accepts a public website URL, detects the site's
  business and audience from page evidence, proposes relevant keywords, studies
  the heading structure and content patterns of public search-result articles,
  and generates an original site-aligned HTML draft with a saved research report.
- AI editorial strategy: checks current trend/news search-result signals, selects
  the best-fit article type and search intent, and records a timely angle,
  alternative headlines, rationale, and confidence without presenting search
  coverage as verified traffic or virality.
- Authority citations: AI may select only exact URLs returned by the current
  research run, adds descriptive inline links where they support a claim, and
  includes a clickable Sources section without inventing or rewriting URLs.
- Rich results: Organization, WebSite, Product, Article, Breadcrumb, FAQ, and
  HowTo JSON-LD inspection and generation.
- Images and performance: alt text analysis, image optimization, WebP support,
  Lighthouse integration, and mobile CSS checks.
- Article media: searches licensed providers for 3-4 topic-relevant images,
  stores attribution and license links, converts downloads to WebP, and proposes
  section-aware placement for editorial review. AI generation is off by default
  and is used only as an explicitly enabled fallback when search is insufficient.
- Generated-article media workflow: opens image search directly from an AI draft,
  lets the editor select licensed candidates, optionally fills shortages with
  configured AI image generation, inserts each image into the most relevant
  section, and returns the complete illustrated article in the same preview.
  Draft assets are retained with the article and included in later GitHub pushes.
- Controlled remediation: dry runs, approval classification, per-page fix
  limits, pre-change snapshots, validation, rollback, Git branches, and PRs.
- Evidence-bound planning: a deterministic SEO Planning Agent orders work by
  phase, records dependencies and expected metrics, and separates unattended
  low-risk fixes from changes requiring approval.
- Measurement: Google Search Console and Analytics integrations, scheduled
  scans, reports, and post-fix verification checkpoints.

## Safety Model

Site Inspector can change source files, so use a disposable clone or staging
repository first.

- `SI_WEB_ALLOW_REPO_WRITES` defaults to `false`.
- Only fixers classified as `fully_auto` can write during an unattended run.
- `semi_auto` and content-generating fixers require review.
- Modified HTML is validated before commit or push. Failed files are restored
  from their pre-fix snapshots and their issues return to `open`.
- The web app validates hosts and mutation origins. These controls are not a
  replacement for authentication when the service is exposed beyond localhost.
- Never commit `.env`, Google credentials, runtime databases, or `data/`.

## Requirements

- Python 3.12+
- Git
- Optional: GitHub CLI (`gh`) for PR creation
- Optional: Ollama for local generation
- Optional: Node.js and Lighthouse for performance audits
- Optional: Google credentials for Search Console and Analytics

## Install

The repository uses `uv.lock` for reproducible environments.

```bash
uv sync --locked --all-extras
```

Alternatively, install with pip for local development:

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Configure

Create a local environment file from the example and edit only local values:

```bash
copy .env.example .env
```

Configure targets in `config/targets.yaml`. For repository-backed fixes, point
the target source at a local disposable clone:

```yaml
targets:
  example:
    name: example
    base_url: https://example.com
    source:
      type: local
      local_path: data/site_sources/example
```

Keep repository writes disabled until a dry run and review have succeeded:

```dotenv
SI_WEB_ALLOW_REPO_WRITES=false
```

Configure one or more image search providers. Wikimedia Commons works without
an API key; provider keys generally improve relevance and coverage:

```dotenv
SI_UNSPLASH_API_KEY=
SI_PEXELS_API_KEY=
SI_PIXABAY_API_KEY=
SI_ARTICLE_IMAGE_COUNT=4
```

AI image fallback remains disabled unless both settings below are supplied.
Search is always attempted first:

```dotenv
SI_IMAGE_GENERATION_ENABLED=false
SI_OPENAI_API_KEY=
SI_IMAGE_GENERATION_MODEL=gpt-image-2
```

Article image changes use the `semi_auto` approval class and are only proposed
during a dry run. Review subject relevance, factual accuracy, crop, attribution,
and license terms before merging.

## Run

Initialize local storage and check optional dependencies:

```bash
uv run site-inspector init
```

Run a scan and inspect status:

```bash
uv run site-inspector scan run --target helinsilver
uv run site-inspector status --target helinsilver
```

Generate a versioned, read-only optimization plan from the latest scan:

```bash
uv run site-inspector plan generate
```

The plan is written to `data/reports/plans/` and includes evidence, ordering,
risk, confidence, approval requirements, validation checks, rollback
conditions, and deferred findings. When configured, the planner also uses
Search Console, Analytics, and completed verification outcomes to score search
opportunity and prior fix reliability. Missing integrations do not block a
plan; the report records which signals were unavailable.

Target-specific business policy can prioritize or protect URL groups:

```yaml
planning:
  goal: qualified_inquiries
  priority_url_patterns: ["/products/", "/contact/"]
  protected_url_patterns: ["/privacy/", "/terms/"]
```

Protected pages always require approval. Optional AI notes are bounded to
existing actions and require `SI_DEEPSEEK_API_KEY`:

```bash
uv run site-inspector plan generate --ai
```

Override the configured scoring goal for one plan with `--goal`, using one of
`organic_visibility`, `qualified_inquiries`, or `technical_health`.

Generate fixes as a dry run before allowing writes:

```bash
uv run site-inspector fix run --dry-run
```

Generate reports or run pending verification checks:

```bash
uv run site-inspector report generate --report-type daily
uv run site-inspector verify check
```

Start the local dashboard:

```bash
uv run site-inspector web start --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/`.

Open `http://127.0.0.1:8000/articles` to generate an article. For the automatic
workflow, enter a public website URL and select **网站研究并生成**. Topic and
keywords are optional: when omitted, the workflow derives them from the site.
Leave the article type at **AI 自动选择（推荐）** to let the editor choose among
blog, market analysis, product review, guide, news, and landing-page formats.
It validates public URLs and redirects, observes `robots.txt`, keeps only
reference-page structure and aggregate statistics, and does not copy reference
article bodies. Generated drafts and their research reports are saved under
`data/generated/` for review; repository publishing remains a separate,
write-gated action.

### Research-first article workflow

The website article workflow is split into two stages. **Start website and user
intent research** profiles the public site, derives keywords, discovers or simulates
natural-language queries, validates simulated queries against live search results,
and studies readable public reference pages. It reports heading structure, word and
content-image counts, paragraphs, tables, FAQ, lists, CTA patterns, and aggregate
benchmarks before any draft is written. The research screen produces an editable
brief with a headline, article type, target length, image range, outline, validated
questions, and differentiation opportunities. Only **Confirm brief and generate
article** invokes the writing model.

Free search results are coverage signals, not verified Google rankings, and query
volume remains unknown unless a real keyword-data provider is configured. AI-only
queries remain visibly labeled and are excluded from the validated brief when live
search produces no supporting result. Research plans are saved under
`data/generated/research-plans/`; confirmed drafts and reports are saved under
`data/generated/`.

## Quality Checks

Run the same checks used by CI:

```bash
uv run ruff check .
uv run python -m compileall -q config src tests
uv run python -m pytest -q --cov --cov-report=term
```

The initial whole-source coverage gate is 36 percent. Raise the threshold as the Git,
orchestration, source, AI-client, and web-route failure paths gain tests.

## Repository Layout

```text
config/             Global and per-target configuration
src/ai/             Model clients and prompts
src/core/           Scan, prioritization, fix orchestration, and validation
src/fixers/         Proposed and automatic remediation implementations
src/git/            Branch, commit, push, PR, and rollback workflow
src/inspectors/     SEO, content, accessibility, mobile, and performance checks
src/integrations/   Google, Lighthouse, image, and sitemap integrations
src/web/            FastAPI dashboard and mutation security
tests/              Unit and safety regression tests
```

## Production Checklist

Before exposing Site Inspector or enabling writes:

1. Put the service behind authentication and TLS.
2. Store secrets in the deployment platform's secret manager.
3. Use a dedicated bot identity with least-privilege repository access.
4. Require CI and human review before merging generated pull requests.
5. Back up the database and target repository.
6. Verify changes in staging and monitor Search Console at days 3, 7, and 14.

## License

No license has been selected yet. Until one is added, normal copyright rules
apply and redistribution rights are not granted.

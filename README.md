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
- Rich results: Organization, WebSite, Product, Article, Breadcrumb, FAQ, and
  HowTo JSON-LD inspection and generation.
- Images and performance: alt text analysis, image optimization, WebP support,
  Lighthouse integration, and mobile CSS checks.
- Controlled remediation: dry runs, approval classification, per-page fix
  limits, pre-change snapshots, validation, rollback, Git branches, and PRs.
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

## Quality Checks

Run the same checks used by CI:

```bash
uv run ruff check .
uv run python -m compileall -q config src tests
uv run python -m pytest -q --cov=src --cov-report=term
```

The initial coverage gate is 45 percent. Raise the threshold as the Git,
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

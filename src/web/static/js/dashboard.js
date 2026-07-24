let scoresChart = null;
let trendChart = null;

async function loadSummary() {
    const resp = await fetch('/api/dashboard/summary');
    const data = await resp.json();
    document.getElementById('stat-scans').textContent = data.total_scans;
    document.getElementById('stat-issues').textContent = data.total_issues;
    document.getElementById('stat-fixes').textContent = data.total_fixes;
    document.getElementById('stat-p0').textContent = data.p0_count;

    if (data.latest_scan) {
        document.getElementById('latest-scan-info').innerHTML = `
            <div class="latest-scan-grid">
                <div><strong>Scan #${data.latest_scan.id}</strong></div>
                <div>${new Date(data.latest_scan.date).toLocaleString()}</div>
                <div><span class="status-badge status-${data.latest_scan.status}">${data.latest_scan.status}</span></div>
            </div>
            <p style="margin-top:8px">Pages: ${data.latest_scan.pages_crawled || 0} &nbsp;|&nbsp; Issues: ${data.latest_scan.total_issues_found || 0}</p>
        `;
    } else {
        document.getElementById('latest-scan-info').innerHTML = '<p class="empty">No scans completed yet.</p>';
    }
}

async function loadScores() {
    const resp = await fetch('/api/dashboard/scores');
    const scores = await resp.json();
    const labels = scores.map(s => s.label);
    const values = scores.map(s => s.score);
    const bgColors = scores.map(s => {
        if (s.status === 'good') return '#27ae60';
        if (s.status === 'warn') return '#f39c12';
        return '#e74c3c';
    });

    if (scoresChart) scoresChart.destroy();
    const ctx = document.getElementById('scores-chart').getContext('2d');
    scoresChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Health Score',
                data: values,
                backgroundColor: bgColors,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

async function loadTrend() {
    const resp = await fetch('/api/dashboard/trend');
    const trend = await resp.json();
    const dates = trend.map(t => t.date);
    const issues = trend.map(t => t.issues);
    const fixes = trend.map(t => t.fixes);

    if (trendChart) trendChart.destroy();
    const ctx = document.getElementById('trend-chart').getContext('2d');
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Issues Found',
                    data: issues,
                    borderColor: '#e8590c',
                    backgroundColor: 'rgba(232,89,12,0.1)',
                    fill: true,
                    tension: 0.3,
                },
                {
                    label: 'Fixes Applied',
                    data: fixes,
                    borderColor: '#0f3460',
                    backgroundColor: 'rgba(15,52,96,0.1)',
                    fill: true,
                    tension: 0.3,
                }
            ]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
        }
    });
}

// ── Quick Scan ────────────────────────────────────────────────────────
const quickScanForm = document.getElementById('quick-scan-form');
const quickScanBtn = document.getElementById('quick-scan-btn');
const quickScanUrl = document.getElementById('quick-scan-url');
const progressEl = document.getElementById('quick-scan-progress');
const resultEl = document.getElementById('quick-scan-result');

let pollTimer = null;

quickScanForm.addEventListener('submit', async () => {
    const url = quickScanUrl.value.trim();
    if (!url) return;

    const repoUrl = document.getElementById('quick-scan-repo').value.trim();
    const branch = document.getElementById('quick-scan-branch').value.trim() || 'main';
    const pushChanges = document.getElementById('quick-scan-push').checked;

    quickScanBtn.disabled = true;
    quickScanBtn.textContent = 'Starting...';
    progressEl.style.display = 'block';
    progressEl.innerHTML = '<span class="spinner"></span> Starting scan...';
    resultEl.style.display = 'none';
    if (pollTimer) clearInterval(pollTimer);

    try {
        const resp = await fetch('/api/quick-scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url, repo_url: repoUrl || null, repo_branch: branch, push_changes: pushChanges }),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed (${resp.status})`);
        }

        const data = await resp.json();
        startPolling(data.scan_id);

    } catch (err) {
        progressEl.innerHTML = `<div class="quick-scan-error">Error: ${escapeHtml(err.message)}</div>`;
        resetButton();
    }
});

function startPolling(scanId) {
    let pollCount = 0;
    const MAX_POLLS = 300;

    pollTimer = setInterval(async () => {
        pollCount++;
        if (pollCount > MAX_POLLS) {
            clearInterval(pollTimer);
            progressEl.innerHTML = '<div class="quick-scan-error">Scan timed out. Please try again.</div>';
            resetButton();
            return;
        }

        try {
            const resp = await fetch(`/api/scans/${scanId}`);
            const scan = await resp.json();

            if (scan.error) {
                clearInterval(pollTimer);
                progressEl.innerHTML = `<div class="quick-scan-error">Error: ${escapeHtml(scan.error)}</div>`;
                resetButton();
                return;
            }

            const label = getPhaseLabel(scan.phase);
            progressEl.innerHTML = `<span class="spinner"></span> <span class="phase">${label}</span>` +
                (scan.pages_crawled > 0
                    ? ` <span class="detail">(${scan.pages_crawled} pages, ${scan.total_issues_found || 0} issues)</span>`
                    : '');

            if (scan.status === 'completed') {
                clearInterval(pollTimer);
                showResult(scan);
                resetButton();
            } else if (scan.status === 'failed') {
                clearInterval(pollTimer);
                progressEl.innerHTML = `<div class="quick-scan-error">Scan failed: ${escapeHtml(scan.error_message || 'Unknown error')}</div>`;
                resetButton();
            }
        } catch (err) {
            console.warn('Poll error:', err);
        }
    }, 2000);
}

function getPhaseLabel(phase) {
    const labels = {
        'starting': 'Initializing...',
        'crawling': 'Crawling pages...',
        'inspecting': 'Inspecting pages...',
        'analyzing': 'Analyzing issues...',
        'fixing': 'Generating fix suggestions...',
        'reporting': 'Generating report...',
        'done': 'Done!',
    };
    return labels[phase] || phase || 'Running...';
}

let lastScanId = null;

function showResult(scan) {
    lastScanId = scan.id;
    progressEl.style.display = 'none';
    resultEl.style.display = 'block';

    let fixHtml = '';
    if (scan.pr_url) {
        fixHtml = `<div style="margin-top:8px;color:var(--good);font-size:.9rem;">
            &#9989; PR created: <a href="${escapeHtml(scan.pr_url)}" target="_blank" rel="noopener">${escapeHtml(scan.pr_url)}</a>
        </div>`;
    } else if (scan.fix_error) {
        fixHtml = `<div style="margin-top:8px;color:var(--bad);font-size:.9rem;">
            &#9888; Fix failed: ${escapeHtml(scan.fix_error)}
        </div>`;
    } else {
        fixHtml = `<div style="margin-top:12px;">
            <a class="btn btn-primary" href="/fixes?scan_id=${scan.id}">Review Fix Suggestions</a>
        </div>`;
    }

    resultEl.innerHTML = `<h3>Scan Complete</h3>
        <div class="stats">
            <div class="stat-item"><strong>${scan.pages_crawled}</strong> pages crawled</div>
            <div class="stat-item"><strong>${scan.total_issues_found}</strong> issues found</div>
            <div class="stat-item"><a href="/issues?scan_id=${scan.id}">View Issues &rarr;</a></div>
            <div class="stat-item"><a href="/scans">View All Scans &rarr;</a></div>
        </div>
        ${fixHtml}`;
    loadSummary();
    loadScores();
    loadTrend();
}

async function applyFixes(scanId) {
    const statusEl = document.getElementById('fix-status');
    if (!statusEl) return;
    statusEl.textContent = 'Applying fixes...';
    try {
        const resp = await fetch(`/api/scans/${scanId}/apply-fixes`, { method: 'POST' });
        const data = await resp.json();
        if (data.error) {
            statusEl.innerHTML = `<span style="color:var(--bad)">${escapeHtml(data.error)}</span>`;
        } else if (data.files_written > 0) {
            statusEl.innerHTML = `<span style="color:var(--good)">Done: ${data.files_written} files written to <code>${escapeHtml(data.output_dir)}</code></span>`;
        } else {
            statusEl.innerHTML = `<span style="color:var(--text-muted)">${escapeHtml(data.message || 'No fixes to apply')}</span>`;
        }
    } catch (e) {
        statusEl.innerHTML = `<span style="color:var(--bad)">Failed: ${escapeHtml(e.message)}</span>`;
    }
}

function resetButton() {
    quickScanBtn.disabled = false;
    quickScanBtn.textContent = 'Quick Scan';
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Email Settings ──────────────────────────────────────────────────
async function loadEmailSettings() {
    try {
        const resp = await fetch('/api/settings/email');
        const data = await resp.json();
        document.getElementById('email-recipients').value = data.recipients || '';
        document.getElementById('email-smtp-host').value = data.smtp_host || '';
        document.getElementById('email-smtp-port').value = data.smtp_port || 587;
        document.getElementById('email-username').value = data.smtp_username || '';
    } catch (e) { console.warn('Failed to load email settings:', e); }
}

document.getElementById('email-form').addEventListener('submit', async () => {
    const status = document.getElementById('email-status');
    status.textContent = '保存中...';
    try {
        const resp = await fetch('/api/settings/email', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                recipients: document.getElementById('email-recipients').value,
                smtp_host: document.getElementById('email-smtp-host').value,
                smtp_port: parseInt(document.getElementById('email-smtp-port').value) || 587,
                smtp_username: document.getElementById('email-username').value,
                smtp_password: document.getElementById('email-password').value,
            }),
        });
        const data = await resp.json();
        status.innerHTML = '<span style=\"color:var(--good)\">' + data.message + '</span>';
    } catch (e) {
        status.innerHTML = '<span style=\"color:var(--bad)\">保存失败: ' + e.message + '</span>';
    }
});

loadSummary();
loadScores();
loadTrend();
loadEmailSettings();

let scoresChart = null;
let trendChart = null;

// ── Summary ──────────────────────────────────────────────────────

async function loadSummary() {
    const resp = await fetch('/api/dashboard/summary');
    const data = await resp.json();
    document.getElementById('stat-scans').textContent = data.total_scans;
    document.getElementById('stat-issues').textContent = data.total_issues;
    document.getElementById('stat-fixes').textContent = data.total_fixes;
    document.getElementById('stat-p0').textContent = data.p0_count;

    if (data.latest_scan) {
        const s = data.latest_scan;
        document.getElementById('latest-scan-info').innerHTML =
            `<p><strong>#${s.id}</strong> &mdash; ${new Date(s.date).toLocaleString()}</p>
             <p><span class="status-badge status-${s.status}">${s.status}</span> &nbsp; Pages: ${s.pages_crawled || 0} &nbsp; Issues: ${s.total_issues_found || 0}</p>`;
    } else {
        document.getElementById('latest-scan-info').innerHTML = '<p class="empty">No scans yet.</p>';
    }
}

// ── Charts ───────────────────────────────────────────────────────

async function loadScores() {
    const resp = await fetch('/api/dashboard/scores');
    const scores = await resp.json();
    const labels = scores.map(s => s.label);
    const values = scores.map(s => s.score);
    const bg = scores.map(s => s.status === 'good' ? '#27ae60' : s.status === 'warn' ? '#f39c12' : '#e74c3c');
    if (scoresChart) scoresChart.destroy();
    scoresChart = new Chart(document.getElementById('scores-chart'), {
        type: 'bar', data: { labels, datasets: [{ label: 'Score', data: values, backgroundColor: bg, borderRadius: 4 }] },
        options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } }, plugins: { legend: { display: false } } }
    });
}

async function loadTrend() {
    const resp = await fetch('/api/dashboard/trend');
    const trend = await resp.json();
    if (trendChart) trendChart.destroy();
    trendChart = new Chart(document.getElementById('trend-chart'), {
        type: 'line',
        data: {
            labels: trend.map(t => t.date),
            datasets: [
                { label: 'Issues', data: trend.map(t => t.issues), borderColor: '#e8590c', backgroundColor: 'rgba(232,89,12,0.1)', fill: true, tension: 0.3 },
                { label: 'Fixes', data: trend.map(t => t.fixes), borderColor: '#0f3460', backgroundColor: 'rgba(15,52,96,0.1)', fill: true, tension: 0.3 }
            ]
        },
        options: { responsive: true, scales: { y: { beginAtZero: true } } }
    });
}

// ── Plan Board ────────────────────────────────────────────────────

async function loadPlan() {
    try {
        const resp = await fetch('/api/dashboard/plan');
        const data = await resp.json();
        if (!data.plan) return;
        const p = data.plan;
        const card = document.getElementById('plan-card');
        card.style.display = '';

        // Plan meta
        const autoCount = p.actions.filter(a => !a.approval_required).length;
        const approvalCount = p.actions.filter(a => a.approval_required).length;
        const totalIssues = p.actions.reduce((s, a) => s + a.issue_ids.length, 0);
        const deferred = (p.deferred || []).length;
        document.getElementById('plan-meta').textContent = `scan #${p.scan_id} | ${p.actions.length} actions | ${totalIssues} issues`;
        document.getElementById('plan-summary').innerHTML =
            `<div class="ps auto"><div class="ps-val">${autoCount}</div><div class="ps-lbl">Auto-Execute</div></div>
             <div class="ps approval"><div class="ps-val">${approvalCount}</div><div class="ps-lbl">Needs Approval</div></div>
             <div class="ps defer"><div class="ps-val">${deferred}</div><div class="ps-lbl">Deferred</div></div>
             <div class="ps"><div class="ps-val">${p.warnings ? p.warnings.length : 0}</div><div class="ps-lbl">Warnings</div></div>`;
        document.getElementById('plan-exec-summary').textContent = p.executive_summary || '';
        if (p.ai_strategy_note) {
            document.getElementById('plan-exec-summary').textContent += ` AI advisory: ${p.ai_strategy_note}`;
        }
        document.getElementById('plan-exec-summary').style.display = p.executive_summary ? '' : 'none';

        // Phase groups
        const phaseLabel = {1:'Crawl & Index',2:'Architecture',3:'Snippets',4:'Rich Results',5:'Authority'};
        const phases = {};
        p.actions.forEach(a => { const ph = a.phase || 1; if (!phases[ph]) phases[ph] = []; phases[ph].push(a); });

        document.getElementById('plan-board').innerHTML = Object.entries(phases).map(([ph, actions]) =>
            `<div class="plan-phase">
                <div class="plan-phase-header ph${ph}">Phase ${ph}: ${phaseLabel[ph]} <span style="float:right">${actions.length}</span></div>
                ${actions.map(a => `
                    <div class="plan-action">
                        <div class="pa-title">${escapeHtml(a.action_id)}. ${escapeHtml(a.title)}</div>
                        <div class="pa-badges">
                            <span class="badge badge-${a.execution_mode === 'fully_auto' ? 'auto' : a.execution_mode === 'semi_auto' ? 'semi' : 'manual'}">${a.execution_mode}</span>
                            <span class="badge badge-${a.risk}">${a.risk} risk</span>
                            <span class="badge badge-${a.decision === 'execute_automatically' ? 'auto' : 'semi'}">${escapeHtml(a.decision || 'review')}</span>
                            ${a.opportunity_score > 0.7 ? '<span class="badge badge-auto">high opp</span>' : ''}
                        </div>
                        <div class="pa-rationale"><strong>Problem:</strong> ${escapeHtml(a.problem_statement || a.rationale)}</div>
                        <div class="pa-rationale"><strong>Solution:</strong> ${escapeHtml(a.proposed_solution || '')}</div>
                    </div>`).join('')}
            </div>`
        ).join('');

    } catch (e) { console.warn('Plan load failed:', e); }
}

// ── Competitor Board ──────────────────────────────────────────────

async function loadCompetitors() {
    try {
        const resp = await fetch('/api/scans');
        const scans = await resp.json();
        const latest = scans.find(s => ['completed', 'degraded'].includes(s.status));
        if (!latest) return;
        const issuesResp = await fetch(`/api/issues?inspector=competitor_gap&scan_id=${latest.id}`);
        const data = await issuesResp.json();
        if (!data.items || !data.items.length) return;
        const card = document.getElementById('competitor-card');
        card.style.display = '';

        const changes = data.items.filter(i => i.category === 'competitor_page_changed');
        const gaps = data.items.filter(i => i.category !== 'competitor_page_changed');

        document.getElementById('competitor-board').innerHTML =
            (changes.length ? `<div style="margin-bottom:8px;font-weight:600;color:var(--warn);">${changes.length} competitor(s) changed since last check:</div>` +
                changes.map(c => `<div class="comp-item changed"><span class="comp-domain">${escapeHtml(c.description.split(':')[0])}</span><span class="comp-status" style="color:var(--warn);">changed</span></div>`).join('') : '') +
            `<div style="font-size:.85rem;color:var(--text-muted);margin-top:${changes.length?8:0}px;">${gaps.length} competitive gaps found in latest scan</div>`;
    } catch (e) { console.warn('Competitor load failed:', e); }
}

// ── Quick Scan ────────────────────────────────────────────────────

let pollTimer = null;
document.getElementById('quick-scan-form').addEventListener('submit', async () => {
    const url = document.getElementById('quick-scan-url').value.trim();
    if (!url) return;
    const btn = document.getElementById('quick-scan-btn');
    btn.disabled = true; btn.textContent = '...';
    const progress = document.getElementById('quick-scan-progress');
    const result = document.getElementById('quick-scan-result');
    progress.style.display = ''; progress.innerHTML = 'Starting scan...';
    result.style.display = 'none';
    if (pollTimer) clearInterval(pollTimer);

    try {
        const resp = await fetch('/api/quick-scan', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url,
                repo_url: document.getElementById('quick-scan-repo').value.trim() || null,
                repo_branch: document.getElementById('quick-scan-branch').value.trim() || 'main',
                push_changes: document.getElementById('quick-scan-push').checked
            })
        });
        if (!resp.ok) { const e = await resp.json().catch(()=>({})); throw new Error(e.detail || 'Failed'); }
        const d = await resp.json();
        pollScan(d.scan_id, progress, result);
    } catch (e) {
        progress.innerHTML = `<div class="qs-error">${escapeHtml(e.message)}</div>`;
        btn.disabled = false; btn.textContent = 'Go';
    }
});

function pollScan(scanId, progress, result) {
    let n = 0;
    const labels = {starting:'Init',crawling:'Crawling',inspecting:'Inspecting',analyzing:'Analyzing',fixing:'Fixing',reporting:'Reporting',done:'Done'};
    pollTimer = setInterval(async () => {
        if (++n > 300) { clearInterval(pollTimer); progress.innerHTML = '<div class="qs-error">Timed out</div>'; return; }
        try {
            const r = await fetch(`/api/scans/${scanId}`);
            const s = await r.json();
            const phase = labels[s.phase] || s.phase;
            progress.innerHTML = `${phase}${s.pages_crawled > 0 ? ` (${s.pages_crawled} pages, ${s.total_issues_found || 0} issues)` : ''}`;
            if (['completed', 'degraded'].includes(s.status)) {
                clearInterval(pollTimer);
                progress.style.display = 'none';
                result.style.display = '';
                const healthLabel = s.status === 'degraded'
                    ? 'Scan Complete with Inspector Errors'
                    : 'Scan Complete';
                result.innerHTML = `<div class="qs-done"><strong>${healthLabel}</strong>
                    <div class="qs-stats"><span>${s.pages_crawled} pages</span><span>${s.total_issues_found} issues</span><span><a href="/issues?scan_id=${s.id}">View Issues</a></span></div>
                    ${s.pr_url ? `<div style="margin-top:6px;"><a href="${escapeHtml(s.pr_url)}" target="_blank">View PR</a></div>` : ''}</div>`;
                document.getElementById('quick-scan-btn').disabled = false;
                document.getElementById('quick-scan-btn').textContent = 'Go';
                loadSummary(); loadScores(); loadTrend(); loadPlan(); loadCompetitors();
            } else if (s.status === 'failed') {
                clearInterval(pollTimer);
                progress.innerHTML = `<div class="qs-error">Failed: ${escapeHtml(s.error_message || 'Unknown')}</div>`;
                document.getElementById('quick-scan-btn').disabled = false;
                document.getElementById('quick-scan-btn').textContent = 'Go';
            }
        } catch(e) { console.warn('Poll error:', e); }
    }, 2000);
}

// ── Email ─────────────────────────────────────────────────────────

async function loadEmail() {
    const r = await fetch('/api/settings/email'); const d = await r.json();
    document.getElementById('email-recipients').value = d.recipients || '';
    document.getElementById('email-smtp-host').value = d.smtp_host || '';
    document.getElementById('email-smtp-port').value = d.smtp_port || 587;
    document.getElementById('email-username').value = d.smtp_username || '';
}
document.getElementById('email-form').addEventListener('submit', async () => {
    const s = document.getElementById('email-status');
    s.textContent = 'Saving...';
    const r = await fetch('/api/settings/email', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
            recipients: document.getElementById('email-recipients').value,
            smtp_host: document.getElementById('email-smtp-host').value,
            smtp_port: parseInt(document.getElementById('email-smtp-port').value) || 587,
            smtp_username: document.getElementById('email-username').value,
            smtp_password: document.getElementById('email-password').value
        })
    });
    const d = await r.json();
    s.innerHTML = '<span style="color:var(--good)">' + d.message + '</span>';
});

// ── Helpers ───────────────────────────────────────────────────────

function escapeHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ── Init ──────────────────────────────────────────────────────────

loadSummary(); loadScores(); loadTrend(); loadPlan(); loadCompetitors(); loadEmail();

let currentPage = 1;
const limit = 50;

function getFilters() {
    return {
        inspector: document.getElementById('filter-inspector').value,
        priority: document.getElementById('filter-priority').value,
        status: document.getElementById('filter-status').value,
    };
}

async function loadIssues(page) {
    currentPage = page || currentPage;
    const f = getFilters();
    const params = new URLSearchParams({ page: currentPage, limit });
    if (f.inspector) params.set('inspector', f.inspector);
    if (f.priority) params.set('priority', f.priority);
    if (f.status) params.set('status', f.status);

    const resp = await fetch('/api/issues?' + params);
    const data = await resp.json();

    const tbody = document.getElementById('issues-tbody');
    if (!data.items.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty">No issues found.</td></tr>';
    } else {
        tbody.innerHTML = data.items.map(i => `
            <tr>
                <td><a href="/issues/${i.id}">${escapeHtml(i.url)}</a></td>
                <td>${i.inspector}</td>
                <td>${i.category}</td>
                <td><span class="badge badge-${i.priority_tier.toLowerCase()}">${i.priority_tier}</span></td>
                <td>${i.priority_score}</td>
                <td><span class="status-badge status-${i.status}">${i.status}</span></td>
                <td>${i.fix_count}</td>
            </tr>
        `).join('');
    }

    renderPagination(data.page, data.pages);
}

function renderPagination(page, pages) {
    const el = document.getElementById('issues-pagination');
    if (pages <= 1) { el.innerHTML = ''; return; }
    const btns = [];
    btns.push(`<button class="page-btn" onclick="loadIssues(${page - 1})" ${page <= 1 ? 'disabled' : ''}>&laquo; Prev</button>`);
    for (let p = 1; p <= pages; p++) {
        btns.push(`<button class="page-btn ${p === page ? 'active' : ''}" onclick="loadIssues(${p})">${p}</button>`);
    }
    btns.push(`<button class="page-btn" onclick="loadIssues(${page + 1})" ${page >= pages ? 'disabled' : ''}>Next &raquo;</button>`);
    el.innerHTML = btns.join('');
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

document.getElementById('filter-apply').addEventListener('click', () => loadIssues(1));
loadIssues(1);

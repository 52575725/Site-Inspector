let currentPage = 1;
const limit = 50;

function getFilters() {
    return {
        fix_type: document.getElementById('filter-fix-type').value,
    };
}

async function loadFixes(page) {
    currentPage = page || currentPage;
    const f = getFilters();
    const params = new URLSearchParams({ page: currentPage, limit });
    if (f.fix_type) params.set('fix_type', f.fix_type);

    const resp = await fetch('/api/fixes?' + params);
    const data = await resp.json();

    const tbody = document.getElementById('fixes-tbody');
    if (!data.items.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty">No fixes found.</td></tr>';
    } else {
        tbody.innerHTML = data.items.map(f => `
            <tr>
                <td>
                    ${f.issue_url ? `<a href="/issues/${f.id}">${f.issue_category}</a>` : '—'}
                </td>
                <td>${f.fixer}</td>
                <td><span class="badge badge-${f.fix_type}">${f.fix_type}</span></td>
                <td class="mono">${f.file_path || '—'}</td>
                <td>${f.applied_at ? new Date(f.applied_at).toLocaleDateString() : '—'}</td>
                <td>${f.git_pr_url ? `<a href="${f.git_pr_url}" target="_blank">PR &#8599;</a>` : '—'}</td>
                <td>
                    <span class="status-badge status-${f.verification_status}">${f.verification_status}</span>
                    ${f.verification_count ? ` (${f.verification_count})` : ''}
                </td>
            </tr>
        `).join('');
    }

    renderPagination(data.page, data.pages);
}

function renderPagination(page, pages) {
    const el = document.getElementById('fixes-pagination');
    if (pages <= 1) { el.innerHTML = ''; return; }
    const btns = [];
    btns.push(`<button class="page-btn" onclick="loadFixes(${page - 1})" ${page <= 1 ? 'disabled' : ''}>&laquo; Prev</button>`);
    for (let p = 1; p <= pages; p++) {
        btns.push(`<button class="page-btn ${p === page ? 'active' : ''}" onclick="loadFixes(${p})">${p}</button>`);
    }
    btns.push(`<button class="page-btn" onclick="loadFixes(${page + 1})" ${page >= pages ? 'disabled' : ''}>Next &raquo;</button>`);
    el.innerHTML = btns.join('');
}

document.getElementById('filter-apply').addEventListener('click', () => loadFixes(1));
loadFixes(1);

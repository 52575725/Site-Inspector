const actions = document.getElementById('fix-actions');
const fixId = Number(actions.dataset.fixId);

function renderActions(status) {
    actions.dataset.status = status;
    if (status === 'proposed') {
        actions.innerHTML = '<button class="btn btn-primary" data-action="approve">批准修复</button><button class="btn" data-action="reject">拒绝</button>';
    } else if (status === 'approved') {
        actions.innerHTML = '<button class="btn btn-primary" data-action="apply">应用修复</button><button class="btn" data-action="reject">拒绝</button>';
    } else {
        actions.innerHTML = '<a class="btn" href="/fixes">返回修复列表</a>';
    }
}

actions.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-action]');
    if (!button) return;
    const action = button.dataset.action;
    if (action === 'apply' && !window.confirm('确认应用这条修复吗？')) return;
    button.disabled = true;
    const response = await fetch(`/api/fixes/${fixId}/${action}`, { method: 'POST' });
    const data = await response.json();
    if (!response.ok) { window.alert(data.detail || '操作失败'); button.disabled = false; return; }
    const status = document.getElementById('fix-status');
    status.className = `status-badge status-${data.status}`;
    status.textContent = data.status_label;
    renderActions(data.status);
});

renderActions(actions.dataset.status);

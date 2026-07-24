import sqlite3
conn = sqlite3.connect(r'data\site_inspector.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

scan = cur.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
if scan:
    print(f"=== Latest Scan ===")
    for k in scan.keys():
        print(f"  {k}: {scan[k]}")

    print(f"\n=== Issues for scan #{scan['id']} ===")
    issues = cur.execute(
        "SELECT id,url,category,severity,priority_score,priority_tier,title "
        "FROM issues WHERE scan_id=? ORDER BY priority_score DESC",
        (scan['id'],)
    ).fetchall()
    for i in issues:
        print(f"  #{i['id']} [{i['priority_tier']}] {i['category']} | "
              f"score={i['priority_score']:.2f} | {i['url']}")
        if i['title']:
            print(f"       title: {i['title'][:120]}")

    print(f"\n=== Fixes for scan #{scan['id']} ===")
    fixes = cur.execute("SELECT * FROM fixes WHERE scan_id=?", (scan['id'],)).fetchall()
    print(f"  Total: {len(fixes)}")
    for f in fixes:
        print(f"  #{f['id']} fixer={f['fixer']} type={f['fix_type']} file={f['file_path']}")
else:
    print("No scans found!")

print(f"\n=== Targets ===")
for t in cur.execute("SELECT * FROM targets"):
    print(f"  #{t['id']} name={t['name']} url={t['base_url']}")

print(f"\n=== Recent Scans ===")
for s in cur.execute("SELECT id, status, phase, fix_error, pr_url, pages_crawled, total_issues_found FROM scans ORDER BY id DESC LIMIT 5"):
    print(f"  #{s['id']} status={s['status']} phase={s['phase']} pages={s['pages_crawled']} issues={s['total_issues_found']}")
    if s['fix_error']:
        print(f"       fix_error: {s['fix_error'][:200]}")
    if s['pr_url']:
        print(f"       pr_url: {s['pr_url']}")

cur.close()
conn.close()

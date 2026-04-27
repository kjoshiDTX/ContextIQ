import { useState, useEffect } from 'react';
import { createCategory, listCategories, deleteCategory } from '../../api';
import { Card, CardTitle, Spinner } from '../ui/Card';
import { Button } from '../ui/Button';

export function CategoriesPanel({ userId, toast }) {
  const [categories, setCategories] = useState([]);
  const [name, setName] = useState('');
  const [color, setColor] = useState('#6366f1');
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => { load(); }, [userId]);

  const load = async () => {
    setLoading(true);
    try {
      const d = await listCategories();
      setCategories(d.categories || []);
    } catch (e) { toast(e.message, 'error'); }
    finally { setLoading(false); }
  };

  const add = async () => {
    if (!name.trim()) { toast('Name required', 'error'); return; }
    setSaving(true);
    try {
      await createCategory(name.trim(), color);
      setName('');
      toast('Category created', 'success');
      load();
    } catch (e) { toast(e.message, 'error'); }
    finally { setSaving(false); }
  };

  const remove = async (id) => {
    try {
      await deleteCategory(id);
      setCategories(cs => cs.filter(c => c.id !== id));
      toast('Category deleted', 'success');
    } catch (e) { toast(e.message, 'error'); }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 600 }}>
      <Card>
        <CardTitle>New Category</CardTitle>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: 11, color: 'var(--muted)', display: 'block', marginBottom: 5 }}>Name</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)}
              placeholder="e.g. Research, Work, Personal"
              onKeyDown={e => e.key === 'Enter' && add()} />
          </div>
          <div>
            <label style={{ fontSize: 11, color: 'var(--muted)', display: 'block', marginBottom: 5 }}>Color</label>
            <input type="color" value={color} onChange={e => setColor(e.target.value)} style={{ width: 60 }} />
          </div>
          <Button onClick={add} disabled={saving || !name.trim()}>
            {saving ? <Spinner /> : 'Add'}
          </Button>
        </div>
      </Card>

      <Card>
        <CardTitle>
          Categories
          <Button variant="ghost" size="sm" onClick={load}>↻ Refresh</Button>
        </CardTitle>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 32 }}><Spinner /></div>
        ) : categories.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--muted)', padding: 32 }}>
            No categories yet. Create one above to organize your documents.
          </div>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {categories.map(c => (
              <div key={c.id} style={{
                display: 'inline-flex', alignItems: 'center', gap: 8,
                padding: '8px 14px', borderRadius: 20,
                background: 'var(--surface)', border: '1px solid var(--border)',
              }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: c.color, display: 'inline-block', flexShrink: 0 }} />
                <span style={{ fontSize: 13 }}>{c.name}</span>
                <button onClick={() => remove(c.id)} style={{
                  background: 'none', border: 'none', color: 'var(--muted)',
                  cursor: 'pointer', fontSize: 14, lineHeight: 1, padding: '0 2px',
                }}>×</button>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

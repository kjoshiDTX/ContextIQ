import { useState, useEffect } from 'react';
import { createTask, listTasks, updateTask, deleteTask } from '../../api';
import { Card, CardTitle, Badge, Spinner } from '../ui/Card';
import { Button } from '../ui/Button';

function fmtDate(s) {
  if (!s) return null;
  return new Date(s).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

export function TasksPanel({ userId, toast }) {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [title, setTitle] = useState('');
  const [desc, setDesc] = useState('');
  const [status, setStatus] = useState('todo');
  const [due, setDue] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => { load(); }, [userId]);

  const load = async () => {
    setLoading(true);
    try {
      const d = await listTasks();
      setTasks(d.tasks || []);
    } catch (e) { toast(e.message, 'error'); }
    finally { setLoading(false); }
  };

  const add = async () => {
    if (!title.trim()) { toast('Title required', 'error'); return; }
    setSaving(true);
    try {
      await createTask({ title: title.trim(), description: desc.trim() || undefined, status, due_date: due || undefined });
      setTitle(''); setDesc(''); setStatus('todo'); setDue('');
      toast('Task created', 'success');
      load();
    } catch (e) { toast(e.message, 'error'); }
    finally { setSaving(false); }
  };

  const changeStatus = async (id, newStatus) => {
    try {
      await updateTask(id, { status: newStatus });
      setTasks(ts => ts.map(t => t.id === id ? { ...t, status: newStatus } : t));
    } catch (e) { toast(e.message, 'error'); }
  };

  const remove = async (id) => {
    try {
      await deleteTask(id);
      setTasks(ts => ts.filter(t => t.id !== id));
      toast('Task deleted', 'success');
    } catch (e) { toast(e.message, 'error'); }
  };

  const byStatus = (s) => tasks.filter(t => t.status === s);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Add task */}
      <Card>
        <CardTitle>New Task</CardTitle>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <input type="text" value={title} onChange={e => setTitle(e.target.value)}
            placeholder="Task title" style={{ gridColumn: '1 / -1' }}
            onKeyDown={e => e.key === 'Enter' && add()} />
          <input type="text" value={desc} onChange={e => setDesc(e.target.value)}
            placeholder="Description (optional)" style={{ gridColumn: '1 / -1' }} />
          <div>
            <label style={{ fontSize: 11, color: 'var(--muted)', display: 'block', marginBottom: 5 }}>Status</label>
            <select value={status} onChange={e => setStatus(e.target.value)}>
              <option value="todo">To Do</option>
              <option value="in_progress">In Progress</option>
              <option value="done">Done</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 11, color: 'var(--muted)', display: 'block', marginBottom: 5 }}>Due Date</label>
            <input type="date" value={due} onChange={e => setDue(e.target.value)} />
          </div>
          <Button onClick={add} disabled={saving || !title.trim()} style={{ gridColumn: '1 / -1', justifyContent: 'center' }}>
            {saving ? <Spinner /> : 'Add Task'}
          </Button>
        </div>
      </Card>

      {/* Task columns */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: 32 }}><Spinner /></div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
          {[
            { key: 'todo',        label: 'To Do',       color: '#6b7280' },
            { key: 'in_progress', label: 'In Progress',  color: '#f59e0b' },
            { key: 'done',        label: 'Done',         color: '#22c55e' },
          ].map(col => (
            <Card key={col.key}>
              <CardTitle>
                <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: col.color, display: 'inline-block' }} />
                  {col.label}
                  <span style={{ marginLeft: 4, fontSize: 11, color: 'var(--muted)', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
                    ({byStatus(col.key).length})
                  </span>
                </span>
                <Button variant="ghost" size="sm" onClick={load}>↻</Button>
              </CardTitle>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {byStatus(col.key).length === 0 && (
                  <div style={{ color: 'var(--muted)', fontSize: 12, textAlign: 'center', padding: '12px 0' }}>Empty</div>
                )}
                {byStatus(col.key).map(t => (
                  <div key={t.id} style={{ background: 'var(--surface)', borderRadius: 8, padding: 12, border: '1px solid var(--border)' }}>
                    <div style={{ fontWeight: 500, fontSize: 13, marginBottom: 4 }}>{t.title}</div>
                    {t.description && <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 6 }}>{t.description}</div>}
                    {t.due_date && <div style={{ fontSize: 11, color: 'var(--warn)', marginBottom: 8 }}>📅 {fmtDate(t.due_date)}</div>}
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {col.key !== 'todo' && (
                        <Button variant="ghost" size="sm" onClick={() => changeStatus(t.id, 'todo')}>← To Do</Button>
                      )}
                      {col.key !== 'in_progress' && (
                        <Button variant="ghost" size="sm" onClick={() => changeStatus(t.id, 'in_progress')}>
                          {col.key === 'todo' ? '→ Start' : '← Restart'}
                        </Button>
                      )}
                      {col.key !== 'done' && (
                        <Button variant="success" size="sm" onClick={() => changeStatus(t.id, 'done')}>✓ Done</Button>
                      )}
                      <Button variant="danger" size="sm" onClick={() => remove(t.id)}>✕</Button>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

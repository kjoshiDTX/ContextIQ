import { useState, useEffect } from 'react';

const NAV = [
  { id: 'query',     icon: '💬', label: 'Query' },
  { id: 'documents', icon: '📄', label: 'Documents' },
  { id: 'graph',     icon: '🕸️', label: 'Knowledge Graph' },
  { id: 'journal',   icon: '📓', label: 'Journal' },
  { id: 'tasks',     icon: '✅', label: 'Tasks' },
];

export function Sidebar({ active, onNavigate, user, onSignOut }) {
  const [health, setHealth] = useState('checking');

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch('http://localhost:8000/health');
        setHealth(r.ok ? 'ok' : 'error');
      } catch {
        setHealth('error');
      }
    };
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{
      width: 220, flexShrink: 0,
      background: 'var(--surface)', borderRight: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', padding: '20px 0',
    }}>
      {/* Logo */}
      <div style={{ padding: '0 20px 24px', display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          width: 32, height: 32, background: 'var(--accent)', borderRadius: 8,
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16,
        }}>⬡</div>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: '-.3px' }}>ContextIQ</div>
          <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 1 }}>Your Second Brain</div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1 }}>
        {NAV.map(item => (
          <div
            key={item.id}
            onClick={() => onNavigate(item.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '9px 20px', cursor: 'pointer', fontSize: 13,
              color: active === item.id ? 'var(--accent-h)' : 'var(--muted)',
              background: active === item.id ? 'rgba(99,102,241,.12)' : 'transparent',
              borderLeft: active === item.id ? '2px solid var(--accent)' : '2px solid transparent',
              transition: 'all .15s',
            }}
          >
            <span style={{ fontSize: 15, width: 18, textAlign: 'center' }}>{item.icon}</span>
            {item.label}
          </div>
        ))}
      </nav>

      {/* Health */}
      <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
          background: health === 'ok' ? 'var(--success)' : health === 'checking' ? 'var(--warn)' : 'var(--danger)',
        }} />
        <span style={{ fontSize: 12, color: 'var(--muted)' }}>
          {health === 'ok' ? 'API connected' : health === 'checking' ? 'Connecting…' : 'API offline'}
        </span>
      </div>

      {/* User + Sign Out */}
      <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border)' }}>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 8, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {user?.email}
        </div>
        <button
          onClick={onSignOut}
          style={{
            width: '100%', padding: '6px 0', fontSize: 12,
            background: 'transparent', border: '1px solid var(--border)',
            color: 'var(--muted)', borderRadius: 6, cursor: 'pointer',
            transition: 'all .15s',
          }}
          onMouseEnter={e => { e.target.style.borderColor = 'var(--danger)'; e.target.style.color = 'var(--danger)'; }}
          onMouseLeave={e => { e.target.style.borderColor = 'var(--border)'; e.target.style.color = 'var(--muted)'; }}
        >
          Sign out
        </button>
      </div>
    </div>
  );
}

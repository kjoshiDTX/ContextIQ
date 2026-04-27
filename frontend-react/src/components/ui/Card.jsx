export function Card({ children, style }) {
  return (
    <div style={{
      background: 'var(--card)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius)', padding: 20, ...style,
    }}>
      {children}
    </div>
  );
}

export function CardTitle({ children, style }) {
  return (
    <div style={{
      fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
      letterSpacing: '.6px', color: 'var(--muted)', marginBottom: 14,
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      ...style,
    }}>
      {children}
    </div>
  );
}

export function Badge({ children, variant = 'default' }) {
  const colors = {
    default:     'rgba(107,114,128,.2)',
    todo:        'rgba(107,114,128,.2)',
    in_progress: 'rgba(245,158,11,.15)',
    done:        'rgba(34,197,94,.15)',
    pending:     'rgba(245,158,11,.15)',
    extracted:   'rgba(34,197,94,.15)',
    skipped:     'rgba(107,114,128,.2)',
    failed:      'rgba(239,68,68,.15)',
    accent:      'rgba(99,102,241,.15)',
  };
  const textColors = {
    default:     '#9ca3af',
    todo:        '#9ca3af',
    in_progress: 'var(--warn)',
    done:        'var(--success)',
    pending:     'var(--warn)',
    extracted:   'var(--success)',
    skipped:     '#9ca3af',
    failed:      'var(--danger)',
    accent:      'var(--accent-h)',
  };
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      padding: '3px 9px', borderRadius: 20, fontSize: 11, fontWeight: 500,
      background: colors[variant] || colors.default,
      color: textColors[variant] || textColors.default,
    }}>
      {children}
    </span>
  );
}

export function Spinner() {
  return (
    <span style={{
      display: 'inline-block', width: 14, height: 14,
      border: '2px solid rgba(255,255,255,.2)',
      borderTopColor: '#fff', borderRadius: '50%',
      animation: 'spin .6s linear infinite',
    }} />
  );
}

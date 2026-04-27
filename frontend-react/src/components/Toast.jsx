export function ToastContainer({ toasts }) {
  const colors = {
    success: { bg: 'rgba(22,163,74,.2)',   border: 'rgba(34,197,94,.3)',   color: 'var(--success)' },
    error:   { bg: 'rgba(220,38,38,.2)',   border: 'rgba(239,68,68,.3)',   color: 'var(--danger)' },
    info:    { bg: 'rgba(99,102,241,.2)',  border: 'rgba(99,102,241,.3)',  color: 'var(--accent-h)' },
  };

  return (
    <div style={{ position: 'fixed', bottom: 24, right: 24, display: 'flex', flexDirection: 'column', gap: 8, zIndex: 1000 }}>
      {toasts.map(t => {
        const c = colors[t.type] || colors.info;
        return (
          <div key={t.id} style={{
            padding: '12px 16px', borderRadius: 8, fontSize: 13,
            maxWidth: 320, border: `1px solid ${c.border}`,
            background: c.bg, color: c.color,
            animation: 'slideIn .2s ease',
          }}>
            {t.message}
          </div>
        );
      })}
    </div>
  );
}

const styles = {
  base: {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    padding: '8px 14px', borderRadius: 7, border: 'none',
    cursor: 'pointer', fontSize: 13, fontWeight: 500,
    transition: 'all .15s', fontFamily: 'inherit',
  },
  primary:  { background: 'var(--accent)', color: '#fff' },
  ghost:    { background: 'transparent', color: 'var(--muted)', border: '1px solid var(--border)' },
  danger:   { background: 'rgba(239,68,68,.15)', color: 'var(--danger)', border: '1px solid rgba(239,68,68,.3)' },
  success:  { background: 'rgba(34,197,94,.15)', color: 'var(--success)', border: '1px solid rgba(34,197,94,.3)' },
  sm:       { padding: '5px 10px', fontSize: 12 },
  disabled: { opacity: .5, cursor: 'not-allowed' },
};

export function Button({ children, variant = 'primary', size, disabled, onClick, style, ...props }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        ...styles.base,
        ...styles[variant],
        ...(size === 'sm' ? styles.sm : {}),
        ...(disabled ? styles.disabled : {}),
        ...style,
      }}
      {...props}
    >
      {children}
    </button>
  );
}

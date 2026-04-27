import { useState } from 'react';
import { supabase } from '../lib/supabase';

export function AuthPage() {
  const [mode, setMode] = useState('signin'); // 'signin' | 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [confirmed, setConfirmed] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (mode === 'signup') {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        setConfirmed(true);
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        // AuthContext.onAuthStateChange handles redirect
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const toggle = () => {
    setMode(m => m === 'signin' ? 'signup' : 'signin');
    setError('');
    setConfirmed(false);
  };

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--bg)',
    }}>
      <div style={{ width: '100%', maxWidth: 400, padding: '0 24px' }}>

        {/* Branding */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{
            width: 48, height: 48, background: 'var(--accent)', borderRadius: 12,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 24, margin: '0 auto 12px',
          }}>⬡</div>
          <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-.4px' }}>ContextIQ</div>
          <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 4 }}>Knowledge Graph RAG</div>
        </div>

        {/* Card */}
        <div style={{
          background: 'var(--surface)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius)', padding: 28,
        }}>
          {confirmed ? (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>📬</div>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Check your email</div>
              <div style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.6 }}>
                We sent a confirmation link to <strong>{email}</strong>.
                Click it to activate your account, then come back to sign in.
              </div>
              <button
                onClick={toggle}
                style={{
                  marginTop: 20, background: 'none', border: 'none',
                  color: 'var(--accent)', cursor: 'pointer', fontSize: 13,
                }}
              >
                ← Back to sign in
              </button>
            </div>
          ) : (
            <>
              <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 20 }}>
                {mode === 'signin' ? 'Sign in to your account' : 'Create an account'}
              </h2>

              <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div>
                  <label style={{ fontSize: 11, color: 'var(--muted)', display: 'block', marginBottom: 5, textTransform: 'uppercase', letterSpacing: '.5px' }}>
                    Email
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    required
                    autoComplete="email"
                    style={{ width: '100%', boxSizing: 'border-box' }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: 11, color: 'var(--muted)', display: 'block', marginBottom: 5, textTransform: 'uppercase', letterSpacing: '.5px' }}>
                    Password
                  </label>
                  <input
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder={mode === 'signup' ? 'At least 6 characters' : '••••••••'}
                    required
                    autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
                    style={{ width: '100%', boxSizing: 'border-box' }}
                  />
                </div>

                {error && (
                  <div style={{
                    fontSize: 13, color: 'var(--danger)', padding: '8px 12px',
                    background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.2)',
                    borderRadius: 6,
                  }}>
                    {error}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  style={{
                    padding: '10px 0', background: 'var(--accent)', color: '#fff',
                    border: 'none', borderRadius: 6, fontSize: 14, fontWeight: 600,
                    cursor: loading ? 'not-allowed' : 'pointer',
                    opacity: loading ? 0.7 : 1, marginTop: 4,
                    transition: 'opacity .15s',
                  }}
                >
                  {loading ? '…' : mode === 'signin' ? 'Sign In' : 'Create Account'}
                </button>
              </form>

              <div style={{ marginTop: 18, textAlign: 'center', fontSize: 13, color: 'var(--muted)' }}>
                {mode === 'signin' ? "Don't have an account? " : 'Already have an account? '}
                <button
                  onClick={toggle}
                  style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontSize: 13, fontWeight: 500 }}
                >
                  {mode === 'signin' ? 'Sign Up' : 'Sign In'}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

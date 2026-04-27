import { useState, useRef, useEffect } from 'react';
import { queryDocuments } from '../../api';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Spinner } from '../ui/Card';

// ── Source badge styling ──────────────────────────────────────────
const SOURCE_STYLES = {
  document:     { bg: 'rgba(99,102,241,.15)',  border: 'rgba(99,102,241,.3)',  color: '#818cf8', label: 'Doc' },
  journal:      { bg: 'rgba(245,158,11,.15)',  border: 'rgba(245,158,11,.3)',  color: '#fbbf24', label: 'Journal' },
  conversation: { bg: 'rgba(139,92,246,.15)',  border: 'rgba(139,92,246,.3)',  color: '#a78bfa', label: 'Chat' },
  personal:     { bg: 'rgba(245,158,11,.15)',  border: 'rgba(245,158,11,.3)',  color: '#fbbf24', label: 'Personal' },
};

function SourceBadge({ type }) {
  const style = SOURCE_STYLES[type] || SOURCE_STYLES.document;
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, letterSpacing: '.04em', textTransform: 'uppercase',
      padding: '2px 6px', borderRadius: 99,
      background: style.bg, border: `1px solid ${style.border}`, color: style.color,
    }}>
      {style.label}
    </span>
  );
}

function UserMessage({ text }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
      <div style={{
        padding: '12px 16px', borderRadius: 12, borderBottomRightRadius: 3,
        background: 'var(--accent)', color: '#fff', maxWidth: '80%', lineHeight: 1.6,
      }}>
        {text}
      </div>
    </div>
  );
}

function AIMessage({ data }) {
  const [showSources, setShowSources] = useState(false);

  const docSources = (data.sources || []).filter(s => s.source_type === 'document' || s.document_id);
  const personalSources = (data.sources || []).filter(s => s.source_type === 'journal' || s.source_type === 'conversation');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-start' }}>
      <div style={{
        padding: '12px 16px', borderRadius: 12, borderBottomLeftRadius: 3,
        background: 'var(--surface)', border: '1px solid var(--border)',
        maxWidth: '85%', lineHeight: 1.7, whiteSpace: 'pre-wrap',
      }}>
        {data.answer}
      </div>

      {/* Graph facts */}
      {data.graph_context?.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, maxWidth: '85%' }}>
          {data.graph_context.map((f, i) => (
            <div key={i} style={{
              fontSize: 11, padding: '4px 10px', color: 'var(--muted)',
              background: 'rgba(99,102,241,.07)', borderRadius: 5,
              borderLeft: '2px solid var(--accent)',
            }}>
              {f.head} <span style={{ color: 'var(--accent)' }}>—[{f.relation}]→</span> {f.tail}
            </div>
          ))}
        </div>
      )}

      {/* Sources toggle */}
      {data.sources?.length > 0 && (
        <div>
          <button
            onClick={() => setShowSources(s => !s)}
            style={{ fontSize: 11, color: 'var(--muted)', background: 'none', border: 'none', cursor: 'pointer', padding: '2px 0' }}
          >
            {showSources ? '▾' : '▸'} {data.sources.length} source{data.sources.length > 1 ? 's' : ''}
            {data.cached && (
              <span style={{ marginLeft: 8, padding: '2px 7px', background: 'rgba(245,158,11,.15)', color: 'var(--warn)', borderRadius: 20, fontSize: 10 }}>
                ⚡ cached
              </span>
            )}
          </button>

          {showSources && (
            <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 10 }}>
              {docSources.length > 0 && (
                <div>
                  <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 5, textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>Documents</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {docSources.map((s, i) => (
                      <span key={i} title={s.content_preview} style={{
                        display: 'flex', alignItems: 'center', gap: 5,
                        fontSize: 11, padding: '3px 9px', cursor: 'help',
                        background: 'rgba(99,102,241,.12)', border: '1px solid rgba(99,102,241,.3)',
                        borderRadius: 20, color: '#818cf8',
                      }}>
                        <SourceBadge type="document" />
                        {s.document_id?.slice(0, 8) || 'doc'}… ({s.similarity != null ? (s.similarity * 100).toFixed(0) : '—'}%)
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {personalSources.length > 0 && (
                <div>
                  <div style={{ fontSize: 10, color: 'var(--muted)', marginBottom: 5, textTransform: 'uppercase', letterSpacing: '.05em', fontWeight: 600 }}>Personal context</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {personalSources.map((s, i) => {
                      const type = s.source_type || 'personal';
                      const style = SOURCE_STYLES[type] || SOURCE_STYLES.personal;
                      return (
                        <span key={i} title={s.content_preview} style={{
                          display: 'flex', alignItems: 'center', gap: 5,
                          fontSize: 11, padding: '3px 9px', cursor: 'help',
                          background: style.bg, border: `1px solid ${style.border}`,
                          borderRadius: 20, color: style.color,
                        }}>
                          <SourceBadge type={type} />
                          {s.content_preview?.slice(0, 30)}…
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function QueryPanel({ userId, toast }) {
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      data: {
        answer: 'Hello! I have access to everything you\'ve shared — your documents, journal reflections, Claude conversations, and tasks. Ask me anything about yourself, your ideas, or what you\'ve been learning.',
      },
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput('');
    setMessages(m => [...m, { role: 'user', text: q }]);
    setLoading(true);
    try {
      const data = await queryDocuments(q);
      setMessages(m => [...m, { role: 'ai', data }]);
    } catch (e) {
      toast(e.message, 'error');
      setMessages(m => [...m, { role: 'ai', data: { answer: 'Query failed. Check the backend is running.' } }]);
    } finally {
      setLoading(false);
    }
  };

  // Example prompts to help users get started
  const examples = [
    'What are my current goals?',
    'What have I been learning about lately?',
    'How do my ideas connect to each other?',
    'What tasks do I have coming up?',
  ];

  return (
    <Card style={{ display: 'flex', flexDirection: 'column', gap: 16, flex: 1 }}>
      <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.6px', color: 'var(--muted)' }}>
        Your Second Brain
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14, minHeight: 200, maxHeight: 480, paddingRight: 4 }}>
        {messages.map((m, i) =>
          m.role === 'user'
            ? <UserMessage key={i} text={m.text} />
            : <AIMessage key={i} data={m.data} />
        )}
        {loading && (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', color: 'var(--muted)', fontSize: 13 }}>
            <Spinner /> Searching your knowledge graph…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid var(--border)' }} />

      {/* Example prompts — only shown when just the welcome message is present */}
      {messages.length === 1 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {examples.map(ex => (
            <button
              key={ex}
              onClick={() => { setInput(ex); }}
              style={{
                fontSize: 12, padding: '5px 12px', borderRadius: 20,
                background: 'var(--surface)', border: '1px solid var(--border)',
                color: 'var(--muted)', cursor: 'pointer', transition: 'all .15s',
              }}
              onMouseEnter={e => { e.target.style.borderColor = 'var(--accent)'; e.target.style.color = 'var(--accent-h)'; }}
              onMouseLeave={e => { e.target.style.borderColor = 'var(--border)'; e.target.style.color = 'var(--muted)'; }}
            >
              {ex}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{ display: 'flex', gap: 10 }}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder="Ask anything about yourself…"
          disabled={loading}
        />
        <Button onClick={send} disabled={loading || !input.trim()}>
          {loading ? <Spinner /> : 'Ask'}
        </Button>
      </div>
    </Card>
  );
}

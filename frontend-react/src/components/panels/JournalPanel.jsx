import { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { listJournalEntries, deleteJournalEntry, getJournalGraph, getJournalQuestions, createJournalSession } from '../../api';
import { Card, CardTitle, Badge, Spinner } from '../ui/Card';
import { Button } from '../ui/Button';

function fmtDate(s) {
  if (!s) return '—';
  return new Date(s).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

// ── Guided Session States ─────────────────────────────────────────
// idle → loading_questions → answering (step 0,1,2) → review → saving → idle

function SessionProgress({ current, total }) {
  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 20 }}>
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{
          height: 4, flex: 1, borderRadius: 2,
          background: i < current ? 'var(--accent)' : i === current ? 'var(--accent-h)' : 'var(--border)',
          transition: 'background .3s',
        }} />
      ))}
      <span style={{ fontSize: 11, color: 'var(--muted)', whiteSpace: 'nowrap', marginLeft: 6 }}>
        {current + 1} / {total}
      </span>
    </div>
  );
}

function CategoryChip({ label }) {
  const COLORS = {
    'Goals & Vision':       { bg: 'rgba(99,102,241,.15)',  text: '#818cf8' },
    'Challenges & Growth':  { bg: 'rgba(239,68,68,.12)',   text: '#f87171' },
    'Learning & Insights':  { bg: 'rgba(16,185,129,.12)',  text: '#34d399' },
    'Relationships':        { bg: 'rgba(245,158,11,.12)',  text: '#fbbf24' },
    'Emotions & Wellbeing': { bg: 'rgba(139,92,246,.12)',  text: '#a78bfa' },
    'Reflection & Decisions': { bg: 'rgba(6,182,212,.12)', text: '#22d3ee' },
  };
  const c = COLORS[label] || { bg: 'rgba(120,120,120,.12)', text: 'var(--muted)' };
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, letterSpacing: '.04em', textTransform: 'uppercase',
      padding: '3px 8px', borderRadius: 99,
      background: c.bg, color: c.text,
    }}>
      {label}
    </span>
  );
}

function GuidedSession({ onSaved, toast, sessionCount }) {
  const [phase, setPhase] = useState('idle'); // idle | loading | answering | review | saving
  const [questions, setQuestions] = useState([]);
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({}); // { questionId: answer }
  const [currentAnswer, setCurrentAnswer] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    if (phase === 'answering' && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [phase, step]);

  const start = async () => {
    setPhase('loading');
    try {
      const data = await getJournalQuestions(sessionCount);
      setQuestions(data.questions || []);
      setStep(0);
      setAnswers({});
      setCurrentAnswer('');
      setPhase('answering');
    } catch (e) {
      toast(e.message, 'error');
      setPhase('idle');
    }
  };

  const handleNext = () => {
    if (!currentAnswer.trim()) { toast('Write at least a few words before continuing', 'error'); return; }
    const q = questions[step];
    const newAnswers = { ...answers, [q.id]: currentAnswer.trim() };
    setAnswers(newAnswers);
    setCurrentAnswer('');

    if (step + 1 < questions.length) {
      setStep(step + 1);
    } else {
      setPhase('review');
    }
  };

  const handleBack = () => {
    if (step === 0) { setPhase('idle'); return; }
    const prevQ = questions[step - 1];
    setCurrentAnswer(answers[prevQ.id] || '');
    const newAnswers = { ...answers };
    delete newAnswers[prevQ.id];
    setAnswers(newAnswers);
    setStep(step - 1);
  };

  const handleSave = async () => {
    setPhase('saving');
    const sessionData = questions.map(q => ({
      question: q.text,
      answer: answers[q.id] || '',
    }));
    try {
      await createJournalSession(sessionData);
      toast('Reflection saved — extracting concepts in background…', 'success');
      setPhase('idle');
      onSaved();
    } catch (e) {
      toast(e.message, 'error');
      setPhase('review');
    }
  };

  // ── Idle ─────────────────────────────────────────────────────
  if (phase === 'idle') {
    return (
      <Card style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16, padding: 32, textAlign: 'center' }}>
        <div style={{ fontSize: 36, lineHeight: 1 }}>🪞</div>
        <div>
          <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 6 }}>Start a Reflection</div>
          <div style={{ fontSize: 13, color: 'var(--muted)', maxWidth: 320 }}>
            You'll be guided through 3 thought-provoking questions. Your answers build your personal knowledge graph.
          </div>
        </div>
        <Button onClick={start} style={{ minWidth: 160 }}>
          Begin Reflection
        </Button>
      </Card>
    );
  }

  // ── Loading ───────────────────────────────────────────────────
  if (phase === 'loading') {
    return (
      <Card style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, padding: 40 }}>
        <Spinner />
        <span style={{ color: 'var(--muted)', fontSize: 13 }}>Preparing your questions…</span>
      </Card>
    );
  }

  // ── Answering ─────────────────────────────────────────────────
  if (phase === 'answering') {
    const q = questions[step];
    return (
      <Card style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <SessionProgress current={step} total={questions.length} />
        <CategoryChip label={q.category} />
        <div style={{ fontSize: 16, fontWeight: 500, lineHeight: 1.5, color: 'var(--text)' }}>
          {q.text}
        </div>
        <textarea
          ref={textareaRef}
          value={currentAnswer}
          onChange={e => setCurrentAnswer(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) handleNext(); }}
          placeholder="Write your thoughts here…"
          style={{ minHeight: 140, resize: 'vertical' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Button variant="ghost" size="sm" onClick={handleBack}>
            {step === 0 ? 'Cancel' : '← Back'}
          </Button>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 11, color: 'var(--muted)' }}>⌘↵ to continue</span>
            <Button onClick={handleNext} disabled={!currentAnswer.trim()}>
              {step + 1 < questions.length ? 'Next Question →' : 'Review →'}
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  // ── Review ────────────────────────────────────────────────────
  if (phase === 'review') {
    return (
      <Card style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <CardTitle>Review your reflection</CardTitle>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {questions.map((q, i) => (
            <div key={q.id} style={{
              borderLeft: '3px solid var(--accent)', paddingLeft: 14,
              display: 'flex', flexDirection: 'column', gap: 6,
            }}>
              <CategoryChip label={q.category} />
              <div style={{ fontSize: 12, color: 'var(--muted)', fontStyle: 'italic' }}>{q.text}</div>
              <div style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.5 }}>{answers[q.id]}</div>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: 8, borderTop: '1px solid var(--border)' }}>
          <Button variant="ghost" size="sm" onClick={() => { setStep(questions.length - 1); setPhase('answering'); }}>
            ← Edit answers
          </Button>
          <Button onClick={handleSave}>
            Save Reflection
          </Button>
        </div>
      </Card>
    );
  }

  // ── Saving ────────────────────────────────────────────────────
  if (phase === 'saving') {
    return (
      <Card style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, padding: 40 }}>
        <Spinner />
        <span style={{ color: 'var(--muted)', fontSize: 13 }}>Saving your reflection…</span>
      </Card>
    );
  }

  return null;
}

function MiniGraph({ svgRef }) {
  return (
    <div style={{ height: 360, background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden', position: 'relative' }}>
      <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}

export function JournalPanel({ userId, toast }) {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionCount, setSessionCount] = useState(0);
  const svgRef = useRef(null);

  useEffect(() => { loadEntries(); loadGraph(); }, [userId]);

  const loadEntries = async () => {
    setLoading(true);
    try {
      const d = await listJournalEntries();
      const loaded = d.entries || [];
      setEntries(loaded);
      setSessionCount(loaded.length);
    } catch (e) { toast(e.message, 'error'); }
    finally { setLoading(false); }
  };

  const loadGraph = async () => {
    try {
      const data = await getJournalGraph();
      renderMiniGraph(data);
    } catch {}
  };

  const renderMiniGraph = (data) => {
    if (!svgRef.current || !data?.nodes?.length) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const W = svgRef.current.clientWidth || 400;
    const H = svgRef.current.clientHeight || 360;
    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const nodes = data.nodes.map(d => ({ ...d }));
    const links = data.links.map(d => ({ ...d }));
    const color = d3.scaleOrdinal(d3.schemeTableau10);

    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([.1, 4]).on('zoom', e => g.attr('transform', e.transform)));

    d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(70))
      .force('charge', d3.forceManyBody().strength(-150))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .on('tick', () => {
        g.selectAll('line').attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        g.selectAll('circle').attr('cx', d => d.x).attr('cy', d => d.y);
        g.selectAll('text').attr('x', d => d.x).attr('y', d => d.y);
      });

    g.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', '#2a2a3a').attr('stroke-width', 1.2).attr('stroke-opacity', .5);

    g.append('g').selectAll('circle').data(nodes).join('circle')
      .attr('r', 7).attr('fill', d => d.source === 'journal' ? '#f59e0b' : color(d.type || 'x'))
      .attr('stroke', '#0d0d14').attr('stroke-width', 2);

    g.append('g').selectAll('text').data(nodes).join('text')
      .text(d => (d.name || d.id || '').slice(0, 16))
      .attr('fill', '#e8e8f0').attr('font-size', 9).attr('text-anchor', 'middle').attr('dy', -11)
      .style('pointer-events', 'none').style('user-select', 'none');
  };

  const handleDelete = async (id) => {
    if (!confirm('Delete this entry?')) return;
    try {
      await deleteJournalEntry(id);
      toast('Entry deleted', 'success');
      loadEntries(); loadGraph();
    } catch (e) { toast(e.message, 'error'); }
  };

  const onSaved = () => {
    loadEntries();
    setTimeout(loadGraph, 5000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Guided session */}
        <GuidedSession onSaved={onSaved} toast={toast} sessionCount={sessionCount} />

        {/* Mini graph */}
        <Card>
          <CardTitle>
            Concept Graph
            <Button variant="ghost" size="sm" onClick={loadGraph}>↻</Button>
          </CardTitle>
          <MiniGraph svgRef={svgRef} />
        </Card>
      </div>

      {/* Entries list */}
      <Card>
        <CardTitle>
          Past Reflections
          <Button variant="ghost" size="sm" onClick={loadEntries}>↻ Refresh</Button>
        </CardTitle>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 32 }}><Spinner /></div>
        ) : entries.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--muted)', padding: 32 }}>
            No reflections yet — start your first session above.
          </div>
        ) : (
          <table>
            <thead><tr><th>Reflection</th><th>Status</th><th>Concepts</th><th>Date</th><th></th></tr></thead>
            <tbody>
              {entries.map(e => (
                <tr key={e.id}>
                  <td>
                    <div style={{ fontWeight: 500, maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {e.title || e.content?.slice(0, 60) + '…'}
                    </div>
                  </td>
                  <td>
                    <Badge variant={e.extraction_status}>{e.extraction_status}</Badge>
                    {e.skip_reason && <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 2 }}>{e.skip_reason}</div>}
                  </td>
                  <td>{e.concept_count || 0}</td>
                  <td style={{ color: 'var(--muted)' }}>{fmtDate(e.created_at)}</td>
                  <td><Button variant="danger" size="sm" onClick={() => handleDelete(e.id)}>Delete</Button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}

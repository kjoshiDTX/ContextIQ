import { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { getGraph, detectCommunities, setNodeContext } from '../../api';
import { Card, CardTitle, Spinner } from '../ui/Card';
import { Button } from '../ui/Button';

// ── Node color logic ──────────────────────────────────────────────
// Document nodes: Tableau10 by community/type
// Journal nodes:  amber  (#F59E0B)
// Conversation:   purple (#8B5CF6)
const communityColor = d3.scaleOrdinal(d3.schemeTableau10);

function nodeColor(n) {
  if (n.source === 'journal') return '#F59E0B';
  if (n.source === 'conversation') return '#8B5CF6';
  return communityColor(n.community_id ?? n.community ?? n.type ?? 'x');
}

function useD3Graph(svgRef, data, onNodeClick) {
  useEffect(() => {
    if (!svgRef.current || !data?.nodes?.length) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const W = svgRef.current.clientWidth || 800;
    const H = svgRef.current.clientHeight || 540;
    svg.attr('viewBox', `0 0 ${W} ${H}`);

    const nodes = data.nodes.map(d => ({ ...d }));
    const links = data.links.map(d => ({ ...d }));
    const nodeR = n => 6 + Math.min((n.connections || 1) * 1.5, 14);

    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([.1, 4]).on('zoom', e => g.attr('transform', e.transform)));

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(90))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide(d => nodeR(d) + 4));

    const link = g.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', '#2a2a3a').attr('stroke-width', 1.5).attr('stroke-opacity', .6);

    const linkLabel = g.append('g').selectAll('text').data(links).join('text')
      .attr('fill', '#4a4a60').attr('font-size', 9).attr('text-anchor', 'middle')
      .text(d => d.relation || '');

    const node = g.append('g').selectAll('circle').data(nodes).join('circle')
      .attr('r', nodeR).attr('fill', nodeColor)
      .attr('stroke', '#0d0d14').attr('stroke-width', 2).style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

    const label = g.append('g').selectAll('text').data(nodes).join('text')
      .text(d => (d.name || d.id || '').slice(0, 20))
      .attr('fill', '#e8e8f0').attr('font-size', 10).attr('text-anchor', 'middle')
      .attr('dy', d => -(nodeR(d) + 4))
      .style('pointer-events', 'none').style('user-select', 'none');

    if (onNodeClick) node.on('click', (e, d) => onNodeClick(d));

    sim.on('tick', () => {
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      linkLabel
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      label.attr('x', d => d.x).attr('y', d => d.y);
    });

    return () => sim.stop();
  }, [data]);
}

// ── Source filter button ──────────────────────────────────────────
function FilterChip({ label, color, active, count, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '4px 12px', borderRadius: 99, fontSize: 12, fontWeight: 500,
        border: `1px solid ${active ? color : 'var(--border)'}`,
        background: active ? `${color}22` : 'transparent',
        color: active ? color : 'var(--muted)',
        cursor: 'pointer', transition: 'all .15s',
      }}
    >
      <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
      {label}
      {count !== undefined && (
        <span style={{ fontSize: 10, opacity: .7 }}>({count})</span>
      )}
    </button>
  );
}

export function GraphPanel({ userId, toast }) {
  const svgRef = useRef(null);
  const [rawData, setRawData] = useState({ nodes: [], links: [], stats: {} });
  const [sourceFilter, setSourceFilter] = useState('all'); // all | document | journal | conversation
  const [limit, setLimit] = useState(60);
  const [loading, setLoading] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeCtx, setNodeCtx] = useState('');

  // Apply source filter to produce the visible dataset
  const filteredData = useMemo(() => {
    if (sourceFilter === 'all') return rawData;
    const visibleNodes = (rawData.nodes || []).filter(n => n.source === sourceFilter);
    const visibleIds = new Set(visibleNodes.map(n => n.id));
    const visibleLinks = (rawData.links || []).filter(
      l => visibleIds.has(typeof l.source === 'object' ? l.source.id : l.source)
        && visibleIds.has(typeof l.target === 'object' ? l.target.id : l.target)
    );
    return { nodes: visibleNodes, links: visibleLinks };
  }, [rawData, sourceFilter]);

  useD3Graph(svgRef, filteredData, node => {
    setSelectedNode(node);
    setNodeCtx(node.user_context || '');
  });

  const load = async () => {
    setLoading(true);
    try {
      const d = await getGraph(limit);
      setRawData(d);
      const stats = d.stats || {};
      toast(
        `Graph: ${d.nodes?.length || 0} nodes · ${stats.doc_count || 0} docs · ${stats.journal_count || 0} journal · ${stats.conversation_count || 0} conversations`,
        'info'
      );
    } catch (e) { toast(e.message, 'error'); }
    finally { setLoading(false); }
  };

  const runCommunities = async () => {
    setDetecting(true);
    try {
      const r = await detectCommunities();
      toast(`Communities found: ${r.communities_found ?? '?'}`, 'success');
      load();
    } catch (e) { toast(e.message, 'error'); }
    finally { setDetecting(false); }
  };

  const saveContext = async () => {
    if (!selectedNode) return;
    try {
      await setNodeContext(selectedNode.name || selectedNode.id, nodeCtx);
      toast('Note saved', 'success');
    } catch (e) { toast(e.message, 'error'); }
  };

  useEffect(() => { load(); }, []);

  const stats = rawData.stats || {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, flex: 1 }}>
      {/* Toolbar */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button onClick={load} disabled={loading}>
          {loading ? <Spinner /> : '↻ Reload'}
        </Button>
        <Button variant="ghost" onClick={runCommunities} disabled={detecting}>
          {detecting ? <Spinner /> : '🔍 Communities'}
        </Button>
        <input type="range" min={10} max={200} value={limit}
          onChange={e => setLimit(+e.target.value)} style={{ width: 120 }} />
        <span style={{ color: 'var(--muted)', fontSize: 12 }}>{limit} nodes</span>
      </div>

      {/* Source filter + stats */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <FilterChip
          label="All" color="var(--accent)" active={sourceFilter === 'all'}
          count={rawData.nodes?.length}
          onClick={() => setSourceFilter('all')}
        />
        <FilterChip
          label="Documents" color="#6366f1" active={sourceFilter === 'document'}
          count={stats.doc_count}
          onClick={() => setSourceFilter('document')}
        />
        <FilterChip
          label="Journal" color="#F59E0B" active={sourceFilter === 'journal'}
          count={stats.journal_count}
          onClick={() => setSourceFilter('journal')}
        />
        <FilterChip
          label="Conversations" color="#8B5CF6" active={sourceFilter === 'conversation'}
          count={stats.conversation_count}
          onClick={() => setSourceFilter('conversation')}
        />
        {/* Legend */}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 12, alignItems: 'center', fontSize: 11, color: 'var(--muted)' }}>
          <span>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#6366f1', marginRight: 4 }} />
            Documents
          </span>
          <span>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#F59E0B', marginRight: 4 }} />
            Journal
          </span>
          <span>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#8B5CF6', marginRight: 4 }} />
            Conversations
          </span>
        </div>
      </div>

      {/* Graph canvas */}
      <div style={{
        flex: 1, position: 'relative',
        background: 'var(--card)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius)', overflow: 'hidden', minHeight: 520,
      }}>
        <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />

        {!filteredData.nodes?.length && !loading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 8, color: 'var(--muted)' }}>
            <div style={{ fontSize: 28 }}>🕸️</div>
            <div style={{ fontSize: 13 }}>
              {sourceFilter === 'all'
                ? 'No graph data — upload documents or complete a journal session first.'
                : `No ${sourceFilter} nodes yet.`}
            </div>
          </div>
        )}

        {/* Node info panel */}
        {selectedNode && (
          <div style={{
            position: 'absolute', right: 16, top: 16, width: 240,
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius)', padding: 14,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
              <div>
                <strong style={{ fontSize: 13 }}>{selectedNode.name || selectedNode.id}</strong>
                <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                  <span style={{
                    fontSize: 10, padding: '2px 7px', borderRadius: 99,
                    background: selectedNode.source === 'journal' ? 'rgba(245,158,11,.15)' :
                                selectedNode.source === 'conversation' ? 'rgba(139,92,246,.15)' :
                                'rgba(99,102,241,.15)',
                    color: selectedNode.source === 'journal' ? '#F59E0B' :
                           selectedNode.source === 'conversation' ? '#8B5CF6' :
                           'var(--accent)',
                  }}>
                    {selectedNode.source || 'document'}
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--muted)' }}>{selectedNode.type}</span>
                </div>
              </div>
              <button onClick={() => setSelectedNode(null)} style={{ background: 'none', border: 'none', color: 'var(--muted)', cursor: 'pointer', fontSize: 16, lineHeight: 1 }}>×</button>
            </div>
            <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 10 }}>
              {selectedNode.connections ?? 0} connections
            </div>
            {selectedNode.source !== 'conversation' && (
              <>
                <textarea
                  value={nodeCtx}
                  onChange={e => setNodeCtx(e.target.value)}
                  placeholder="Add a note about this concept…"
                  style={{ fontSize: 12, minHeight: 60 }}
                />
                <Button onClick={saveContext} style={{ marginTop: 8, width: '100%', justifyContent: 'center' }}>
                  Save Note
                </Button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

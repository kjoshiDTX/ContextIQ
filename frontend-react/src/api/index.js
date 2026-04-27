import { supabase } from '../lib/supabase';

const BASE = 'http://localhost:8000/api';

async function authHeaders(extra = {}) {
  const { data: { session } } = await supabase.auth.getSession();
  if (!session) throw new Error('Not authenticated');
  return {
    'Authorization': `Bearer ${session.access_token}`,
    ...extra,
  };
}

async function req(path, options = {}) {
  const headers = await authHeaders(options.headers);
  const res = await fetch(`${BASE}${path}`, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

// ── Documents ─────────────────────────────────
export const uploadDocument = async (file, categoryId) => {
  const { data: { session } } = await supabase.auth.getSession();
  if (!session) throw new Error('Not authenticated');

  const form = new FormData();
  form.append('file', file);
  if (categoryId) form.append('category_id', categoryId);

  const res = await fetch(`${BASE}/upload`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${session.access_token}` },
    body: form,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
};

export const listDocuments = () => req('/documents');

export const deleteDocument = (documentId) =>
  req(`/documents/${documentId}`, { method: 'DELETE' });

// ── Query ─────────────────────────────────────
export const queryDocuments = (query) =>
  req('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });

// ── Graph ─────────────────────────────────────
export const getGraph = async (limit = 60) => {
  // Graph is public — no auth needed, but still attach token if available
  const { data: { session } } = await supabase.auth.getSession();
  const headers = session ? { 'Authorization': `Bearer ${session.access_token}` } : {};
  const res = await fetch(`${BASE}/graph?limit=${limit}`, { headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status}`);
  return data;
};

export const detectCommunities = () =>
  req('/graph/detect-communities', { method: 'POST' });

export const setNodeContext = (nodeName, context) =>
  req('/graph/node-context', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ node_name: nodeName, context }),
  });

// ── Journal ───────────────────────────────────
export const createJournalEntry = (content, title) =>
  req('/journal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, title: title || undefined }),
  });

export const listJournalEntries = (limit = 50) =>
  req(`/journal?limit=${limit}`);

export const deleteJournalEntry = (entryId) =>
  req(`/journal/${entryId}`, { method: 'DELETE' });

export const getJournalGraph = () =>
  req('/journal/graph');

// ── Tasks ─────────────────────────────────────
export const createTask = (task) =>
  req('/tasks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(task),
  });

export const listTasks = () => req('/tasks');

export const updateTask = (taskId, updates) =>
  req(`/tasks/${taskId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });

export const deleteTask = (taskId) =>
  req(`/tasks/${taskId}`, { method: 'DELETE' });

// ── MCP ───────────────────────────────────────
export const getMcpKey = () => req('/mcp/key');

export const saveConversationInsight = (content, title) =>
  req('/mcp/insight', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, title: title || undefined }),
  });

// ── Journal Questions ─────────────────────────
export const getJournalQuestions = () => req('/journal/questions');

export const createJournalSession = (sessionData, title) =>
  req('/journal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content: '', title: title || undefined, session_data: sessionData }),
  });

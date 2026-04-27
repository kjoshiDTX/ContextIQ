import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ToastContainer } from './components/Toast';
import { QueryPanel } from './components/panels/QueryPanel';
import { DocumentsPanel } from './components/panels/DocumentsPanel';
import { GraphPanel } from './components/panels/GraphPanel';
import { JournalPanel } from './components/panels/JournalPanel';
import { TasksPanel } from './components/panels/TasksPanel';
import { AuthPage } from './components/AuthPage';
import { useToast } from './hooks/useToast';
import { useAuth } from './contexts/AuthContext';
import { Spinner } from './components/ui/Card';

const SECTIONS = {
  query:      { title: 'Query',           desc: 'Ask anything about yourself — your documents, journal, conversations, and tasks' },
  documents:  { title: 'Documents',       desc: 'Upload and manage your document library' },
  graph:      { title: 'Knowledge Graph', desc: 'Visualize how your ideas, documents, and conversations connect' },
  journal:    { title: 'Journal',         desc: 'Guided reflections that build your personal knowledge graph' },
  tasks:      { title: 'Tasks',           desc: 'Track to-dos — they appear as context in AI queries' },
};

export default function App() {
  const [section, setSection] = useState('query');
  const { session, signOut } = useAuth();
  const { toasts, toast } = useToast();

  // Still resolving session from Supabase
  if (session === undefined) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg)' }}>
        <Spinner />
      </div>
    );
  }

  // Not logged in — show auth screen
  if (!session) {
    return <AuthPage />;
  }

  const panelProps = { toast };

  return (
    <>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes slideIn { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
      `}</style>

      <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
        <Sidebar
          active={section}
          onNavigate={setSection}
          user={session.user}
          onSignOut={signOut}
        />

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Topbar */}
          <div style={{
            padding: '16px 28px', borderBottom: '1px solid var(--border)',
            background: 'var(--surface)', flexShrink: 0,
          }}>
            <h1 style={{ fontSize: 16, fontWeight: 600 }}>{SECTIONS[section].title}</h1>
            <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 2 }}>{SECTIONS[section].desc}</p>
          </div>

          {/* Panel */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px', display: 'flex', flexDirection: 'column' }}>
            {section === 'query'      && <QueryPanel      {...panelProps} />}
            {section === 'documents'  && <DocumentsPanel  {...panelProps} />}
            {section === 'graph'      && <GraphPanel      {...panelProps} />}
            {section === 'journal'    && <JournalPanel    {...panelProps} />}
            {section === 'tasks'      && <TasksPanel      {...panelProps} />}
          </div>
        </div>
      </div>

      <ToastContainer toasts={toasts} />
    </>
  );
}

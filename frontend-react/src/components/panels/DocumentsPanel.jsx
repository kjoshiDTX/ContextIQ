import { useState, useEffect, useRef } from 'react';
import { uploadDocument, listDocuments, deleteDocument } from '../../api';
import { Card, CardTitle, Badge, Spinner } from '../ui/Card';
import { Button } from '../ui/Button';

function fmtDate(s) {
  if (!s) return '—';
  return new Date(s).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

export function DocumentsPanel({ userId, toast }) {
  const [docs, setDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef(null);

  useEffect(() => { load(); }, [userId]);

  const load = async () => {
    setLoading(true);
    try {
      const d = await listDocuments();
      setDocs(d.documents || []);
    } catch (e) { toast(e.message, 'error'); }
    finally { setLoading(false); }
  };

  const handleFile = async (file) => {
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadDocument(file, undefined);
      toast(`✓ "${file.name}" — ${result.chunks_created} chunks, ${result.triplets_extracted} triplets`, 'success');
      load();
    } catch (e) { toast(e.message, 'error'); }
    finally { setUploading(false); if (fileRef.current) fileRef.current.value = ''; }
  };

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleDelete = async (id) => {
    if (!confirm('Remove this document?')) return;
    try {
      await deleteDocument(id);
      toast('Document removed', 'success');
      load();
    } catch (e) { toast(e.message, 'error'); }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Upload */}
      <Card>
        <CardTitle>Upload Document</CardTitle>
        <div
          onClick={() => !uploading && fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          style={{
            border: `2px dashed ${dragging ? 'var(--accent)' : 'var(--border)'}`,
            borderRadius: 'var(--radius)', padding: 40, textAlign: 'center',
            cursor: uploading ? 'not-allowed' : 'pointer',
            background: dragging ? 'rgba(99,102,241,.07)' : 'transparent',
            transition: 'all .2s',
          }}
        >
          <div style={{ fontSize: 32, marginBottom: 10 }}>📂</div>
          {uploading
            ? <><Spinner /> <span style={{ color: 'var(--muted)', marginLeft: 8 }}>Uploading & ingesting…</span></>
            : <>
                <p><strong>Click to upload</strong> or drag and drop</p>
                <p style={{ color: 'var(--muted)', marginTop: 6, fontSize: 13 }}>PDF, DOCX, TXT supported</p>
              </>
          }
        </div>
        <input ref={fileRef} type="file" accept=".pdf,.docx,.txt" style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files[0])} />
      </Card>

      {/* List */}
      <Card>
        <CardTitle>
          Documents
          <Button variant="ghost" size="sm" onClick={load}>↻ Refresh</Button>
        </CardTitle>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 32 }}><Spinner /></div>
        ) : docs.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--muted)', padding: 32 }}>No documents yet. Upload one above.</div>
        ) : (
          <table>
            <thead><tr>
              <th>Filename</th><th>Type</th><th>Chunks</th><th>Added</th><th></th>
            </tr></thead>
            <tbody>
              {docs.map(d => (
                <tr key={d.id}>
                  <td style={{ fontWeight: 500 }}>{d.filename}</td>
                  <td><Badge variant="accent">{d.file_type}</Badge></td>
                  <td>{d.chunk_count}</td>
                  <td style={{ color: 'var(--muted)' }}>{fmtDate(d.added_at)}</td>
                  <td><Button variant="danger" size="sm" onClick={() => handleDelete(d.id)}>Delete</Button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}

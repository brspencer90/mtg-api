import { useState } from 'react'
import type { CSSProperties } from 'react'
import { importApi } from '../api/client'
import type { MoxfieldCard, MoxfieldPreview } from '../api/client'

type CollectionType = 'deck' | 'commander' | 'pool' | 'storage'

const SECTION_LABELS: Record<string, string> = {
  commander: 'Commander',
  mainboard: 'Main Deck',
  sideboard: 'Sideboard',
  companion: 'Companion',
}

export function ImportDeck() {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState<MoxfieldPreview | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [collectionName, setCollectionName] = useState('')
  const [collectionType, setCollectionType] = useState<CollectionType>('deck')
  const [purchaseDate, setPurchaseDate] = useState('')

  const [saving, setSaving] = useState(false)
  const [progress, setProgress] = useState<{ done: number; total: number; card: string } | null>(null)
  const [saved, setSaved] = useState<{ id: number; count: number } | null>(null)

  async function handleFetch() {
    if (!url.trim()) return
    setLoading(true)
    setError(null)
    setPreview(null)
    setSaved(null)
    setProgress(null)
    try {
      const data = await importApi.preview(url.trim())
      setPreview(data)
      setCollectionName(data.deck_name)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to fetch deck'
      setError((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? msg)
    } finally {
      setLoading(false)
    }
  }

  async function handleSave() {
    if (!preview || !collectionName.trim()) return
    setSaving(true)
    setError(null)
    setProgress(null)
    try {
      const result = await importApi.save(
        {
          collection_name: collectionName.trim(),
          collection_type: collectionType,
          purchase_date: purchaseDate || undefined,
          cards: preview.cards,
        },
        ev => setProgress(ev),
      )
      setSaved({ id: result.collection_id, count: result.imported })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
      setProgress(null)
    }
  }

  const cardsBySection = preview
    ? preview.cards.reduce<Record<string, MoxfieldCard[]>>((acc, c) => {
        ;(acc[c.section] ??= []).push(c)
        return acc
      }, {})
    : {}

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '24px 16px' }}>
      <h2 style={{ marginBottom: 16 }}>Import Deck from Moxfield</h2>

      {/* URL input */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <input
          type="text"
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleFetch()}
          placeholder="https://www.moxfield.com/decks/..."
          style={inputStyle}
        />
        <button onClick={handleFetch} disabled={loading || !url.trim()} style={btnStyle}>
          {loading ? 'Fetching…' : 'Fetch'}
        </button>
      </div>

      {error && <p style={{ color: '#f87171', marginBottom: 12 }}>{error}</p>}

      {/* Preview */}
      {preview && (
        <>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h3 style={{ margin: '0 0 4px' }}>{preview.deck_name}</h3>
            <small style={{ color: '#94a3b8' }}>
              {preview.format} · {preview.card_count} cards
            </small>

            {Object.entries(cardsBySection).map(([section, cards]) => (
              <div key={section} style={{ marginTop: 12 }}>
                <p style={{ margin: '0 0 4px', fontWeight: 600, color: '#cbd5e1' }}>
                  {SECTION_LABELS[section] ?? section} ({cards.length})
                </p>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ color: '#64748b', textAlign: 'left' }}>
                      <th style={thStyle}>Qty</th>
                      <th style={thStyle}>Name</th>
                      <th style={thStyle}>Set</th>
                      <th style={thStyle}>#</th>
                      <th style={thStyle}>Foil</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cards.map((c, i) => (
                      <tr key={i} style={{ borderTop: '1px solid #334155' }}>
                        <td style={tdStyle}>{c.quantity}</td>
                        <td style={tdStyle}>{c.name}</td>
                        <td style={{ ...tdStyle, textTransform: 'uppercase' }}>{c.set_id}</td>
                        <td style={tdStyle}>{c.collector_no}</td>
                        <td style={tdStyle}>{c.foil ? '★' : c.etched ? 'E' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>

          {/* Save form */}
          {!saved ? (
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
              <h3 style={{ margin: '0 0 12px' }}>Save to Collection</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                <label style={labelStyle}>
                  Collection name
                  <input
                    value={collectionName}
                    onChange={e => setCollectionName(e.target.value)}
                    style={inputStyle}
                  />
                </label>
                <label style={labelStyle}>
                  Type
                  <select
                    value={collectionType}
                    onChange={e => setCollectionType(e.target.value as CollectionType)}
                    style={inputStyle}
                  >
                    <option value="deck">Deck</option>
                    <option value="commander">Commander</option>
                    <option value="pool">Sealed Pool</option>
                    <option value="storage">Storage</option>
                  </select>
                </label>
                <label style={labelStyle}>
                  Purchase date (optional)
                  <input
                    type="date"
                    value={purchaseDate}
                    onChange={e => setPurchaseDate(e.target.value)}
                    style={inputStyle}
                  />
                </label>
              </div>

              {/* Progress bar shown during save */}
              {saving && progress && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>
                    <span>{progress.card}</span>
                    <span>{progress.done} / {progress.total}</span>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, height: 6, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${Math.round((progress.done / progress.total) * 100)}%`,
                      background: '#7c3aed',
                      borderRadius: 4,
                      transition: 'width 0.15s ease',
                    }} />
                  </div>
                </div>
              )}
              {saving && !progress && (
                <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 12 }}>Starting…</p>
              )}

              <button
                onClick={handleSave}
                disabled={saving || !collectionName.trim()}
                style={{ ...btnStyle, background: saving ? '#334155' : '#2563eb' }}
              >
                {saving ? 'Saving…' : 'Save Deck'}
              </button>
            </div>
          ) : (
            <div style={{ background: '#14532d', borderRadius: 8, padding: 16 }}>
              <p style={{ margin: 0, color: '#86efac' }}>
                Saved as collection #{saved.id} — {saved.count} cards imported.
              </p>
              <button
                onClick={() => { setUrl(''); setPreview(null); setSaved(null) }}
                style={{ ...btnStyle, marginTop: 8, background: 'transparent', border: '1px solid #86efac', color: '#86efac' }}
              >
                Import another
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

const inputStyle: CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  background: '#0f172a',
  border: '1px solid #334155',
  borderRadius: 6,
  color: '#f1f5f9',
  fontSize: 14,
  boxSizing: 'border-box',
}

const btnStyle: CSSProperties = {
  padding: '8px 20px',
  background: '#475569',
  border: 'none',
  borderRadius: 6,
  color: '#f1f5f9',
  cursor: 'pointer',
  fontSize: 14,
  whiteSpace: 'nowrap',
}

const labelStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 4,
  color: '#94a3b8',
  fontSize: 13,
}

const thStyle: CSSProperties = { padding: '4px 8px', fontWeight: 500 }
const tdStyle: CSSProperties = { padding: '3px 8px', color: '#e2e8f0' }

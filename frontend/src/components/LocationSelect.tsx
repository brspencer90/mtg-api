import { useState } from 'react'
import type { CSSProperties } from 'react'
import { collectionApi } from '../api/client'
import type { Location } from '../api/client'

const LOC_TYPES = ['pool', 'deck', 'storage', 'trade', 'wishlist']
const TYPE_LABELS: Record<string, string> = {
  pool: 'Card Pool', deck: 'Deck', storage: 'Storage', trade: 'Trade', wishlist: 'Wishlist',
}

interface Props {
  locations: Location[]
  value: number | ''
  onChange: (id: number) => void
  onCreated: (loc: Location) => void
}

export function LocationSelect({ locations, value, onChange, onCreated }: Props) {
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [type, setType] = useState('pool')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function create() {
    if (!name.trim()) return
    setSaving(true); setError(null)
    try {
      const loc = await collectionApi.createLocation(name.trim(), type)
      onCreated(loc)
      onChange(loc.id)
      setCreating(false)
      setName('')
      setType('pool')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to create')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        <select
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          style={{ ...selectStyle, flex: 1 }}
        >
          {value === '' && <option value="">Select location…</option>}
          {locations.map(l => (
            <option key={l.id} value={l.id}>
              {l.name}
              {l.type !== 'storage' ? ` (${TYPE_LABELS[l.type] ?? l.type})` : ''}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => setCreating(c => !c)}
          title="Create new location"
          style={newBtnStyle}
        >
          {creating ? '✕' : '+ New'}
        </button>
      </div>

      {creating && (
        <div style={{ marginTop: 8, background: '#0f172a', borderRadius: 6, padding: 10, border: '1px solid #334155' }}>
          <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
            <input
              value={name}
              onChange={e => setName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && create()}
              placeholder="Location name…"
              autoFocus
              style={{ ...selectStyle, flex: 1 }}
            />
            <select value={type} onChange={e => setType(e.target.value)} style={{ ...selectStyle, width: 110 }}>
              {LOC_TYPES.map(t => <option key={t} value={t}>{TYPE_LABELS[t] ?? t}</option>)}
            </select>
          </div>
          {error && <p style={{ color: '#f87171', fontSize: 12, margin: '0 0 6px' }}>{error}</p>}
          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={create} disabled={saving || !name.trim()}
              style={{ ...actionBtnStyle, background: '#2563eb' }}>
              {saving ? 'Creating…' : 'Create'}
            </button>
            <button onClick={() => { setCreating(false); setName(''); setError(null) }}
              style={actionBtnStyle}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

const selectStyle: CSSProperties = {
  padding: '6px 10px', background: '#0f172a', border: '1px solid #334155',
  borderRadius: 6, color: '#f1f5f9', fontSize: 13, boxSizing: 'border-box',
}
const newBtnStyle: CSSProperties = {
  padding: '6px 10px', background: '#1e293b', border: '1px solid #334155',
  borderRadius: 6, color: '#94a3b8', cursor: 'pointer', fontSize: 12, whiteSpace: 'nowrap',
}
const actionBtnStyle: CSSProperties = {
  padding: '5px 12px', background: '#334155', border: 'none',
  borderRadius: 6, color: '#f1f5f9', cursor: 'pointer', fontSize: 12,
}

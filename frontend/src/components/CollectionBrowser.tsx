import { useEffect, useState, useCallback } from 'react'
import type { CSSProperties } from 'react'
import { collectionApi } from '../api/client'
import type { OwnedCopy, FilterOptions, Location } from '../api/client'

const CONDITIONS = ['NM', 'LP', 'MP', 'HP', 'DMG']

interface EditModalProps {
  copy: OwnedCopy
  onSave: (updated: OwnedCopy) => void
  onClose: () => void
}

function EditModal({ copy, onSave, onClose }: EditModalProps) {
  const [condition, setCondition]     = useState(copy.condition)
  const [foil, setFoil]               = useState(Boolean(copy.foil))
  const [etched, setEtched]           = useState(Boolean(copy.etched))
  const [purchaseDate, setPurchaseDate] = useState(copy.purchase_date ?? '')
  const [purchasePrice, setPurchasePrice] = useState(copy.purchase_price?.toString() ?? '')
  const [purchaseSource, setPurchaseSource] = useState(copy.purchase_source ?? '')
  const [notes, setNotes]             = useState(copy.notes ?? '')
  const [saving, setSaving]           = useState(false)
  const [error, setError]             = useState<string | null>(null)

  async function submit() {
    setSaving(true); setError(null)
    try {
      const updated = await collectionApi.updateCopy(copy.id, {
        condition,
        foil,
        etched,
        purchase_date: purchaseDate || undefined,
        purchase_price: purchasePrice ? Number(purchasePrice) : undefined,
        purchase_source: purchaseSource || undefined,
        notes: notes || undefined,
      })
      onSave(updated)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={{ ...modalStyle, width: 420 }} onClick={e => e.stopPropagation()}>
        <h3 style={{ margin: '0 0 4px' }}>{copy.name}</h3>
        <p style={{ color: '#64748b', fontSize: 12, margin: '0 0 16px' }}>
          {copy.set_id.toUpperCase()} #{copy.collector_no}
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 }}>
          <label style={labelStyle}>
            Condition
            <select value={condition} onChange={e => setCondition(e.target.value)} style={inputStyle}>
              {CONDITIONS.map(c => <option key={c}>{c}</option>)}
            </select>
          </label>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', paddingTop: 20 }}>
            <label style={checkStyle}>
              <input type="checkbox" checked={foil} onChange={e => setFoil(e.target.checked)} /> Foil
            </label>
            <label style={checkStyle}>
              <input type="checkbox" checked={etched} onChange={e => setEtched(e.target.checked)} /> Etched
            </label>
          </div>
          <label style={labelStyle}>
            Purchase date
            <input type="date" value={purchaseDate} onChange={e => setPurchaseDate(e.target.value)} style={inputStyle} />
          </label>
          <label style={labelStyle}>
            Price paid ($)
            <input type="number" step="0.01" value={purchasePrice}
              onChange={e => setPurchasePrice(e.target.value)} placeholder="0.00" style={inputStyle} />
          </label>
          <label style={{ ...labelStyle, gridColumn: '1 / -1' }}>
            Source
            <input value={purchaseSource} onChange={e => setPurchaseSource(e.target.value)}
              placeholder="LGS, TCGPlayer…" style={inputStyle} />
          </label>
          <label style={{ ...labelStyle, gridColumn: '1 / -1' }}>
            Notes
            <input value={notes} onChange={e => setNotes(e.target.value)}
              placeholder="Optional notes" style={inputStyle} />
          </label>
        </div>

        {error && <p style={{ color: '#f87171', fontSize: 13, marginBottom: 8 }}>{error}</p>}

        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={submit} disabled={saving} style={{ ...btnStyle, background: '#2563eb' }}>
            {saving ? 'Saving & refreshing price…' : 'Save'}
          </button>
          <button onClick={onClose} style={btnStyle}>Cancel</button>
        </div>
      </div>
    </div>
  )
}

const RARITIES = ['common', 'uncommon', 'rare', 'mythic']
const CARD_TYPES = ['Creature', 'Non-Creature', 'Land']
const COLORS = [
  { code: 'W', label: 'White' },
  { code: 'U', label: 'Blue' },
  { code: 'B', label: 'Black' },
  { code: 'R', label: 'Red' },
  { code: 'G', label: 'Green' },
]
const RARITY_COLORS: Record<string, string> = {
  common: '#94a3b8', uncommon: '#67e8f9', rare: '#fbbf24', mythic: '#f97316',
}

type SortKey = 'name' | 'set_id' | 'card_type' | 'rarity' | 'cmc' | 'price' | 'condition' | 'location'

const COLUMNS: { key: SortKey | null; label: string }[] = [
  { key: 'name',      label: 'Name' },
  { key: 'set_id',    label: 'Set' },
  { key: null,        label: '#' },
  { key: 'card_type', label: 'Type' },
  { key: 'rarity',    label: 'Rar.' },
  { key: 'cmc',       label: 'CMC' },
  { key: 'condition', label: 'Cond.' },
  { key: 'location',  label: 'Location' },
  { key: null,        label: 'Foil' },
  { key: 'price',     label: 'Price' },
  { key: null,        label: '' },
]

interface MoveModalProps {
  copy: OwnedCopy
  locations: Location[]
  onMove: (locationId: number, reason: string) => Promise<void>
  onClose: () => void
}

function MoveModal({ copy, locations, onMove, onClose }: MoveModalProps) {
  const [toId, setToId] = useState<number>(copy.location_id ?? locations[0]?.id ?? 0)
  const [reason, setReason] = useState('')
  const [saving, setSaving] = useState(false)

  async function submit() {
    setSaving(true)
    await onMove(toId, reason)
    setSaving(false)
  }

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={e => e.stopPropagation()}>
        <h3 style={{ margin: '0 0 8px' }}>Move — {copy.name}</h3>
        <p style={{ color: '#94a3b8', fontSize: 12, margin: '0 0 14px' }}>
          Currently: {copy.location_name ?? 'unplaced'}
        </p>
        <label style={labelStyle}>
          Move to
          <select value={toId} onChange={e => setToId(Number(e.target.value))} style={inputStyle}>
            {locations.map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
          </select>
        </label>
        <label style={{ ...labelStyle, marginTop: 10 }}>
          Reason (optional)
          <input value={reason} onChange={e => setReason(e.target.value)}
            placeholder="e.g. Added to deck" style={inputStyle} />
        </label>
        <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
          <button onClick={submit} disabled={saving}
            style={{ ...btnStyle, background: '#2563eb' }}>
            {saving ? 'Moving…' : 'Move'}
          </button>
          <button onClick={onClose} style={btnStyle}>Cancel</button>
        </div>
      </div>
    </div>
  )
}

export function CollectionBrowser() {
  const [copies, setCopies] = useState<OwnedCopy[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)

  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null)
  const [search, setSearch] = useState('')
  const [filterSet, setFilterSet] = useState('')
  const [filterLocation, setFilterLocation] = useState<number | ''>('')
  const [filterRarity, setFilterRarity] = useState('')
  const [filterType, setFilterType] = useState('')
  const [filterColor, setFilterColor] = useState('')

  const [sortBy, setSortBy] = useState<SortKey>('name')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')

  const [movingCopy, setMovingCopy]   = useState<OwnedCopy | null>(null)
  const [editingCopy, setEditingCopy] = useState<OwnedCopy | null>(null)

  const PAGE_SIZE = 50

  useEffect(() => {
    collectionApi.filterOptions().then(setFilterOptions)
  }, [])

  const load = useCallback(async (p: number, sb: SortKey = sortBy, sd: 'asc' | 'desc' = sortDir) => {
    setLoading(true)
    try {
      const res = await collectionApi.list({
        search: search || undefined,
        set_id: filterSet || undefined,
        location_id: filterLocation !== '' ? filterLocation : undefined,
        rarity: filterRarity || undefined,
        card_type: filterType || undefined,
        color: filterColor || undefined,
        sort_by: sb,
        sort_dir: sd,
        page: p,
        page_size: PAGE_SIZE,
      })
      setCopies(res.copies)
      setTotal(res.total)
      setPage(p)
    } finally {
      setLoading(false)
    }
  }, [search, filterSet, filterLocation, filterRarity, filterType, filterColor, sortBy, sortDir])

  useEffect(() => { load(1) }, [load])

  function handleSort(key: SortKey) {
    const newDir = sortBy === key && sortDir === 'asc' ? 'desc' : 'asc'
    setSortBy(key)
    setSortDir(newDir)
    load(1, key, newDir)
  }

  function handleEdit(updated: OwnedCopy) {
    setCopies(prev => prev.map(c => c.id === updated.id ? updated : c))
    setEditingCopy(null)
  }

  async function handleMove(locationId: number, reason: string) {
    if (!movingCopy) return
    const updated = await collectionApi.move(movingCopy.id, locationId, reason)
    setCopies(prev => prev.map(c => c.id === updated.id ? updated : c))
    setMovingCopy(null)
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  function sortIndicator(key: SortKey) {
    if (sortBy !== key) return <span style={{ color: '#334155', marginLeft: 3 }}>↕</span>
    return <span style={{ color: '#7c3aed', marginLeft: 3 }}>{sortDir === 'asc' ? '↑' : '↓'}</span>
  }

  return (
    <div style={{ padding: '16px 0' }}>
      {/* Filters */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14, alignItems: 'center' }}>
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && load(1)}
          placeholder="Search by name…"
          style={{ ...inputStyle, width: 190 }}
        />
        <select value={filterSet} onChange={e => setFilterSet(e.target.value)} style={{ ...inputStyle, width: 100 }}>
          <option value="">All sets</option>
          {filterOptions?.sets.map(s => <option key={s} value={s}>{s.toUpperCase()}</option>)}
        </select>
        <select
          value={filterLocation}
          onChange={e => setFilterLocation(e.target.value === '' ? '' : Number(e.target.value))}
          style={{ ...inputStyle, width: 170 }}
        >
          <option value="">All locations</option>
          {filterOptions?.locations.map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
        </select>
        <select value={filterRarity} onChange={e => setFilterRarity(e.target.value)} style={{ ...inputStyle, width: 115 }}>
          <option value="">All rarities</option>
          {RARITIES.map(r => <option key={r} value={r}>{r[0].toUpperCase() + r.slice(1)}</option>)}
        </select>
        <select value={filterType} onChange={e => setFilterType(e.target.value)} style={{ ...inputStyle, width: 130 }}>
          <option value="">All types</option>
          {CARD_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
        <div style={{ display: 'flex', gap: 4 }}>
          {COLORS.map(c => (
            <button key={c.code} title={c.label}
              onClick={() => setFilterColor(filterColor === c.code ? '' : c.code)}
              style={{
                width: 28, height: 28, borderRadius: 4, cursor: 'pointer',
                fontSize: 12, fontWeight: 700, border: '1px solid',
                borderColor: filterColor === c.code ? '#7c3aed' : '#334155',
                background: filterColor === c.code ? '#7c3aed' : '#1e293b',
                color: '#f1f5f9',
              }}
            >{c.code}</button>
          ))}
        </div>
        <button onClick={() => load(1)} style={btnStyle}>{loading ? '…' : 'Filter'}</button>
        <span style={{ color: '#64748b', fontSize: 12, marginLeft: 'auto' }}>
          {total.toLocaleString()} cards
        </span>
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto', borderRadius: 8, border: '1px solid #1e293b' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {COLUMNS.map(col => (
                <th
                  key={col.label}
                  onClick={col.key ? () => handleSort(col.key!) : undefined}
                  style={{
                    ...thStyle,
                    cursor: col.key ? 'pointer' : 'default',
                    userSelect: 'none',
                    color: sortBy === col.key ? '#c4b5fd' : '#94a3b8',
                  }}
                >
                  {col.label}
                  {col.key && sortIndicator(col.key)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {copies.map((copy, i) => (
              <tr key={copy.id}
                style={{ background: i % 2 === 0 ? 'transparent' : '#0f1f30', borderBottom: '1px solid #1e293b' }}>
                <td style={{ ...tdStyle, fontWeight: 500 }}>
                  <a
                    href={`https://scryfall.com/card/${copy.set_id}/${copy.collector_no}`}
                    target="_blank"
                    rel="noreferrer"
                    style={{ color: '#c4b5fd', textDecoration: 'none' }}
                    onMouseEnter={e => (e.currentTarget.style.textDecoration = 'underline')}
                    onMouseLeave={e => (e.currentTarget.style.textDecoration = 'none')}
                  >
                    {copy.name}
                  </a>
                </td>
                <td style={{ ...tdStyle, textTransform: 'uppercase', color: '#64748b' }}>{copy.set_id}</td>
                <td style={{ ...tdStyle, color: '#64748b' }}>{copy.collector_no}</td>
                <td style={tdStyle}>{copy.card_type}</td>
                <td style={{ ...tdStyle, color: RARITY_COLORS[copy.rarity] ?? '#94a3b8' }}>
                  {copy.rarity?.[0]?.toUpperCase() ?? '—'}
                </td>
                <td style={{ ...tdStyle, color: '#94a3b8' }}>{copy.cmc ?? '—'}</td>
                <td style={tdStyle}>{copy.condition}</td>
                <td style={{ ...tdStyle, color: '#94a3b8', maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {copy.location_name ?? '—'}
                </td>
                <td style={tdStyle}>{copy.foil ? '★' : copy.etched ? 'E' : '—'}</td>
                <td style={{ ...tdStyle, color: '#64748b' }}>
                  {copy.foil && copy.price_foil != null ? `$${copy.price_foil.toFixed(2)}`
                    : copy.price_std != null ? `$${copy.price_std.toFixed(2)}` : '—'}
                </td>
                <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>
                  <button onClick={() => setEditingCopy(copy)}
                    style={{ ...btnStyle, padding: '2px 9px', fontSize: 11, marginRight: 4 }}>
                    Edit
                  </button>
                  <button onClick={() => setMovingCopy(copy)}
                    style={{ ...btnStyle, padding: '2px 9px', fontSize: 11 }}>
                    Move
                  </button>
                </td>
              </tr>
            ))}
            {!loading && copies.length === 0 && (
              <tr><td colSpan={11} style={{ padding: 24, color: '#64748b', textAlign: 'center' }}>No cards found</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', gap: 8, marginTop: 14, justifyContent: 'center', alignItems: 'center' }}>
          <button onClick={() => load(page - 1)} disabled={page <= 1} style={btnStyle}>←</button>
          <span style={{ color: '#94a3b8', fontSize: 13 }}>Page {page} / {totalPages}</span>
          <button onClick={() => load(page + 1)} disabled={page >= totalPages} style={btnStyle}>→</button>
        </div>
      )}

      {editingCopy && (
        <EditModal
          copy={editingCopy}
          onSave={handleEdit}
          onClose={() => setEditingCopy(null)}
        />
      )}

      {movingCopy && filterOptions && (
        <MoveModal
          copy={movingCopy}
          locations={filterOptions.locations}
          onMove={handleMove}
          onClose={() => setMovingCopy(null)}
        />
      )}
    </div>
  )
}

const inputStyle: CSSProperties = {
  padding: '6px 10px', background: '#1e293b', border: '1px solid #334155',
  borderRadius: 6, color: '#f1f5f9', fontSize: 13,
}
const btnStyle: CSSProperties = {
  padding: '6px 12px', background: '#334155', border: 'none',
  borderRadius: 6, color: '#f1f5f9', cursor: 'pointer', fontSize: 13,
}
const thStyle: CSSProperties = { padding: '9px 10px', fontWeight: 500, textAlign: 'left', whiteSpace: 'nowrap' }
const tdStyle: CSSProperties = { padding: '6px 10px', color: '#e2e8f0' }
const labelStyle: CSSProperties = { display: 'flex', flexDirection: 'column', gap: 4, color: '#94a3b8', fontSize: 13 }
const checkStyle: CSSProperties = { display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', color: '#94a3b8', fontSize: 13 }
const overlayStyle: CSSProperties = {
  position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50,
}
const modalStyle: CSSProperties = {
  background: '#1e293b', borderRadius: 10, padding: 24, width: 360, border: '1px solid #334155',
}

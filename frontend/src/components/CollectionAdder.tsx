import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { collectionApi } from '../api/client'
import type { CardRecord, Location } from '../api/client'
import { LocationSelect } from './LocationSelect'

type Mode = 'name' | 'set-id' | 'bulk'
const CONDITIONS = ['NM', 'LP', 'MP', 'HP', 'DMG']

// ── Scryfall image URL from UUID ──────────────────────────────────────────────
function scryfallImg(id: string) {
  return `https://cards.scryfall.io/normal/front/${id[0]}/${id[1]}/${id}.jpg`
}

// ── Shared card preview card ──────────────────────────────────────────────────
function CardPreview({ card }: { card: CardRecord }) {
  const [imgErr, setImgErr] = useState(false)
  return (
    <div style={{ display: 'flex', gap: 14, background: '#0f172a', borderRadius: 8, padding: 12 }}>
      {!imgErr ? (
        <img
          src={scryfallImg(card.id)}
          alt={card.name}
          onError={() => setImgErr(true)}
          style={{ width: 120, borderRadius: 6, flexShrink: 0 }}
        />
      ) : (
        <div style={{ width: 120, height: 167, borderRadius: 6, background: '#1e293b', flexShrink: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: 11 }}>
          No image
        </div>
      )}
      <div style={{ flex: 1 }}>
        <p style={{ fontWeight: 600, fontSize: 15, margin: '0 0 4px' }}>{card.name}</p>
        <p style={{ color: '#94a3b8', fontSize: 12, margin: '0 0 2px' }}>
          {card.set_id.toUpperCase()} #{card.collector_no}
          <span style={{ marginLeft: 8, color: rarityColor(card.rarity) }}>
            {card.rarity[0].toUpperCase() + card.rarity.slice(1)}
          </span>
        </p>
        <p style={{ color: '#cbd5e1', fontSize: 12, margin: '0 0 2px' }}>{card.card_type}</p>
        {card.mana_cost && <p style={{ color: '#94a3b8', fontSize: 12, margin: '0 0 6px' }}>{card.mana_cost}</p>}
        <p style={{ color: '#64748b', fontSize: 12, margin: 0 }}>
          {card.price_std != null && `$${card.price_std.toFixed(2)} std`}
          {card.price_foil != null && ` · $${card.price_foil.toFixed(2)} foil`}
        </p>
      </div>
    </div>
  )
}

function rarityColor(r: string) {
  return ({ common: '#94a3b8', uncommon: '#67e8f9', rare: '#fbbf24', mythic: '#f97316' } as Record<string, string>)[r] ?? '#94a3b8'
}

// ── Shared "add details" form ─────────────────────────────────────────────────
interface DetailsFormProps {
  locations: Location[]
  locationId: number | ''
  setLocationId: (v: number) => void
  onLocationCreated: (loc: Location) => void
  condition: string
  setCondition: (v: string) => void
  foil: boolean
  setFoil: (v: boolean) => void
  etched: boolean
  setEtched: (v: boolean) => void
  purchaseDate: string
  setPurchaseDate: (v: string) => void
  purchasePrice: string
  setPurchasePrice: (v: string) => void
  purchaseSource: string
  setPurchaseSource: (v: string) => void
  onSubmit: () => void
  saving: boolean
  label?: string
}

function DetailsForm(p: DetailsFormProps) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginTop: 12 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <label style={labelStyle}>
          Physical location
          <LocationSelect
            locations={p.locations}
            value={p.locationId}
            onChange={p.setLocationId}
            onCreated={p.onLocationCreated}
          />
        </label>
        <label style={labelStyle}>
          Condition
          <select value={p.condition} onChange={e => p.setCondition(e.target.value)} style={inputStyle}>
            {CONDITIONS.map(c => <option key={c}>{c}</option>)}
          </select>
        </label>
        <label style={labelStyle}>
          Purchase date
          <input type="date" value={p.purchaseDate} onChange={e => p.setPurchaseDate(e.target.value)} style={inputStyle} />
        </label>
        <label style={labelStyle}>
          Price paid ($)
          <input type="number" step="0.01" value={p.purchasePrice}
            onChange={e => p.setPurchasePrice(e.target.value)} placeholder="0.00" style={inputStyle} />
        </label>
        <label style={{ ...labelStyle, gridColumn: '1 / -1' }}>
          Source
          <input value={p.purchaseSource} onChange={e => p.setPurchaseSource(e.target.value)}
            placeholder="LGS, TCGPlayer…" style={inputStyle} />
        </label>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <label style={checkStyle}>
            <input type="checkbox" checked={p.foil} onChange={e => p.setFoil(e.target.checked)} />
            Foil
          </label>
          <label style={checkStyle}>
            <input type="checkbox" checked={p.etched} onChange={e => p.setEtched(e.target.checked)} />
            Etched
          </label>
        </div>
      </div>
      <button onClick={p.onSubmit} disabled={p.saving || p.locationId === ''}
        style={{ ...btnStyle, marginTop: 12, background: '#2563eb' }}>
        {p.saving ? 'Adding…' : (p.label ?? 'Add to Collection')}
      </button>
    </div>
  )
}

// ── By-name mode ──────────────────────────────────────────────────────────────
interface Printing { id: string; set: string; set_name: string; collector_number: string; rarity: string }

function ByNameMode({ locations, onLocationCreated, onAdded }: {
  locations: Location[]; onLocationCreated: (l: Location) => void; onAdded: () => void
}) {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSug, setShowSug] = useState(false)
  const [printings, setPrintings] = useState<Printing[]>([])
  const [loadingPrintings, setLoadingPrintings] = useState(false)
  const [card, setCard] = useState<CardRecord | null>(null)
  const [loadingCard, setLoadingCard] = useState(false)

  const [locationId, setLocationId] = useState<number | ''>('')
  const [condition, setCondition] = useState('NM')
  const [foil, setFoil] = useState(false)
  const [etched, setEtched] = useState(false)
  const [purchaseDate, setPurchaseDate] = useState('')
  const [purchasePrice, setPurchasePrice] = useState('')
  const [purchaseSource, setPurchaseSource] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (locationId === '' && locations.length > 0) setLocationId(locations[0].id)
  }, [locations, locationId])

  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (debounce.current) clearTimeout(debounce.current)
    if (query.length < 2) { setSuggestions([]); return }
    debounce.current = setTimeout(async () => {
      const r = await fetch(`https://api.scryfall.com/cards/autocomplete?q=${encodeURIComponent(query)}`)
      const d = await r.json()
      setSuggestions(d.data ?? [])
      setShowSug(true)
    }, 250)
  }, [query])

  async function pickName(name: string) {
    setQuery(name); setShowSug(false); setCard(null); setPrintings([])
    setLoadingPrintings(true)
    const r = await fetch(`https://api.scryfall.com/cards/search?q=!"${encodeURIComponent(name)}"&unique=prints&order=released`)
    const d = await r.json()
    setPrintings(d.data ?? [])
    setLoadingPrintings(false)
  }

  async function pickPrinting(p: Printing) {
    setPrintings([]); setLoadingCard(true)
    try {
      setCard(await collectionApi.ensureCard({ scryfall_id: p.id }))
    } catch (e: unknown) { setError(e instanceof Error ? e.message : 'Not found') }
    setLoadingCard(false)
  }

  async function submit() {
    if (!card || locationId === '') return
    setSaving(true); setError(null)
    try {
      await collectionApi.addCopy({
        card_id: card.id, location_id: locationId, foil, etched, condition,
        purchase_date: purchaseDate || undefined,
        purchase_price: purchasePrice ? Number(purchasePrice) : undefined,
        purchase_source: purchaseSource || undefined,
      })
      setCard(null); setQuery(''); setFoil(false); setEtched(false)
      setPurchaseDate(''); setPurchasePrice(''); setPurchaseSource('')
      onAdded()
    } catch (e: unknown) { setError(e instanceof Error ? e.message : 'Failed') }
    setSaving(false)
  }

  return (
    <div>
      <div style={{ position: 'relative', marginBottom: 12 }}>
        <input value={query} onChange={e => { setQuery(e.target.value); setCard(null) }}
          onFocus={() => suggestions.length > 0 && setShowSug(true)}
          placeholder="Type a card name…" style={{ ...inputStyle, width: '100%' }} />
        {showSug && suggestions.length > 0 && (
          <ul style={dropdownStyle}>
            {suggestions.slice(0, 10).map(name => (
              <li key={name} onClick={() => pickName(name)} style={dropdownItemStyle}
                onMouseEnter={e => (e.currentTarget.style.background = '#334155')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                {name}
              </li>
            ))}
          </ul>
        )}
      </div>

      {loadingPrintings && <p style={mutedStyle}>Loading printings…</p>}
      {printings.length > 0 && (
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 12, marginBottom: 12 }}>
          <p style={{ ...mutedStyle, margin: '0 0 8px' }}>{printings.length} printings — pick one:</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
            {printings.map(p => (
              <button key={p.id} onClick={() => pickPrinting(p)}
                style={{ padding: '3px 9px', borderRadius: 4, cursor: 'pointer',
                  background: '#0f172a', border: '1px solid #334155', color: '#f1f5f9', fontSize: 12 }}>
                <b style={{ textTransform: 'uppercase' }}>{p.set}</b> #{p.collector_number}
                <span style={{ color: '#64748b', marginLeft: 4 }}>{p.rarity[0].toUpperCase()}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {loadingCard && <p style={mutedStyle}>Looking up card…</p>}
      {card && (
        <>
          <CardPreview card={card} />
          <DetailsForm locations={locations} locationId={locationId} setLocationId={setLocationId}
            onLocationCreated={onLocationCreated}
            condition={condition} setCondition={setCondition} foil={foil} setFoil={setFoil}
            etched={etched} setEtched={setEtched} purchaseDate={purchaseDate} setPurchaseDate={setPurchaseDate}
            purchasePrice={purchasePrice} setPurchasePrice={setPurchasePrice}
            purchaseSource={purchaseSource} setPurchaseSource={setPurchaseSource}
            onSubmit={submit} saving={saving} />
        </>
      )}
      {error && <p style={{ color: '#f87171', fontSize: 13, marginTop: 8 }}>{error}</p>}
    </div>
  )
}

// ── By set + ID mode ──────────────────────────────────────────────────────────
function BySetIdMode({ locations, onLocationCreated, onAdded }: {
  locations: Location[]; onLocationCreated: (l: Location) => void; onAdded: () => void
}) {
  const [setCode, setSetCode] = useState('')
  const [collNo, setCollNo] = useState('')
  const [card, setCard] = useState<CardRecord | null>(null)
  const [loading, setLoading] = useState(false)
  const [locationId, setLocationId] = useState<number | ''>('')
  const [condition, setCondition] = useState('NM')
  const [foil, setFoil] = useState(false)
  const [etched, setEtched] = useState(false)
  const [purchaseDate, setPurchaseDate] = useState('')
  const [purchasePrice, setPurchasePrice] = useState('')
  const [purchaseSource, setPurchaseSource] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (locationId === '' && locations.length > 0) setLocationId(locations[0].id)
  }, [locations, locationId])

  async function lookup() {
    if (!setCode.trim() || !collNo.trim()) return
    setLoading(true); setCard(null); setError(null)
    const rawNo = collNo.trim().replace(/[fe]$/i, '')
    const isFoil = /f$/i.test(collNo.trim())
    const isEtched = /e$/i.test(collNo.trim())
    try {
      const c = await collectionApi.ensureCard({ set_id: setCode.trim(), collector_no: rawNo })
      setCard(c)
      setFoil(isFoil)
      setEtched(isEtched)
    } catch (e: unknown) { setError(e instanceof Error ? e.message : 'Not found') }
    setLoading(false)
  }

  async function submit() {
    if (!card || locationId === '') return
    setSaving(true); setError(null)
    try {
      await collectionApi.addCopy({
        card_id: card.id, location_id: locationId, foil, etched, condition,
        purchase_date: purchaseDate || undefined,
        purchase_price: purchasePrice ? Number(purchasePrice) : undefined,
        purchase_source: purchaseSource || undefined,
      })
      setCard(null); setCollNo(''); setFoil(false); setEtched(false)
      setPurchaseDate(''); setPurchasePrice(''); setPurchaseSource('')
      onAdded()
    } catch (e: unknown) { setError(e instanceof Error ? e.message : 'Failed') }
    setSaving(false)
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'flex-end' }}>
        <label style={{ ...labelStyle, width: 100 }}>
          Set code
          <input value={setCode} onChange={e => setSetCode(e.target.value)}
            placeholder="e.g. dft" style={inputStyle} />
        </label>
        <label style={{ ...labelStyle, flex: 1 }}>
          Collector # (append f/e for foil/etched)
          <input value={collNo} onChange={e => setCollNo(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && lookup()}
            placeholder="e.g. 42f" style={inputStyle} />
        </label>
        <button onClick={lookup} disabled={loading || !setCode.trim() || !collNo.trim()}
          style={{ ...btnStyle, marginBottom: 1 }}>
          {loading ? '…' : 'Look up'}
        </button>
      </div>

      {card && (
        <>
          <CardPreview card={card} />
          <DetailsForm locations={locations} locationId={locationId} setLocationId={setLocationId}
            onLocationCreated={onLocationCreated}
            condition={condition} setCondition={setCondition} foil={foil} setFoil={setFoil}
            etched={etched} setEtched={setEtched} purchaseDate={purchaseDate} setPurchaseDate={setPurchaseDate}
            purchasePrice={purchasePrice} setPurchasePrice={setPurchasePrice}
            purchaseSource={purchaseSource} setPurchaseSource={setPurchaseSource}
            onSubmit={submit} saving={saving} />
        </>
      )}
      {error && <p style={{ color: '#f87171', fontSize: 13, marginTop: 8 }}>{error}</p>}
    </div>
  )
}

// ── Bulk mode ─────────────────────────────────────────────────────────────────
interface BulkCard { raw: string; no: string; foil: boolean; etched: boolean; card: CardRecord | null; err: boolean }

function BulkMode({ locations, onLocationCreated, onAdded }: {
  locations: Location[]; onLocationCreated: (l: Location) => void; onAdded: () => void
}) {
  const [setCode, setSetCode] = useState('')
  const [ids, setIds] = useState('')
  const [cards, setCards] = useState<BulkCard[]>([])
  const [previewing, setPreviewing] = useState(false)
  const [progress, setProgress] = useState(0)

  const [locationId, setLocationId] = useState<number | ''>('')
  const [condition, setCondition] = useState('NM')
  const [purchaseDate, setPurchaseDate] = useState('')
  const [purchaseSource, setPurchaseSource] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveProgress, setSaveProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (locationId === '' && locations.length > 0) setLocationId(locations[0].id)
  }, [locations, locationId])

  const lines = ids.split('\n').map(l => l.trim()).filter(Boolean)

  async function preview() {
    if (!setCode.trim() || lines.length === 0) return
    setPreviewing(true); setProgress(0); setError(null)
    const results: BulkCard[] = []
    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i]
      const no = raw.replace(/[fe]$/i, '')
      const isFoil = /f$/i.test(raw)
      const isEtched = /e$/i.test(raw)
      try {
        const c = await collectionApi.ensureCard({ set_id: setCode.trim(), collector_no: no })
        results.push({ raw, no, foil: isFoil, etched: isEtched, card: c, err: false })
      } catch {
        results.push({ raw, no, foil: isFoil, etched: isEtched, card: null, err: true })
      }
      setProgress(i + 1)
    }
    setCards(results)
    setPreviewing(false)
  }

  async function saveAll() {
    if (locationId === '') return
    setSaving(true); setSaveProgress(0); setError(null)
    const toAdd = cards.filter(c => c.card !== null)
    for (let i = 0; i < toAdd.length; i++) {
      const bc = toAdd[i]
      try {
        await collectionApi.addCopy({
          card_id: bc.card!.id, location_id: locationId,
          foil: bc.foil, etched: bc.etched, condition,
          purchase_date: purchaseDate || undefined,
          purchase_source: purchaseSource || undefined,
        })
      } catch { /* continue on error */ }
      setSaveProgress(i + 1)
    }
    setSaving(false)
    setCards([]); setIds('')
    onAdded()
  }

  const ready = cards.length > 0 && !previewing

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'flex-end' }}>
        <label style={{ ...labelStyle, width: 110 }}>
          Set code
          <input value={setCode} onChange={e => setSetCode(e.target.value)}
            placeholder="e.g. dft" style={inputStyle} />
        </label>
        <label style={{ ...labelStyle, flex: 1 }}>
          Collector numbers (one per line, f/e suffix for foil/etched)
          <textarea value={ids} onChange={e => setIds(e.target.value)} rows={6}
            placeholder={'42\n99f\n137e\n...'} style={{ ...inputStyle, resize: 'vertical', fontFamily: 'monospace' }} />
        </label>
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <button onClick={preview} disabled={previewing || !setCode.trim() || lines.length === 0} style={btnStyle}>
          {previewing ? `Previewing ${progress}/${lines.length}…` : `Preview ${lines.length} card${lines.length !== 1 ? 's' : ''}`}
        </button>
      </div>

      {cards.length > 0 && (
        <>
          <div style={{ background: '#1e293b', borderRadius: 8, overflow: 'hidden', marginBottom: 14 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#0f172a', color: '#64748b' }}>
                  <th style={thStyle}>#</th>
                  <th style={thStyle}>Name</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Rarity</th>
                  <th style={thStyle}>Foil</th>
                  <th style={thStyle}>Price</th>
                </tr>
              </thead>
              <tbody>
                {cards.map((bc, i) => (
                  <tr key={i} style={{ borderTop: '1px solid #1e293b', opacity: bc.err ? 0.4 : 1 }}>
                    <td style={tdStyle}>{bc.raw}</td>
                    <td style={{ ...tdStyle, color: bc.err ? '#f87171' : '#f1f5f9' }}>
                      {bc.err ? '(not found)' : bc.card?.name}
                    </td>
                    <td style={tdStyle}>{bc.card?.card_type ?? '—'}</td>
                    <td style={{ ...tdStyle, color: bc.card ? rarityColor(bc.card.rarity) : '#475569' }}>
                      {bc.card?.rarity?.[0]?.toUpperCase() ?? '—'}
                    </td>
                    <td style={tdStyle}>{bc.foil ? '★' : bc.etched ? 'E' : '—'}</td>
                    <td style={{ ...tdStyle, color: '#64748b' }}>
                      {bc.card?.price_std != null ? `$${bc.card.price_std.toFixed(2)}` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {ready && (
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 14 }}>
              <p style={{ ...mutedStyle, marginBottom: 10 }}>
                {cards.filter(c => !c.err).length} cards ready · {cards.filter(c => c.err).length} not found
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 12 }}>
                <label style={labelStyle}>
                  Physical location
                  <LocationSelect
                    locations={locations}
                    value={locationId}
                    onChange={setLocationId}
                    onCreated={onLocationCreated}
                  />
                </label>
                <label style={labelStyle}>
                  Condition
                  <select value={condition} onChange={e => setCondition(e.target.value)} style={inputStyle}>
                    {CONDITIONS.map(c => <option key={c}>{c}</option>)}
                  </select>
                </label>
                <label style={labelStyle}>
                  Purchase date
                  <input type="date" value={purchaseDate} onChange={e => setPurchaseDate(e.target.value)} style={inputStyle} />
                </label>
                <label style={labelStyle}>
                  Source
                  <input value={purchaseSource} onChange={e => setPurchaseSource(e.target.value)}
                    placeholder="LGS, TCGPlayer…" style={inputStyle} />
                </label>
              </div>

              {saving && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>
                    <span>Saving…</span><span>{saveProgress} / {cards.filter(c => !c.err).length}</span>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, height: 5 }}>
                    <div style={{
                      height: '100%', borderRadius: 4, background: '#7c3aed', transition: 'width 0.1s',
                      width: `${Math.round(saveProgress / cards.filter(c => !c.err).length * 100)}%`,
                    }} />
                  </div>
                </div>
              )}

              <button onClick={saveAll} disabled={saving || locationId === ''}
                style={{ ...btnStyle, background: '#2563eb' }}>
                {saving ? 'Adding…' : `Add ${cards.filter(c => !c.err).length} Cards`}
              </button>
            </div>
          )}
        </>
      )}
      {error && <p style={{ color: '#f87171', fontSize: 13, marginTop: 8 }}>{error}</p>}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export function CollectionAdder({ onAdded }: { onAdded?: () => void }) {
  const [mode, setMode] = useState<Mode>('name')
  const [locations, setLocations] = useState<Location[]>([])
  const [success, setSuccess] = useState<string | null>(null)

  useEffect(() => { collectionApi.locations().then(setLocations) }, [])

  function addLocation(loc: Location) {
    setLocations(prev => [...prev, loc])
  }

  function handleAdded() {
    setSuccess('Added to collection')
    setTimeout(() => setSuccess(null), 3000)
    onAdded?.()
  }

  const tabBtn = (m: Mode, label: string) => (
    <button key={m} onClick={() => setMode(m)} style={{
      padding: '7px 16px', background: 'transparent', border: 'none', cursor: 'pointer',
      borderBottom: mode === m ? '2px solid #7c3aed' : '2px solid transparent',
      color: mode === m ? '#f1f5f9' : '#64748b', fontSize: 13,
    }}>{label}</button>
  )

  return (
    <div style={{ maxWidth: 600 }}>
      <h3 style={{ margin: '0 0 12px' }}>Add Cards</h3>
      <div style={{ display: 'flex', borderBottom: '1px solid #1e293b', marginBottom: 16 }}>
        {tabBtn('name', 'By Name')}
        {tabBtn('set-id', 'By Set + ID')}
        {tabBtn('bulk', 'Bulk Add')}
      </div>

      {success && <p style={{ color: '#86efac', fontSize: 13, marginBottom: 10 }}>{success}</p>}

      {mode === 'name'   && <ByNameMode  locations={locations} onLocationCreated={addLocation} onAdded={handleAdded} />}
      {mode === 'set-id' && <BySetIdMode locations={locations} onLocationCreated={addLocation} onAdded={handleAdded} />}
      {mode === 'bulk'   && <BulkMode    locations={locations} onLocationCreated={addLocation} onAdded={handleAdded} />}
    </div>
  )
}

const inputStyle: CSSProperties = {
  padding: '6px 10px', background: '#0f172a', border: '1px solid #334155',
  borderRadius: 6, color: '#f1f5f9', fontSize: 13, width: '100%', boxSizing: 'border-box',
}
const btnStyle: CSSProperties = {
  padding: '7px 16px', background: '#334155', border: 'none',
  borderRadius: 6, color: '#f1f5f9', cursor: 'pointer', fontSize: 13, whiteSpace: 'nowrap',
}
const labelStyle: CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 4, color: '#94a3b8', fontSize: 13,
}
const checkStyle: CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', color: '#94a3b8', fontSize: 13,
}
const mutedStyle: CSSProperties = { color: '#64748b', fontSize: 13, margin: 0 }
const thStyle: CSSProperties = { padding: '7px 10px', fontWeight: 500, textAlign: 'left', color: '#94a3b8' }
const tdStyle: CSSProperties = { padding: '5px 10px', color: '#e2e8f0' }
const dropdownStyle: CSSProperties = {
  position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 20,
  background: '#1e293b', border: '1px solid #334155', borderRadius: 6,
  listStyle: 'none', margin: '2px 0 0', padding: 4, maxHeight: 240, overflowY: 'auto',
}
const dropdownItemStyle: CSSProperties = {
  padding: '7px 10px', borderRadius: 4, cursor: 'pointer', fontSize: 13, color: '#f1f5f9',
}

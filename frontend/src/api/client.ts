import axios from 'axios'

const api = axios.create({ baseURL: '/' })

export interface MoxfieldCard {
  name: string
  set_id: string
  collector_no: string
  scryfall_id: string
  quantity: number
  foil: number
  etched: number
  section: 'commander' | 'mainboard' | 'sideboard' | 'companion'
}

export interface MoxfieldPreview {
  deck_name: string
  format: string
  card_count: number
  cards: MoxfieldCard[]
}

export interface SaveDeckRequest {
  collection_name: string
  collection_type: string
  set_id?: string
  purchase_date?: string
  cards: MoxfieldCard[]
}

export interface SaveDeckResponse {
  collection_id: number
  imported: number
}

export type SaveProgressEvent =
  | { done: number; total: number; card: string; fetched: boolean }
  | { complete: true; collection_id: number; imported: number }
  | { error: string }

export const importApi = {
  preview: (url: string) =>
    api.post<MoxfieldPreview>('/api/import/moxfield/preview', { url }).then(r => r.data),

  /**
   * SSE-streaming save. Calls onProgress for each card, resolves with the
   * final complete event, or rejects on error.
   */
  save: (
    req: SaveDeckRequest,
    onProgress: (ev: { done: number; total: number; card: string }) => void,
  ): Promise<SaveDeckResponse> =>
    new Promise(async (resolve, reject) => {
      const resp = await fetch('/api/import/moxfield/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      })
      if (!resp.ok || !resp.body) {
        reject(new Error(`HTTP ${resp.status}`))
        return
      }
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const ev: SaveProgressEvent = JSON.parse(line.slice(6))
          if ('error' in ev) { reject(new Error(ev.error)); return }
          if ('complete' in ev) { resolve({ collection_id: ev.collection_id, imported: ev.imported }); return }
          onProgress({ done: ev.done, total: ev.total, card: ev.card })
        }
      }
    }),
}

export interface OwnedCopy {
  id: number
  card_id: string
  name: string
  set_id: string
  collector_no: string
  card_type: string
  colours: string[]
  rarity: string
  mana_cost: string
  cmc: number
  price_std: number | null
  price_foil: number | null
  foil: number
  etched: number
  condition: string
  purchase_date: string | null
  purchase_price: number | null
  purchase_source: string | null
  notes: string | null
  location_id: number | null
  location_name: string | null
  location_type: string | null
}

export interface CopiesPage {
  total: number
  page: number
  page_size: number
  copies: OwnedCopy[]
}

export interface Location {
  id: number
  name: string
  type: string
  archived: number
}

export interface FilterOptions {
  sets: string[]
  locations: Location[]
}

export interface CardRecord {
  id: string
  name: string
  set_id: string
  collector_no: string
  card_type: string
  colours: string[]
  rarity: string
  mana_cost: string
  cmc: number
  price_std: number | null
  price_foil: number | null
}

export const collectionApi = {
  list: (params: {
    search?: string; set_id?: string; location_id?: number
    rarity?: string; card_type?: string; color?: string
    sort_by?: string; sort_dir?: 'asc' | 'desc'
    page?: number; page_size?: number
  }) => api.get<CopiesPage>('/collection/copies', { params }).then(r => r.data),

  filterOptions: () => api.get<FilterOptions>('/collection/filter-options').then(r => r.data),

  ensureCard: (p: { scryfall_id?: string; set_id?: string; collector_no?: string }) =>
    api.get<CardRecord>('/collection/ensure-card', { params: p }).then(r => r.data),

  addCopy: (req: {
    card_id: string; location_id: number; foil?: boolean; etched?: boolean
    condition?: string; purchase_date?: string; purchase_price?: number
    purchase_source?: string; notes?: string
  }) => api.post<OwnedCopy>('/collection/copies', req).then(r => r.data),

  updateCopy: (copy_id: number, patch: {
    condition?: string; notes?: string; purchase_price?: number | null
    foil?: boolean; etched?: boolean; purchase_date?: string; purchase_source?: string
  }) => api.patch<OwnedCopy>(`/collection/copies/${copy_id}`, patch).then(r => r.data),

  move: (copy_id: number, to_location_id: number, reason?: string) =>
    api.post<OwnedCopy>('/collection/move', { copy_id, to_location_id, reason: reason ?? '' }).then(r => r.data),

  locations: () => api.get<Location[]>('/collection/locations').then(r => r.data),

  createLocation: (name: string, type: string) =>
    api.post<Location>('/collection/locations', { name, type }).then(r => r.data),
}

export interface SetInfo {
  id: string
  name: string
  released_at: string
}

export const setsApi = {
  list: () => api.get<SetInfo[]>('/api/sets').then(r => r.data),
}

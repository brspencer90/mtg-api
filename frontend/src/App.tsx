import { useState } from 'react'
import { ImportDeck } from './components/ImportDeck'
import { Collection } from './pages/Collection'
import './App.css'

type Tab = 'collection' | 'import'

const TABS: { id: Tab; label: string }[] = [
  { id: 'collection', label: 'Collection' },
  { id: 'import', label: 'Import Deck' },
]

function App() {
  const [tab, setTab] = useState<Tab>('collection')

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#f1f5f9', fontFamily: 'system-ui, sans-serif' }}>
      <header style={{ borderBottom: '1px solid #1e293b', padding: '0 24px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', alignItems: 'center', gap: 32, height: 52 }}>
          <span style={{ fontWeight: 700, fontSize: 16, color: '#7c3aed' }}>MTG Analysis</span>
          <nav style={{ display: 'flex', gap: 4 }}>
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                style={{
                  padding: '6px 14px',
                  background: 'transparent',
                  border: 'none',
                  borderBottom: tab === t.id ? '2px solid #7c3aed' : '2px solid transparent',
                  color: tab === t.id ? '#f1f5f9' : '#64748b',
                  cursor: 'pointer',
                  fontSize: 14,
                }}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main style={{ maxWidth: 1200, margin: '0 auto' }}>
        {tab === 'collection' && <Collection />}
        {tab === 'import' && <ImportDeck />}
      </main>
    </div>
  )
}

export default App

import { useState, useCallback } from 'react'
import { CollectionBrowser } from '../components/CollectionBrowser'
import { CollectionAdder } from '../components/CollectionAdder'

type SubTab = 'browse' | 'add'

export function Collection() {
  const [sub, setSub] = useState<SubTab>('browse')
  const [refreshKey, setRefreshKey] = useState(0)

  const handleAdded = useCallback(() => {
    setSub('browse')
    setRefreshKey(k => k + 1)
  }, [])

  return (
    <div style={{ padding: '0 16px' }}>
      <div style={{ display: 'flex', gap: 4, borderBottom: '1px solid #1e293b' }}>
        {(['browse', 'add'] as SubTab[]).map(t => (
          <button
            key={t}
            onClick={() => setSub(t)}
            style={{
              padding: '10px 18px',
              background: 'transparent',
              border: 'none',
              borderBottom: sub === t ? '2px solid #7c3aed' : '2px solid transparent',
              color: sub === t ? '#f1f5f9' : '#64748b',
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            {t === 'browse' ? 'Browse' : 'Add Card'}
          </button>
        ))}
      </div>

      {sub === 'browse' && <CollectionBrowser key={refreshKey} />}
      {sub === 'add' && (
        <div style={{ padding: '24px 0' }}>
          <CollectionAdder onAdded={handleAdded} />
        </div>
      )}
    </div>
  )
}

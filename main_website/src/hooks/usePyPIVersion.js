import { useState, useEffect } from 'react'

const CACHE_KEY = 'bb-pypi-version'
const CACHE_TTL = 1000 * 60 * 60 // 1 hour

export function usePyPIVersion(packageName = 'beyondbench') {
  const [version, setVersion] = useState(() => {
    try {
      const cached = JSON.parse(localStorage.getItem(CACHE_KEY))
      if (cached && Date.now() - cached.ts < CACHE_TTL) return cached.version
    } catch {}
    return null
  })

  useEffect(() => {
    let cancelled = false

    async function fetchVersion() {
      try {
        const res = await fetch(`https://pypi.org/pypi/${packageName}/json`)
        if (!res.ok) return
        const data = await res.json()
        const v = data.info?.version
        if (v && !cancelled) {
          setVersion(v)
          localStorage.setItem(CACHE_KEY, JSON.stringify({ version: v, ts: Date.now() }))
        }
      } catch {
        // silently fail, version stays as cached or null
      }
    }

    fetchVersion()
    return () => { cancelled = true }
  }, [packageName])

  return version
}

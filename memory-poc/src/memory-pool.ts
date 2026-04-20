import type { MemoryEntry } from './types'

export class MemoryPool {
  private entries = new Map<string, MemoryEntry>()
  private readonly maxEntries: number

  constructor(maxEntries: number) {
    this.maxEntries = maxEntries
  }

  upsert(entry: MemoryEntry): { evicted?: string } {
    const existing = this.entries.get(entry.keyword)
    if (existing) {
      this.entries.set(entry.keyword, {
        ...entry,
        useCount: existing.useCount + 1,
        lastUsed: Date.now(),
      })
      return {}
    }

    let evicted: string | undefined
    if (this.entries.size >= this.maxEntries) {
      evicted = this.evictOne()
    }

    this.entries.set(entry.keyword, entry)
    return { evicted }
  }

  retrieve(contextKeywords: string[], budget: number): MemoryEntry[] {
    const now = Date.now()
    const scored = Array.from(this.entries.values())
      .map(e => ({ entry: e, score: this.score(e, contextKeywords, now) }))
      .sort((a, b) => b.score - a.score)

    const result: MemoryEntry[] = []
    let tokenEstimate = 0

    for (const { entry } of scored) {
      const tokens = Math.ceil(entry.content.length / 4)
      if (tokenEstimate + tokens > budget) break
      result.push(entry)
      tokenEstimate += tokens
      this.entries.set(entry.keyword, { ...entry, lastUsed: now, useCount: entry.useCount + 1 })
    }

    return result
  }

  get size() { return this.entries.size }

  getAll(): MemoryEntry[] {
    return Array.from(this.entries.values())
  }

  private score(entry: MemoryEntry, contextKeywords: string[], now: number): number {
    const ageMinutes = (now - entry.lastUsed) / 60000
    const recency = 1 / (1 + ageMinutes * 0.1)
    const frequency = Math.log1p(entry.useCount) * 0.5
    const lowerCtx = contextKeywords.map(k => k.toLowerCase())
    const relevance = (
      lowerCtx.some(k => entry.keyword.toLowerCase().includes(k)) ||
      entry.relatedKeywords.some(rk => lowerCtx.some(k => rk.toLowerCase().includes(k)))
    ) ? 2 : 0
    return recency + frequency + relevance
  }

  private evictOne(): string {
    const now = Date.now()
    let lowestScore = Infinity
    let lowestKey = ''
    for (const [key, entry] of this.entries) {
      const s = this.score(entry, [], now)
      if (s < lowestScore) { lowestScore = s; lowestKey = key }
    }
    this.entries.delete(lowestKey)
    return lowestKey
  }
}

import type { MemoryEntry } from './types'

export class MemoryPool {
  private entries: MemoryEntry[] = []

  constructor(private readonly maxEntries: number = 100) {}

  get size() { return this.entries.length }

  private score(entry: MemoryEntry, keywords: string[]): number {
    const recency = 1 / (1 + (Date.now() - entry.lastUsed) / 60000)
    const freq = Math.log1p(entry.useCount) * 0.5
    const kws = [entry.keyword, ...entry.relatedKeywords].map(k => k.toLowerCase())
    const relevance = keywords.some(k => kws.some(e => e.includes(k) || k.includes(e))) ? 2 : 0
    return recency + freq + relevance
  }

  upsert(entry: MemoryEntry): string | null {
    const existing = this.entries.find(e => e.keyword.toLowerCase() === entry.keyword.toLowerCase())
    if (existing) {
      existing.content = entry.content
      existing.relatedKeywords = entry.relatedKeywords
      existing.lastUsed = Date.now()
      existing.useCount++
      return null
    }

    let evicted: string | null = null
    if (this.entries.length >= this.maxEntries) {
      const worst = this.entries.reduce((a, b) => this.score(a, []) < this.score(b, []) ? a : b)
      evicted = worst.keyword
      this.entries = this.entries.filter(e => e !== worst)
    }

    this.entries.push(entry)
    return evicted
  }

  retrieve(keywords: string[], budget: number = 800): MemoryEntry[] {
    const scored = [...this.entries].sort((a, b) => this.score(b, keywords) - this.score(a, keywords))
    const result: MemoryEntry[] = []
    let remaining = budget
    for (const entry of scored) {
      if (remaining <= 0) break
      entry.lastUsed = Date.now()
      entry.useCount++
      result.push(entry)
      remaining -= entry.content.length
    }
    return result
  }

  getAll(): MemoryEntry[] { return [...this.entries] }

  load(entries: MemoryEntry[]): void {
    for (const e of entries) this.upsert(e)
  }
}

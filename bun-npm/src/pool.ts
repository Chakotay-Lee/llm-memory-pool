import type { MemoryEntry } from './types'

const USE_COUNT_SHRINK_THRESHOLD = 10_000

export class MemoryPool {
  private entries: MemoryEntry[] = []
  private totalChars = 0
  private index = new Map<string, Set<string>>()   // keyword token → Set<entry.id>

  readonly maxEntries: number
  readonly maxPoolChars: number | null

  constructor(options: {
    maxEntries?: number
    contextWindow?: number
    poolRatio?: number
  } = {}) {
    this.maxEntries = options.maxEntries ?? 1000
    this.maxPoolChars = options.contextWindow
      ? Math.floor(options.contextWindow * (options.poolRatio ?? 0.15))
      : null
  }

  get size() { return this.entries.length }

  // ── Tokenisation ──────────────────────────────────────────────────────────

  private tokenize(text: string): string[] {
    return text.toLowerCase().split(/[\s\-_,。、!?！？\n]+/).filter(w => w.length > 1)
  }

  // ── Inverted index maintenance ────────────────────────────────────────────

  private indexEntry(entry: MemoryEntry): void {
    const kws = [entry.keyword, ...entry.relatedKeywords]
    for (const kw of kws) {
      for (const token of this.tokenize(kw)) {
        if (!this.index.has(token)) this.index.set(token, new Set())
        this.index.get(token)!.add(entry.id)
      }
    }
  }

  private deindexEntry(entry: MemoryEntry): void {
    const kws = [entry.keyword, ...entry.relatedKeywords]
    for (const kw of kws) {
      for (const token of this.tokenize(kw)) {
        const set = this.index.get(token)
        if (set) {
          set.delete(entry.id)
          if (set.size === 0) this.index.delete(token)
        }
      }
    }
  }

  // ── Scoring ───────────────────────────────────────────────────────────────

  private score(entry: MemoryEntry, queryLower: string, totalUseCount: number, maxUseCount: number): number {
    const recency = entry.useCount / (totalUseCount || 1)
    const freq = Math.log1p(entry.useCount) / Math.log1p(maxUseCount || 1)
    const kws = [entry.keyword, ...entry.relatedKeywords].map(k => k.toLowerCase())
    const relevance = kws.some(kw => queryLower.includes(kw)) ? 2 : 0
    return recency + freq + relevance
  }

  // ── Capacity helpers ──────────────────────────────────────────────────────

  private isFull(): boolean {
    if (this.maxPoolChars !== null) return this.totalChars >= this.maxPoolChars
    return this.entries.length >= this.maxEntries
  }

  private shrinkCounts(): void {
    if (this.entries.some(e => e.useCount > USE_COUNT_SHRINK_THRESHOLD)) {
      for (const e of this.entries) e.useCount = Math.max(1, Math.floor(e.useCount / 2))
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────

  upsert(entry: MemoryEntry): string | null {
    const existing = this.entries.find(e => e.keyword.toLowerCase() === entry.keyword.toLowerCase())
    if (existing) {
      this.deindexEntry(existing)
      this.totalChars -= existing.content.length
      existing.content = entry.content
      existing.relatedKeywords = entry.relatedKeywords
      existing.useCount++
      this.totalChars += existing.content.length
      this.indexEntry(existing)
      return null
    }

    let evicted: string | null = null
    if (this.isFull()) {
      this.shrinkCounts()
      const total = this.entries.reduce((s, e) => s + e.useCount, 0)
      const maxUc = Math.max(...this.entries.map(e => e.useCount))
      const worst = this.entries.reduce((a, b) =>
        this.score(a, '', total, maxUc) < this.score(b, '', total, maxUc) ? a : b
      )
      evicted = worst.keyword
      this.deindexEntry(worst)
      this.totalChars -= worst.content.length
      this.entries = this.entries.filter(e => e !== worst)
    }

    this.entries.push(entry)
    this.totalChars += entry.content.length
    this.indexEntry(entry)
    return evicted
  }

  retrieve(query: string, budget = 800): MemoryEntry[] {
    if (this.entries.length === 0) return []

    const queryLower = query.toLowerCase()
    const total = this.entries.reduce((s, e) => s + e.useCount, 0)
    const maxUc = Math.max(...this.entries.map(e => e.useCount))

    // Inverted index → candidate IDs
    const candidateIds = new Set<string>()
    for (const token of this.tokenize(queryLower)) {
      const ids = this.index.get(token)
      if (ids) for (const id of ids) candidateIds.add(id)
    }

    // Fall back to all entries if no index hits
    const candidates = candidateIds.size > 0
      ? this.entries.filter(e => candidateIds.has(e.id))
      : this.entries

    const scored = [...candidates].sort(
      (a, b) => this.score(b, queryLower, total, maxUc) - this.score(a, queryLower, total, maxUc)
    )

    const result: MemoryEntry[] = []
    let remaining = budget
    for (const entry of scored) {
      if (remaining <= 0) break
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

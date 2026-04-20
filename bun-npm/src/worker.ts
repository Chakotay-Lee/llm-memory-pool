import OpenAI from 'openai'
import { MemoryPool } from './pool'
import type { Turn, MemoryEntry, AppEvent } from './types'
import { chatComplete, parseJSON } from './llm'

export function extractKeywords(text: string): string[] {
  return text.toLowerCase().split(/[\s,。、!?！？\n]+/).filter(w => w.length > 2).slice(0, 10)
}

export class MemoryWorker {
  private pending: Turn[] = []
  private compressing = false

  constructor(
    private readonly pool: MemoryPool,
    private readonly client: OpenAI,
    private readonly memoryModel: string,
    private readonly minTurnsToCompress: number = 2,
    private readonly emit: (event: AppEvent) => void = () => {},
  ) {}

  addTurn(turn: Turn): void {
    this.pending.push(turn)
    if (this.pending.length >= this.minTurnsToCompress) {
      const batch = this.pending.splice(0)
      this.compress(batch)
    }
  }

  getMemory(userMessage: string, budget = 800): MemoryEntry[] {
    const keywords = extractKeywords(userMessage)
    const entries = this.pool.retrieve(keywords, budget)
    this.emit({ kind: 'memory_inject', ts: Date.now(), data: { count: entries.length, keywords, poolSize: this.pool.size } })
    return entries
  }

  private async compress(turns: Turn[]): Promise<void> {
    if (this.compressing) { this.pending.unshift(...turns); return }
    this.compressing = true
    this.emit({ kind: 'compress_start', ts: Date.now(), data: { turnCount: turns.length } })

    const turnText = turns.map(t => `${t.role === 'user' ? 'User' : 'AI'}: ${t.content}`).join('\n')

    try {
      const stratRaw = await chatComplete(this.client, this.memoryModel, [
        { role: 'system', content: 'You are a memory compression assistant. Output JSON only, no markdown.' },
        { role: 'user', content: `Analyze these conversation turns. Identify key topics and keywords.\n\n${turnText}\n\nPRIORITY: Always extract user-stated facts first (name, role, project, goals, preferences) before summarizing AI responses.\n\nOutput JSON:\n{"strategy":"one sentence","keywords":["kw1","kw2"]}` },
      ])

      let keywords: string[] = []
      let strategy = 'general compression'
      try {
        const parsed = parseJSON(stratRaw) as { strategy: string; keywords: string[] }
        keywords = parsed.keywords ?? []
        strategy = parsed.strategy ?? strategy
      } catch {
        keywords = extractKeywords(turnText)
      }

      const compRaw = await chatComplete(this.client, this.memoryModel, [
        { role: 'system', content: 'You are a memory compression assistant. Output JSON only, no markdown.' },
        { role: 'user', content: `Strategy: ${strategy}\nKeywords: ${keywords.join(', ')}\n\nConversation:\n${turnText}\n\nRules:\n1. FIRST extract user-stated facts: name, identity, role, project, preferences. Store as "User stated: ...".\n2. Then summarize key topics from AI responses (max 50 words each).\n3. Only store session-specific info, not general world knowledge.\n\nOutput JSON:\n{"entries":[{"keyword":"topic","content":"summary","relatedKeywords":["rel1"]}]}` },
      ])

      const parsed = parseJSON(compRaw) as { entries: Array<{ keyword: string; content: string; relatedKeywords?: string[] }> }
      const entries: MemoryEntry[] = (parsed.entries ?? []).map(e => ({
        id: crypto.randomUUID(),
        keyword: e.keyword,
        content: e.content,
        relatedKeywords: e.relatedKeywords ?? [],
        lastUsed: Date.now(),
        useCount: 1,
        createdAtTurn: turns[turns.length - 1].index,
      }))

      let evictedCount = 0
      for (const entry of entries) {
        const evicted = this.pool.upsert(entry)
        if (evicted) { evictedCount++; this.emit({ kind: 'pool_evict', ts: Date.now(), data: { keyword: evicted } }) }
      }

      this.emit({ kind: 'compress_done', ts: Date.now(), data: { entriesAdded: entries.length, evicted: evictedCount, poolSize: this.pool.size } })

    } catch (err) {
      this.emit({ kind: 'compress_error', ts: Date.now(), data: { error: String(err) } })
    } finally {
      this.compressing = false
      if (this.pending.length >= this.minTurnsToCompress) {
        const next = this.pending.splice(0)
        this.compress(next)
      }
    }
  }
}

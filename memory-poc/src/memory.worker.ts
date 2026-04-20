import { memoryConfig } from '../config/memory.config'
import { chatComplete } from './llm'
import { MemoryPool } from './memory-pool'
import type { Turn, MemoryEntry, ToWorker, FromWorker, AppEvent } from './types'

const POOL_FILE = './pool.json'
const pool = new MemoryPool(memoryConfig.pool.maxEntries)
const pendingTurns: Turn[] = []
let isCompressing = false

async function loadPool() {
  try {
    const file = Bun.file(POOL_FILE)
    if (await file.exists()) {
      const entries = await file.json() as MemoryEntry[]
      for (const entry of entries) pool.upsert(entry)
      emit({ kind: 'pool_state', ts: Date.now(), data: { restored: entries.length, poolSize: pool.size } })
    }
  } catch { /* fresh start */ }
}

async function savePool() {
  try {
    await Bun.write(POOL_FILE, JSON.stringify(pool.getAll(), null, 2))
  } catch { /* non-critical */ }
}

loadPool()

function emit(event: AppEvent) {
  self.postMessage({ type: 'EVENT', event } satisfies FromWorker)
}

function extractKeywords(text: string): string[] {
  return text.toLowerCase()
    .split(/[\s,。、!?！？\n]+/)
    .filter(w => w.length > 2)
    .slice(0, 10)
}

function parseJSON(text: string): unknown {
  const noThoughts = text.replace(/<thought>[\s\S]*?<\/thought>/g, '')
  const noFences = noThoughts.replace(/```[\w]*\n?|\n?```/g, '').trim()
  const match = noFences.match(/(\{[\s\S]*\}|\[[\s\S]*\])/)
  if (!match) throw new SyntaxError(`No JSON found in: ${noFences.slice(0, 100)}`)

  // Replace invalid JSON escape sequences (keep only: \" \\ \/ \b \f \n \r \t \uXXXX)
  const sanitized = match[0].replace(/\\([^"\\/bfnrtu])/g, '$1')
  return JSON.parse(sanitized)
}

async function compress(turns: Turn[]): Promise<void> {
  if (isCompressing || turns.length === 0) return
  isCompressing = true

  const turnText = turns.map(t => `${t.role === 'user' ? 'User' : 'AI'}: ${t.content}`).join('\n')

  emit({ kind: 'compress_start', ts: Date.now(), data: { turnCount: turns.length } })

  try {
    // Step 2a: determine compression strategy
    const strategyRaw = await chatComplete(memoryConfig.llm, [
      {
        role: 'system',
        content: 'You are a memory compression assistant. Output JSON only, no markdown.',
      },
      {
        role: 'user',
        content: `Analyze these conversation turns. Identify key topics and keywords to remember.\n\n${turnText}\n\nPRIORITY: Always extract user-stated facts first (name, role, project, goals, preferences) before summarizing AI responses.\n\nOutput JSON:\n{"strategy":"one sentence","keywords":["kw1","kw2","kw3"]}`,
      },
    ])

    let keywords: string[] = []
    let strategy = 'general compression'
    try {
      const parsed = parseJSON(strategyRaw) as { strategy: string; keywords: string[] }
      keywords = parsed.keywords ?? []
      strategy = parsed.strategy ?? strategy
    } catch {
      keywords = extractKeywords(turnText)
    }

    // Step 2b: compress into memory entries
    const compressionRaw = await chatComplete(memoryConfig.llm, [
      {
        role: 'system',
        content: 'You are a memory compression assistant. Output JSON only, no markdown.',
      },
      {
        role: 'user',
        content: `Strategy: ${strategy}\nFocus keywords: ${keywords.join(', ')}\n\nConversation:\n${turnText}\n\nRules:\n1. FIRST extract user-stated facts: name, identity, role, project, preferences. Store as "User stated: ...".\n2. Then summarize key topics from AI responses (max 50 words each).\n3. Only store session-specific info, not general world knowledge.\n\nOutput JSON:\n{"entries":[{"keyword":"topic","content":"concise summary","relatedKeywords":["rel1","rel2"]}]}`,
      },
    ])

    const parsed = parseJSON(compressionRaw) as { entries: Array<{ keyword: string; content: string; relatedKeywords?: string[] }> }
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
      const { evicted } = pool.upsert(entry)
      if (evicted) {
        evictedCount++
        emit({ kind: 'pool_evict', ts: Date.now(), data: { keyword: evicted } })
      }
    }

    emit({
      kind: 'compress_done',
      ts: Date.now(),
      data: {
        entriesAdded: entries.length,
        evicted: evictedCount,
        poolSize: pool.size,
        entries: entries.map(e => ({ keyword: e.keyword, preview: e.content.slice(0, 60) })),
      },
    })
    await savePool()
  } catch (err) {
    emit({ kind: 'compress_error', ts: Date.now(), data: { error: String(err) } })
  } finally {
    isCompressing = false
    // drain remaining pending turns if any accumulated during compression
    if (pendingTurns.length >= memoryConfig.pool.minTurnsToCompress) {
      const next = pendingTurns.splice(0)
      compress(next)
    }
  }
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const msg = event.data

  if (msg.type === 'ADD_TURN') {
    pendingTurns.push(msg.turn)
    if (pendingTurns.length >= memoryConfig.pool.minTurnsToCompress) {
      const toCompress = pendingTurns.splice(0)
      compress(toCompress)
    }
  }

  if (msg.type === 'DUMP_POOL') {
    self.postMessage({
      type: 'POOL_DUMP',
      requestId: msg.requestId,
      entries: pool.getAll(),
      pendingTurns: pendingTurns.length,
      isCompressing,
    } satisfies FromWorker)
  }

  if (msg.type === 'GET_MEMORY') {
    const keywords = extractKeywords(msg.recentContext)
    const entries = pool.retrieve(keywords, memoryConfig.pool.injectBudget)

    emit({
      kind: 'memory_inject',
      ts: Date.now(),
      data: {
        count: entries.length,
        poolSize: pool.size,
        keywords,
        entries: entries.map(e => ({ keyword: e.keyword, preview: e.content.slice(0, 60) })),
      },
    })

    self.postMessage({
      type: 'MEMORY_RESULT',
      requestId: msg.requestId,
      entries,
    } satisfies FromWorker)
  }
}

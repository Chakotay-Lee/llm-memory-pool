import OpenAI from 'openai'
import { MemoryPool } from './pool'
import { MemoryWorker, extractKeywords } from './worker'
import { streamChatComplete, chatComplete } from './llm'
import type { Turn, MemoryEntry, AppEvent, MemoryAgentOptions } from './types'

const DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai'
const DEFAULT_MAIN_MODEL = 'gemma-4-31b-it'
const DEFAULT_MEMORY_MODEL = 'gemma-4-26b-a4b-it'

export class MemoryAgent {
  private client: OpenAI
  private pool: MemoryPool
  private worker: MemoryWorker
  private turns: Turn[] = []
  private turnIndex = 0
  private poolFile?: string
  private emit: (event: AppEvent) => void

  readonly maxRawTurns: number
  readonly injectBudget: number
  readonly mainModel: string

  constructor(options: MemoryAgentOptions = {}) {
    this.mainModel = options.mainModel ?? DEFAULT_MAIN_MODEL
    this.maxRawTurns = options.maxRawTurns ?? 4
    this.injectBudget = options.injectBudget ?? 800
    this.poolFile = options.poolFile
    this.emit = options.onEvent ?? (() => {})

    this.client = new OpenAI({
      apiKey: options.apiKey ?? process.env.GEMINI_API_KEY ?? '',
      baseURL: options.baseURL ?? DEFAULT_BASE_URL,
    })

    this.pool = new MemoryPool({
      maxEntries: options.maxPoolEntries ?? 1000,
      contextWindow: options.contextWindow,
      poolRatio: options.poolRatio,
    })
    this.worker = new MemoryWorker(
      this.pool,
      this.client,
      options.memoryModel ?? DEFAULT_MEMORY_MODEL,
      options.minTurnsToCompress ?? 2,
      (event) => {
        this.emit(event)
        if (event.kind === 'compress_done' && this.poolFile) {
          this.savePool()
        }
      },
    )
  }

  async loadPool(): Promise<void> {
    if (!this.poolFile) return
    try {
      const file = Bun.file(this.poolFile)
      if (await file.exists()) {
        const entries = await file.json() as MemoryEntry[]
        this.pool.load(entries)
        this.emit({ kind: 'pool_state', ts: Date.now(), data: { restored: entries.length, source: this.poolFile } })
      }
    } catch (err) {
      this.emit({ kind: 'compress_error', ts: Date.now(), data: { error: `pool load: ${err}` } })
    }
  }

  private async savePool(): Promise<void> {
    if (!this.poolFile) return
    try { await Bun.write(this.poolFile, JSON.stringify(this.pool.getAll(), null, 2)) } catch { /* non-critical */ }
  }

  private buildMessages(userMessage: string, memoryText: string): OpenAI.Chat.ChatCompletionMessageParam[] {
    const recent = this.turns.slice(-this.maxRawTurns)
    const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
    ]
    for (const t of recent.slice(0, -1)) messages.push({ role: t.role, content: t.content })
    const content = memoryText ? `## Conversation Memory\n${memoryText}\n\n---\n\n${userMessage}` : userMessage
    messages.push({ role: 'user', content })
    return messages
  }

  async *chatStream(userMessage: string): AsyncGenerator<string> {
    const userTurn: Turn = { role: 'user', content: userMessage, index: this.turnIndex++ }
    this.turns.push(userTurn)
    this.emit({ kind: 'turn_user', ts: Date.now(), data: { index: userTurn.index, chars: userMessage.length } })

    const memEntries = this.worker.getMemory(userMessage, this.injectBudget)
    const memoryText = memEntries.map(e => `[${e.keyword}] ${e.content}`).join('\n')

    const messages = this.buildMessages(userMessage, memoryText)
    this.emit({ kind: 'context_built', ts: Date.now(), data: {
      rawTurns: Math.min(this.turns.length, this.maxRawTurns),
      memoryEntries: memEntries.length,
      totalMessages: messages.length,
    }})

    let fullText = ''
    for await (const chunk of streamChatComplete(this.client, this.mainModel, messages)) {
      fullText += chunk
      yield chunk
    }

    const asstTurn: Turn = { role: 'assistant', content: fullText, index: this.turnIndex++ }
    this.turns.push(asstTurn)
    this.emit({ kind: 'turn_assistant', ts: Date.now(), data: { index: asstTurn.index, chars: fullText.length } })

    this.worker.addTurn(userTurn)
    this.worker.addTurn(asstTurn)
  }

  async chat(userMessage: string): Promise<string> {
    let result = ''
    for await (const chunk of this.chatStream(userMessage)) result += chunk
    return result
  }

  getPoolEntries(): MemoryEntry[] { return this.pool.getAll() }
  getPoolSize(): number { return this.pool.size }
}

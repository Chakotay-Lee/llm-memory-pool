export interface Turn {
  role: 'user' | 'assistant'
  content: string
  index: number
}

export interface MemoryEntry {
  id: string
  keyword: string
  content: string
  relatedKeywords: string[]
  lastUsed: number
  useCount: number
  createdAtTurn: number
}

export type EventKind =
  | 'turn_user' | 'turn_assistant'
  | 'compress_start' | 'compress_done' | 'compress_error'
  | 'memory_inject' | 'pool_evict' | 'pool_state'
  | 'context_built' | 'ping'

export interface AppEvent {
  kind: EventKind
  ts: number
  data: Record<string, unknown>
}

export interface MemoryAgentOptions {
  apiKey?: string
  baseURL?: string
  mainModel?: string
  memoryModel?: string
  maxRawTurns?: number
  minTurnsToCompress?: number
  maxPoolEntries?: number
  injectBudget?: number
  poolFile?: string
  onEvent?: (event: AppEvent) => void
}

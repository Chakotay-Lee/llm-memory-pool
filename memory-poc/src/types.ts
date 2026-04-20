export interface Turn {
  index: number
  role: 'user' | 'assistant'
  content: string
  timestamp: number
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
  | 'turn_user'
  | 'turn_assistant'
  | 'compress_start'
  | 'compress_done'
  | 'compress_error'
  | 'memory_inject'
  | 'pool_evict'
  | 'pool_state'
  | 'context_built'
  | 'prompt_sent'

export interface AppEvent {
  kind: EventKind
  ts: number
  data: Record<string, unknown>
}

export type ToWorker =
  | { type: 'ADD_TURN'; turn: Turn }
  | { type: 'GET_MEMORY'; requestId: string; recentContext: string }
  | { type: 'DUMP_POOL'; requestId: string }

export type FromWorker =
  | { type: 'MEMORY_RESULT'; requestId: string; entries: MemoryEntry[] }
  | { type: 'POOL_DUMP'; requestId: string; entries: MemoryEntry[]; pendingTurns: number; isCompressing: boolean }
  | { type: 'EVENT'; event: AppEvent }

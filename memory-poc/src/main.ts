import { mainConfig } from '../config/main.config'
import { streamChatComplete } from './llm'
import { logEvent } from './logger'
import type { Turn, MemoryEntry, ToWorker, FromWorker, AppEvent } from './types'

// SSE clients
const sseClients = new Set<ReadableStreamDefaultController<Uint8Array>>()
const encoder = new TextEncoder()

function broadcastEvent(event: AppEvent) {
  const data = encoder.encode(`data: ${JSON.stringify(event)}\n\n`)
  for (const ctrl of sseClients) {
    try { ctrl.enqueue(data) } catch { sseClients.delete(ctrl) }
  }
  logEvent(event)
}

// Memory worker
const worker = new Worker(new URL('./memory.worker.ts', import.meta.url))
const pendingMemoryRequests = new Map<string, (entries: MemoryEntry[]) => void>()
const pendingPoolDumps = new Map<string, (result: { entries: MemoryEntry[]; pendingTurns: number; isCompressing: boolean }) => void>()

worker.onmessage = (event: MessageEvent<FromWorker>) => {
  const msg = event.data
  if (msg.type === 'MEMORY_RESULT') {
    pendingMemoryRequests.get(msg.requestId)?.(msg.entries)
    pendingMemoryRequests.delete(msg.requestId)
  } else if (msg.type === 'POOL_DUMP') {
    pendingPoolDumps.get(msg.requestId)?.({ entries: msg.entries, pendingTurns: msg.pendingTurns, isCompressing: msg.isCompressing })
    pendingPoolDumps.delete(msg.requestId)
  } else if (msg.type === 'EVENT') {
    broadcastEvent(msg.event)
  }
}

function dumpPool() {
  return new Promise<{ entries: MemoryEntry[]; pendingTurns: number; isCompressing: boolean }>(resolve => {
    const requestId = crypto.randomUUID()
    pendingPoolDumps.set(requestId, resolve)
    worker.postMessage({ type: 'DUMP_POOL', requestId } satisfies ToWorker)
    setTimeout(() => { pendingPoolDumps.delete(requestId); resolve({ entries: [], pendingTurns: -1, isCompressing: false }) }, 1000)
  })
}

// Conversation state
const turns: Turn[] = []
let turnIndex = 0

function addTurn(turn: Turn) {
  turns.push(turn)
  worker.postMessage({ type: 'ADD_TURN', turn } satisfies ToWorker)
}

function getMemory(recentContext: string): Promise<MemoryEntry[]> {
  return new Promise(resolve => {
    const requestId = crypto.randomUUID()
    pendingMemoryRequests.set(requestId, resolve)
    worker.postMessage({ type: 'GET_MEMORY', requestId, recentContext } satisfies ToWorker)
    // Graceful degradation: if worker is busy, return empty after 2s
    setTimeout(() => {
      if (pendingMemoryRequests.has(requestId)) {
        pendingMemoryRequests.delete(requestId)
        resolve([])
      }
    }, 2000)
  })
}

async function chat(userMessage: string, onChunk: (chunk: string) => void): Promise<string> {
  const userTurn: Turn = { index: turnIndex++, role: 'user', content: userMessage, timestamp: Date.now() }
  addTurn(userTurn)
  broadcastEvent({ kind: 'turn_user', ts: Date.now(), data: { content: userMessage } })

  // Pull compressed memory before generating response
  const recentContext = turns.slice(-5).map(t => t.content).join(' ')
  const memoryEntries = await getMemory(recentContext)

  // Build context: stable system prompt + recent raw turns + memory prepended to current user message
  // Memory is NOT in system prompt — keeps system prefix stable for KV cache hits.
  const recentTurns = turns.slice(-mainConfig.context.maxRawTurns)
  const droppedTurns = Math.max(0, turns.length - mainConfig.context.maxRawTurns - 1)
  const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = []

  // Stable system prompt — never changes, maximises KV cache prefix
  messages.push({ role: 'system', content: 'You are a helpful assistant.' })

  // Historical raw turns (unchanged from previous turn — KV cache hits here)
  const historyTurns = recentTurns.slice(0, -1)   // everything except the current user turn
  messages.push(...historyTurns.map(t => ({ role: t.role, content: t.content })))

  // Current user message with memory prepended (only this final message is new each turn)
  const memoryText = memoryEntries.length > 0
    ? memoryEntries.map(e => `[${e.keyword}]: ${e.content}`).join('\n')
    : null

  const currentUserContent = memoryText
    ? `## Conversation Memory\n${memoryText}\n\n---\n\n${userMessage}`
    : userMessage

  messages.push({ role: 'user', content: currentUserContent })

  // Emit what was actually sent to the LLM for verification
  broadcastEvent({
    kind: 'context_built',
    ts: Date.now(),
    data: {
      rawTurns: recentTurns.length,
      rawTurnIndices: recentTurns.map(t => t.index),
      droppedTurns,
      memoryEntries: memoryEntries.length,
      memoryFromPool: memoryEntries.map(e => ({
        keyword: e.keyword,
        preview: e.content.slice(0, 80),
        useCount: e.useCount,
        createdAtTurn: e.createdAtTurn,
      })),
    },
  })

  broadcastEvent({
    kind: 'prompt_sent',
    ts: Date.now(),
    data: { messages },
  })

  const assistantText = await streamChatComplete(mainConfig.llm, messages, onChunk)

  const assistantTurn: Turn = { index: turnIndex++, role: 'assistant', content: assistantText, timestamp: Date.now() }
  addTurn(assistantTurn)
  broadcastEvent({ kind: 'turn_assistant', ts: Date.now(), data: { content: assistantText } })

  return assistantText
}

// HTTP server
Bun.serve({
  port: mainConfig.server.port,
  idleTimeout: 120,

  async fetch(req) {
    const url = new URL(req.url)

    // Debug: raw pool.json file on disk
    if (req.method === 'GET' && url.pathname === '/debug/pool-file') {
      const file = Bun.file('./pool.json')
      if (await file.exists()) {
        return new Response(file, { headers: { 'Content-Type': 'application/json' } })
      }
      return new Response('[]', { headers: { 'Content-Type': 'application/json' } })
    }

    // Debug: pool state
    if (req.method === 'GET' && url.pathname === '/debug/pool') {
      const pool = await dumpPool()
      return new Response(JSON.stringify({
        poolSize: pool.entries.length,
        isCompressing: pool.isCompressing,
        pendingTurns: pool.pendingTurns,
        entries: pool.entries.map(e => ({
          keyword: e.keyword,
          content: e.content,
          createdAtTurn: e.createdAtTurn,
          useCount: e.useCount,
          lastUsed: new Date(e.lastUsed).toISOString(),
          relatedKeywords: e.relatedKeywords,
        })),
      }, null, 2), { headers: { 'Content-Type': 'application/json' } })
    }

    // Debug: raw turns in memory
    if (req.method === 'GET' && url.pathname === '/debug/turns') {
      return new Response(JSON.stringify({
        totalTurns: turns.length,
        maxRawTurns: mainConfig.context.maxRawTurns,
        inWindow: turns.slice(-mainConfig.context.maxRawTurns).map(t => ({
          index: t.index, role: t.role, preview: t.content.slice(0, 60),
        })),
        dropped: Math.max(0, turns.length - mainConfig.context.maxRawTurns),
      }, null, 2), { headers: { 'Content-Type': 'application/json' } })
    }

    // Serve dashboard
    if (req.method === 'GET' && url.pathname === '/') {
      return new Response(Bun.file(new URL('../public/index.html', import.meta.url)))
    }

    // SSE event stream
    if (req.method === 'GET' && url.pathname === '/events') {
      let ctrl: ReadableStreamDefaultController<Uint8Array>
      const stream = new ReadableStream<Uint8Array>({
        start(c) {
          ctrl = c
          sseClients.add(ctrl)
          // Send a heartbeat so the browser knows the connection is alive
          ctrl.enqueue(encoder.encode(': connected\n\n'))
        },
        cancel() { sseClients.delete(ctrl) },
      })
      req.signal.addEventListener('abort', () => sseClients.delete(ctrl))
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
        },
      })
    }

    // Chat endpoint — streams AI response chunks as SSE
    if (req.method === 'POST' && url.pathname === '/chat') {
      const body = await req.json() as { message: string }
      if (!body.message?.trim()) {
        return new Response(JSON.stringify({ error: 'empty message' }), { status: 400 })
      }

      let ctrl: ReadableStreamDefaultController<Uint8Array>
      const stream = new ReadableStream<Uint8Array>({
        async start(c) {
          ctrl = c
          try {
            await chat(body.message, (chunk) => {
              ctrl.enqueue(encoder.encode(`data: ${JSON.stringify({ chunk })}\n\n`))
            })
            ctrl.enqueue(encoder.encode(`data: ${JSON.stringify({ done: true })}\n\n`))
          } catch (err) {
            ctrl.enqueue(encoder.encode(`data: ${JSON.stringify({ error: String(err) })}\n\n`))
          } finally {
            ctrl.close()
          }
        },
      })

      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Access-Control-Allow-Origin': '*',
        },
      })
    }

    return new Response('Not Found', { status: 404 })
  },
})

console.log(`Memory PoC running at http://localhost:${mainConfig.server.port}`)

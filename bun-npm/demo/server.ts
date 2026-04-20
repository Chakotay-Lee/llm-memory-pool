import { MemoryAgent } from '../src'
import type { AppEvent } from '../src'

const PORT = 3001
const POOL_FILE = './demo-pool.json'

const sseClients = new Set<(data: string) => void>()

function broadcast(event: AppEvent) {
  const data = `data: ${JSON.stringify(event)}\n\n`
  for (const send of sseClients) send(data)
}

const agent = new MemoryAgent({
  apiKey: process.env.GEMINI_API_KEY,
  poolFile: POOL_FILE,
  onEvent: broadcast,
})

await agent.loadPool()

const publicDir = `${import.meta.dir}/public`

Bun.serve({
  port: PORT,
  idleTimeout: 120,
  async fetch(req) {
    const url = new URL(req.url)

    // Serve static files
    if (url.pathname === '/' || url.pathname === '/index.html') {
      return new Response(Bun.file(`${publicDir}/index.html`), {
        headers: { 'Content-Type': 'text/html; charset=utf-8' },
      })
    }

    // SSE events stream
    if (url.pathname === '/events') {
      let send: (data: string) => void
      const stream = new ReadableStream({
        start(controller) {
          send = (data: string) => controller.enqueue(new TextEncoder().encode(data))
          sseClients.add(send)
          controller.enqueue(new TextEncoder().encode(': connected\n\n'))
        },
        cancel() { sseClients.delete(send) },
      })
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      })
    }

    // Chat endpoint — streaming
    if (url.pathname === '/chat' && req.method === 'POST') {
      const { message } = await req.json() as { message: string }
      const stream = new ReadableStream({
        async start(controller) {
          for await (const chunk of agent.chatStream(message)) {
            controller.enqueue(new TextEncoder().encode(chunk))
          }
          controller.close()
        },
      })
      return new Response(stream, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } })
    }

    // Debug endpoints
    if (url.pathname === '/debug/pool') {
      return Response.json({ size: agent.getPoolSize(), entries: agent.getPoolEntries() })
    }

    return new Response('Not found', { status: 404 })
  },
})

console.log(`Memory Pool demo running at http://localhost:${PORT}`)

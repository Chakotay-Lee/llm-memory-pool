interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

interface LLMConfig {
  baseURL: string
  apiKey: string
  model: string
  maxTokens: number
  temperature: number
}

function stripThoughts(text: string): string {
  return text.replace(/<thought>[\s\S]*?<\/thought>/g, '').trim()
}

export async function chatComplete(config: LLMConfig, messages: ChatMessage[]): Promise<string> {
  const resp = await fetch(`${config.baseURL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      messages,
      max_tokens: config.maxTokens,
      temperature: config.temperature,
    }),
  })

  if (!resp.ok) {
    const err = await resp.text()
    throw new Error(`LLM ${resp.status}: ${err.slice(0, 200)}`)
  }

  const data = await resp.json() as { choices: Array<{ message: { content: string } }> }
  return stripThoughts(data.choices[0].message.content)
}

export async function streamChatComplete(
  config: LLMConfig,
  messages: ChatMessage[],
  onChunk: (chunk: string) => void,
): Promise<string> {
  const resp = await fetch(`${config.baseURL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      messages,
      max_tokens: config.maxTokens,
      temperature: config.temperature,
      stream: true,
    }),
  })

  if (!resp.ok) {
    const err = await resp.text()
    throw new Error(`LLM ${resp.status}: ${err.slice(0, 200)}`)
  }

  const reader = resp.body!.getReader()
  const decoder = new TextDecoder()
  let fullText = ''
  let buffer = ''
  let inThought = false

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const lines = decoder.decode(value).split('\n').filter(l => l.startsWith('data: '))
    for (const line of lines) {
      const json = line.slice(6).trim()
      if (json === '[DONE]') continue
      try {
        const d = JSON.parse(json) as { choices: Array<{ delta: { content?: string } }> }
        const raw = d.choices[0]?.delta?.content ?? ''
        if (!raw) continue

        buffer += raw

        // Process buffer: emit everything outside <thought>...</thought> blocks.
        // Tags may span multiple chunks so we loop until buffer is exhausted.
        let output = ''
        while (buffer.length > 0) {
          if (!inThought) {
            const tagStart = buffer.indexOf('<thought>')
            if (tagStart === -1) {
              // No opening tag — but a partial '<' at the end might be one starting;
              // hold the last 9 chars so the next chunk can complete the tag check.
              const ltIdx = buffer.lastIndexOf('<')
              if (ltIdx !== -1 && ltIdx > buffer.length - 9) {
                output += buffer.slice(0, ltIdx)
                buffer = buffer.slice(ltIdx)
              } else {
                output += buffer
                buffer = ''
              }
              break
            }
            output += buffer.slice(0, tagStart)
            buffer = buffer.slice(tagStart + 9)   // skip past <thought>
            inThought = true
          } else {
            const tagEnd = buffer.indexOf('</thought>')
            if (tagEnd === -1) {
              // Still inside thought — discard content but keep last 10 chars
              // in case </thought> is split across chunks
              buffer = buffer.slice(Math.max(0, buffer.length - 10))
              break
            }
            buffer = buffer.slice(tagEnd + 10)    // skip past </thought>
            inThought = false
          }
        }

        if (output) {
          fullText += output
          onChunk(output)
        }
      } catch { /* skip malformed chunk */ }
    }
  }

  return fullText
}

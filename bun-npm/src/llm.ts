import OpenAI from 'openai'

export function stripThoughts(text: string): string {
  return text.replace(/<thought>[\s\S]*?<\/thought>/g, '').trim()
}

export function parseJSON(text: string): unknown {
  const noThoughts = text.replace(/<thought>[\s\S]*?<\/thought>/g, '')
  const noFences = noThoughts.replace(/```[\w]*\n?|\n?```/g, '').trim()
  const match = noFences.match(/(\{[\s\S]*\}|\[[\s\S]*\])/)
  if (!match) throw new SyntaxError(`No JSON found in: ${noFences.slice(0, 100)}`)
  const sanitized = match[0].replace(/\\([^"\\/bfnrtu])/g, '$1')
  return JSON.parse(sanitized)
}

export async function chatComplete(
  client: OpenAI,
  model: string,
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
  maxTokens = 512,
  temperature = 0.2,
): Promise<string> {
  const res = await client.chat.completions.create({ model, messages, max_tokens: maxTokens, temperature })
  return stripThoughts(res.choices[0]?.message?.content ?? '')
}

export async function* streamChatComplete(
  client: OpenAI,
  model: string,
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
  maxTokens = 2048,
  temperature = 0.7,
): AsyncGenerator<string> {
  const stream = await client.chat.completions.create({
    model, messages, max_tokens: maxTokens, temperature, stream: true,
  })

  let buf = '', inThought = false

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? ''
    if (!delta) continue

    buf += delta
    let output = ''

    while (buf.length > 0) {
      if (!inThought) {
        const tagStart = buf.indexOf('<thought>')
        if (tagStart === -1) {
          const ltIdx = buf.lastIndexOf('<')
          if (ltIdx !== -1 && ltIdx > buf.length - 9) {
            output += buf.slice(0, ltIdx); buf = buf.slice(ltIdx)
          } else { output += buf; buf = '' }
          break
        }
        output += buf.slice(0, tagStart); buf = buf.slice(tagStart + 9); inThought = true
      } else {
        const tagEnd = buf.indexOf('</thought>')
        if (tagEnd === -1) { buf = buf.slice(Math.max(0, buf.length - 10)); break }
        buf = buf.slice(tagEnd + 10); inThought = false
      }
    }

    if (output) yield output
  }
}

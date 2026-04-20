export const mainConfig = {
  llm: {
    baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
    apiKey: process.env.GEMINI_API_KEY ?? '',
    model: 'gemma-4-31b-it',
    maxTokens: 2048,
    temperature: 0.7,
  },
  context: {
    maxRawTurns: 4,   // intentionally short to prove memory pool is needed
  },
  server: {
    port: 3000,
  },
} as const

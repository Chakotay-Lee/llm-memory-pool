export const memoryConfig = {
  llm: {
    baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai',
    apiKey: process.env.GEMINI_API_KEY ?? '',
    model: 'gemma-4-26b-a4b-it',
    maxTokens: 512,
    temperature: 0.2,
  },
  pool: {
    maxEntries: 100,
    compressionWindow: 20,
    injectBudget: 800,       // max tokens injected into main thread per turn
    minTurnsToCompress: 2,   // must be << maxRawTurns so data compresses before falling out of window
  },
} as const

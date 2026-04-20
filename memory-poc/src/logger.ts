import { appendFileSync } from 'node:fs'
import type { AppEvent } from './types'

const logFile = process.env.LOG_FILE ?? ''

export function logEvent(event: AppEvent): void {
  if (!logFile) return
  try {
    appendFileSync(logFile, JSON.stringify(event) + '\n')
  } catch { /* non-critical */ }
}

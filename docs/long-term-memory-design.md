# Long-Term Memory — Design Draft

## Relationship to Short-Term Memory Pool

Short-term and long-term memory are **parallel systems**, not a waterfall:

```
User Input
    ├──► Short-term pool query  ──► immediate result ──► inject into turn
    └──► Long-term keyword gate ──► if hit, async LTM query
                                          │
                                    result arrives later
                                          │
                                    ├── update short-term cache
                                    └── supplement reply to user
```

Long-term is **not** an overflow bucket. Each has its own write conditions.

---

## Two Categories of Long-Term Memory

### Category 1: Derived Facts
Persistent facts extracted from conversation by the compression LLM.

```
Examples:
  - User identity: name, role, location
  - User preferences: food, communication style
  - Recurring decisions: technology choices, project directions
```

Write trigger: compression LLM marks entry with `persist: true`.

Rule-based pre-filter (before LLM judgment):
- Contains identity words (name, role, location) → persist
- Contains "always", "never", "I prefer" → persist
- Contains "this time", "for now", "temporarily" → do NOT persist

### Category 2: Documents and Artifacts

All external content provided to the AI:

| Type | Source | Example |
|---|---|---|
| `paste_text` | User pastes in chat | Article, code snippet |
| `paste_image` | User pastes screenshot | Error screenshot, diagram |
| `upload_pdf` | User uploads file | Contract, report |
| `upload_docx` | User uploads file | Word document |
| `upload_xlsx` | User uploads file | Spreadsheet |
| `email` | Email subsystem | Incoming email with attachments |
| `artifact` | AI-generated | Report, plan, letter |

Write trigger: explicit user request ("save this", "make a report") or intent detection (verbs: 整理、製作、寫、產生、建立).

---

## Storage Structure

```
long-term/
├── index.json          ← always in memory (summaries + keywords only)
├── hash_table.json     ← SHA-256 → item_id (deduplication)
├── chunks/
│   ├── doc_001.json    ← chunked text + embeddings
│   └── ...
└── files/
    ├── doc_001.pdf     ← original file
    ├── doc_002.png     ← original image
    └── ...
```

`index.json` is kept in memory at all times — it contains only summaries and keywords, so it stays small. `chunks/` and `files/` are read on demand.

---

## Item Schema

```json
{
  "id": "doc_007",
  "type": "report",
  "title": "Q1 Sales Strategy",
  "original_filename": "sales_plan_final_v3.pdf",
  "summary": "OO company's Q1 sales plan covering three markets...",
  "keywords": ["sales", "Q1", "strategy", "OO company"],
  "entities": {
    "persons": ["張總", "張偉民"],
    "organizations": ["OO公司"],
    "locations": ["Tokyo", "Osaka"]
  },
  "created_at": "2026-03-15T10:30:00Z",
  "source": "user_upload",
  "hash": "sha256:abc123...",
  "chunks_file": "chunks/doc_007.json",
  "raw_file": "files/doc_007.pdf",
  "use_count": 3,
  "attachments": [],
  "superseded_by": null
}
```

---

## Deduplication via Content Hash

```
Any file arrives
    ↓
Compute SHA-256 hash
    ├── hash exists → return existing item_id, skip storage
    └── hash new    → assign new id, store file, record hash
```

Two files with the same name but different content = two independent items.  
No version inference — if the user wants to link them, that is handled at the graph layer.

Email attachments: each attachment is stored independently and deduplicated. Multiple emails referencing the same attachment point to the same item_id.

```
email_001.attachments = ["doc_005"]
email_002.attachments = ["doc_005"]   ← same file, shared reference
```

---

## Global Keyword Table (The Gate)

Every item's keywords are merged into a global keyword table:

```json
{
  "alex":     ["fact_001", "doc_003"],
  "tokyo":    ["fact_001", "doc_007"],
  "contract": ["doc_005", "doc_006"],
  "張總":     ["doc_007", "email_001"]
}
```

At each conversation turn, user message tokens are checked against this table:

```
user message tokens ∩ keyword_table keys → candidate item IDs
    ├── empty set → skip LTM query
    └── non-empty → trigger async LTM retrieval
```

This is a cheap O(1) gate — no embedding computation, no disk access — that decides whether the expensive semantic search is worth running.

---

## Indexing by Content Type

### Text files (paste_text, md, txt, email body)
→ chunk by paragraph → embed each chunk → store in chunks/

### Structured documents (pdf, docx, xlsx, pptx)
→ parse to text → chunk → embed → store in chunks/
→ xlsx: also extract table structure as JSON summary

### Images (png, jpg, screenshot)
→ Vision LLM generates text description
→ description is treated as the "content" for chunking and embedding
→ original image stored in files/ as-is

### AI artifacts (report, letter, plan)
→ treat as paste_text
→ also record `generated_at_turn` and `trigger_message`

---

## Retrieval Flow

```
keyword gate hit → async thread starts
    ↓
1. Filter by entities / time range (fast, in-memory)
       "上個月和張總的報告"
       → time: last month + person: 張總 + type: report
    ↓
2. Semantic search on matching chunks (disk read)
    ↓
3. Return top-k chunks → update short-term cache → supplement reply
```

Multi-dimensional query:
- **Time**: `created_at` range from natural language ("last month", "yesterday")
- **Person / Org**: named entity matching
- **Type**: document type filter
- **Semantic**: embedding cosine similarity on chunks

---

## Write Path: Compression Integration

The compression LLM (Step 2) output is extended with two new fields:

```json
{
  "entries": [...],       ← short-term pool (existing)
  "persist": [...],       ← Category 1: facts to write to LTM
  "artifacts": [...]      ← Category 2: AI-generated content to store
}
```

`persist` entries go through the same Category 2 pipeline (summary + keywords + entities + chunks).

`artifacts` reference the assistant turn content directly:
```json
{
  "type": "report",
  "title": "銷售計劃 2026",
  "trigger": "user requested report",
  "content_ref": "turn_index:42"
}
```

---

## What LTM Does NOT Store

- General world knowledge (already in model weights)
- Transient session context ("this time", "for now")
- Intermediate AI reasoning (chain-of-thought, not conclusions)
- Duplicate content (same hash = same item, no re-storage)

---

## Open Questions

- [ ] Storage backend: SQLite + local files, or vector DB (Qdrant/Chroma)?
- [ ] Cross-session user identity layer (single user vs. multi-user)
- [ ] Conflict resolution: old and new facts coexist until eviction — explicit update step needed?
- [ ] LTM eviction policy: does LTM ever forget, or is it permanent?
- [ ] Graph edges: how are item relationships stored and traversed?

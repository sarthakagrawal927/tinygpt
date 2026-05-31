---
title: Using tinygpt with Continue.dev / Cline / Aider
description: tinygpt exposes an Ollama-compatible HTTP surface so it drops straight into Continue.dev, Cline, and Aider configs as a local provider. Setup + caveats.
---

# Using tinygpt with Continue.dev / Cline / Aider

`tinygpt serve` exposes both an OpenAI-compatible surface (for
lm-evaluation-harness, langchain, etc.) **and** an Ollama-compatible
surface (for the dev-tool ecosystem). The Ollama surface uses the
same generation core but emits NDJSON streaming chunks per the Ollama
protocol — Continue.dev, Cline, and Aider all configure against
`provider: ollama` and "just work" with tinygpt as the backend.

## Quickstart

1. **Start the server on Ollama's conventional port:**
   ```bash
   tinygpt serve <model.tinygpt> --port 11434
   # → listening on http://127.0.0.1:11434
   ```

2. **Sanity check the Ollama surface:**
   ```bash
   curl -s http://127.0.0.1:11434/api/tags
   # → {"models":[{"name":"tinygpt:latest", ...}]}

   curl -s http://127.0.0.1:11434/api/generate \
        -d '{"model":"tinygpt","prompt":"Hello","stream":false}'
   # → {"model":"tinygpt:latest","response":"...","done":true,...}
   ```

3. **Point a client at it** (see per-tool sections below).

## Continue.dev

Add to `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "tinygpt",
      "provider": "ollama",
      "model": "tinygpt:latest",
      "apiBase": "http://127.0.0.1:11434"
    }
  ]
}
```

Reload Continue (Cmd-Shift-P → "Continue: Reload") and the `tinygpt`
model appears in the model picker. Chat, autocomplete, and edit all
route through tinygpt's local model now — code never leaves device.

## Cline (VS Code)

Cline's settings → API Provider → "Ollama":

| Field | Value |
|---|---|
| Base URL | `http://127.0.0.1:11434` |
| Model ID | `tinygpt:latest` |

That's it. Cline's Plan-mode dialogue + Act-mode tool-call enforcement
both work against the tinygpt local model.

## Aider

`aider --model ollama/tinygpt:latest --openai-api-base http://127.0.0.1:11434`

Or in `~/.aider.conf.yml`:

```yaml
model: ollama/tinygpt:latest
openai-api-base: http://127.0.0.1:11434
```

Aider's "architect mode" can use tinygpt for either the planner or
the editor; configure two model entries if you want different
specialists for each role.

## Endpoints exposed

| Endpoint | Purpose |
|---|---|
| `GET /api/tags` | Model list — Continue/Cline use this to discover models |
| `GET /api/version` | Version probe — clients call before first request |
| `POST /api/show` | Model info — Continue uses to verify the model loaded |
| `POST /api/chat` | Chat (NDJSON streaming) — primary inference path |
| `POST /api/generate` | Completion (NDJSON streaming) — alternative path |

Plus the OpenAI surface on the same port:
- `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`

Same server, two compatible wire protocols.

## Wire format details

**Streaming = on by default** for `/api/chat` and `/api/generate`
(matches Ollama's behavior). Pass `"stream": false` for one-shot
responses.

When streaming, the wire format is **NDJSON** — one JSON object per
line, terminated by a chunk with `"done": true`:

```
{"model":"tinygpt:latest","created_at":"...","message":{"role":"assistant","content":"Hello"},"done":false}
{"model":"tinygpt:latest","created_at":"...","message":{"role":"assistant","content":" there"},"done":false}
{"model":"tinygpt:latest","created_at":"...","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}
```

This is *not* SSE (no `data: ` prefix). The OpenAI surface on
`/v1/chat/completions` is SSE — pick the right port + endpoint
for your client.

## Tool calling

Today: tinygpt's Ollama surface does **not** emit tool-call chunks
(`message.tool_calls`). Continue/Cline tool calling requires the
OpenAI surface (`/v1/chat/completions` with `tools: [...]`) or the
agent runtime path (`tinygpt agent --tools tools.json`).

Roadmap: once the **tool-call extractor (mini-router)** ships
(see docs/roadmap/north_star_refined.md), the Ollama surface will
embed tool calls in the streamed `message.content` per Ollama's
own format. Tracked as a Wave 2.6 follow-on.

## Tradeoffs vs the cloud providers

You probably came here because you wanted to keep code on-device.
Honest framing:

| Dimension | tinygpt (local) | Claude / GPT cloud |
|---|---|---|
| Latency (TTFT) | < 10 ms warm | 200–500 ms |
| Cost per token | Free | $0.50–$2 / M tokens |
| Privacy | Code stays local | Sent to cloud |
| Raw quality (SWE-bench) | Lower (1-3B class) | Higher (Sonnet 4.6) |
| Tool calling polish | Basic | Strong (Sonnet, GPT-4o) |

The right mental model: tinygpt handles **routine** asks (rename a
symbol, write a small function, explain a snippet) instantly +
privately; escalate to cloud (via tinygpt's `--cloud-escalate` flag,
or by switching providers in Continue) for the hard ones.

## Limits + bugs to know about

- `tinygpt serve` is single-threaded for inference (one model, one
  request at a time). Multiple concurrent Continue requests queue.
- Tokenizer mismatch: Continue may count tokens with its own
  tokenizer and over-/under-shoot budgets. Use generous
  `num_predict` budgets on the client side.
- `--port 11434` collides with a real Ollama install. Either stop
  Ollama (`ollama serve` shutdown) or pick a different port + set
  `apiBase` accordingly in the client config.
- Tool calls don't round-trip yet (see above).
- Streaming cancellation works (close the socket → generation
  stops; see commit c11265b for the SIGPIPE-safe wiring).

## What's next

Once the tool-call extractor (mini-router) lands, this page becomes
the canonical "use tinygpt in your editor" walkthrough — with
tool-call support, it's the first real-user-visible product surface
for the project. See docs/progress.md for shipping status.

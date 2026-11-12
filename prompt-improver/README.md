# AI Prompt Improver — Browser Extension

**Personal project, started Sept 2026 · in progress**
**My role:** sole developer (extension, backend, architecture)

A Manifest V3 browser extension that adds an "Improve" button next to the chat input on AI sites (ChatGPT, Claude.ai, and later Gemini/Perplexity). It rewrites the user's draft prompt before they send it, grounded only in the draft itself and the visible thread on that same site — it's designed to restructure and clarify, never to invent facts the user didn't state.

## The problem

Several "rewrite my prompt" extensions already exist (MetaPrompt, Promptly, PromptGPT, and others). The common failure mode is that they aren't context-aware and can silently add content the user never said. The goal here is a rewriter that (1) shows exactly what it changed and why, and (2) only ever grounds additions in what's already on the page.

## Design decisions

- **Browser extension, not an MCP connector.** MCP was considered and rejected: it only fires after a model has already received a message (too late to intercept pre-send), it only sees context the host app chooses to share, and it would only work inside Claude, not across other AI sites. A content script can intercept the draft before send, on any site.
- **Backend holds the API key.** The extension never embeds or sees the real LLM key — it calls a backend service, which is the only place a future API key and usage metering would live.
- **Thread context is read fresh from the DOM, same-site only.** No cross-platform aggregation, no persistent history store — by design, to limit scope creep and ToS risk.

## Differentiators (planned, build order)

1. **Transparency.** The rewrite contract is `{ rewritten_prompt, changes: [{type, before, after, reason}] }`, meant to be rendered as a diff/reasoning panel the user approves before sending — not yet built.
2. **Custom "lenses".** User-defined rewrite styles (Coding, Creative writing, Research, Business email) instead of one fixed tone, auto-detected but overridable — not yet built.
3. **Context-awareness.** Reading the current thread to ground additions instead of inventing new content — this is the core thesis of the project and the most defensible piece technically, since the diff UI and lenses are easy for competitors to copy.

## Tech stack

Manifest V3 (content scripts + service worker) · vanilla JS, `MutationObserver` for async DOM injection · Node.js, Express, CORS · dotenv for key management

## Repository layout

```
prompt-improver/
├── extension/
│   ├── manifest.json              # Manifest V3 config, scoped to chatgpt.com + claude.ai for now
│   ├── background.js              # service worker: routes rewrite requests to the backend
│   └── content-scripts/
│       ├── common.js              # createImproveButton, showTooltip, injectOnce (MutationObserver helper)
│       ├── chatgpt.js             # site adapter for chatgpt.com — targets #prompt-textarea
│       └── claude.js              # site adapter for claude.ai — targets the ProseMirror contenteditable div
└── backend/
    ├── server.js                  # Express server, POST /rewrite + GET /health
    ├── package.json
    └── .env.example                # PORT, LLM_API_KEY placeholders — real .env is gitignored
```

Each site needs its own adapter because `chatgpt.com` uses a plain `textarea` while `claude.ai` uses a ProseMirror `contenteditable` div, and both render their input asynchronously — `injectOnce` in `common.js` uses a `MutationObserver` to wait for the element and mount the button exactly once.

## Running it

```bash
# Backend
cd backend
npm install
cp .env.example .env   # fill in PORT (LLM_API_KEY unused until the real rewrite call is wired up)
npm start               # http://localhost:3000

# Extension
# chrome://extensions → enable Developer mode → "Load unpacked" → select extension/
# Visit chatgpt.com or claude.ai and start typing a prompt — an "Improve" button appears near the input
```

## Current state — honestly

This is a shell, not a finished product. Clicking "Improve" calls the real backend over the real message-passing path, but the backend is a stub: `POST /rewrite` echoes the draft back unchanged with an empty `changes` array. There is no LLM call, no diff UI, and no lens system yet. What **is** real and working: the Manifest V3 scaffolding, the per-site DOM adapters, and the extension → service worker → backend round trip — the plumbing the rest of the project builds on.

## What's left

- Wire a real LLM call into `backend/server.js` and return actual `changes` entries
- Build the diff/reasoning panel (currently just a "Backend reached (stub)" tooltip)
- Thread-context extraction per site, with a context budget that scales to the complexity of the draft (the complexity heuristic itself isn't defined yet)
- Usage metering in the backend for a free-tier / pay-per-prompt model
- The lens system, and Gemini/Perplexity adapters
- Check each platform's ToS before reading thread content beyond the draft box

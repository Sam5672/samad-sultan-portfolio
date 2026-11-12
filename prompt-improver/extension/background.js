// Service worker: routes rewrite requests to the backend, which holds the
// real API key. The extension itself never sees or stores that key.

const BACKEND_URL = "http://localhost:3000/rewrite";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "IMPROVE_PROMPT") {
    fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ draft: message.draft }),
    })
      .then((res) => res.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => {
        console.error("[AI Prompt Improver] backend call failed:", err);
        sendResponse({ ok: false, error: String(err) });
      });
    return true; // keep the message channel open for the async response
  }
});
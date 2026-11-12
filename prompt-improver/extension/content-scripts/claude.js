// Site adapter for claude.ai. The input is a ProseMirror contenteditable div.

function findClaudeInput() {
  return document.querySelector("div[contenteditable='true']");
}

function getClaudeDraftText(inputEl) {
  return inputEl.innerText ?? "";
}

injectOnce("aiPromptImproverMounted", findClaudeInput, (inputEl) => {
  const container = inputEl.parentElement;
  const button = createImproveButton(() => {
    const draft = getClaudeDraftText(inputEl);
    showTooltip(button, "Improving…");
    chrome.runtime.sendMessage({ type: "IMPROVE_PROMPT", draft }, (response) => {
      console.log("[AI Prompt Improver] Claude response:", response);
      showTooltip(button, response?.ok ? "Backend reached (stub)" : "Backend call failed");
    });
  });
  container.appendChild(button);
});
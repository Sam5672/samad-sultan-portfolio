// Site adapter for chatgpt.com / chat.openai.com.

function findChatGptInput() {
  return document.querySelector("#prompt-textarea");
}

function getChatGptDraftText(inputEl) {
  return inputEl.innerText ?? inputEl.value ?? "";
}

injectOnce("aiPromptImproverMounted", findChatGptInput, (inputEl) => {
  const container = inputEl.closest("form") ?? inputEl.parentElement;
  const button = createImproveButton(() => {
    const draft = getChatGptDraftText(inputEl);
    showTooltip(button, "Improving…");
    chrome.runtime.sendMessage({ type: "IMPROVE_PROMPT", draft }, (response) => {
      console.log("[AI Prompt Improver] ChatGPT response:", response);
      showTooltip(button, response?.ok ? "Backend reached (stub)" : "Backend call failed");
    });
  });
  container.appendChild(button);
});
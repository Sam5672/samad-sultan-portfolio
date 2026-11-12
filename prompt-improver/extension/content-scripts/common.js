// Shared helpers used by every per-site adapter.

function createImproveButton(onClick) {
  const button = document.createElement("button");
  button.textContent = "Improve";
  button.type = "button";
  button.dataset.aiPromptImprover = "true";
  button.style.cssText = [
    "margin-left: 8px",
    "padding: 4px 10px",
    "font-size: 12px",
    "border-radius: 6px",
    "border: 1px solid #8884",
    "background: #6c5ce7",
    "color: white",
    "cursor: pointer",
    "z-index: 999999",
  ].join(";");
  button.addEventListener("click", onClick);
  return button;
}

function showTooltip(anchorEl, text) {
  const tip = document.createElement("div");
  tip.textContent = text;
  tip.style.cssText = [
    "position: absolute",
    "margin-top: 4px",
    "padding: 4px 8px",
    "font-size: 12px",
    "background: #222",
    "color: #fff",
    "border-radius: 4px",
    "z-index: 999999",
  ].join(";");
  anchorEl.insertAdjacentElement("afterend", tip);
  setTimeout(() => tip.remove(), 2000);
}

// Waits for `findTargetFn()` to return a truthy element, then runs
// `mountFn(target)` exactly once for that element. Re-checks on DOM mutations
// since these sites render the input asynchronously and can replace it.
function injectOnce(markerAttr, findTargetFn, mountFn) {
  const tryMount = () => {
    const target = findTargetFn();
    if (target && !target.dataset[markerAttr]) {
      target.dataset[markerAttr] = "true";
      mountFn(target);
    }
  };

  tryMount();
  const observer = new MutationObserver(tryMount);
  observer.observe(document.body, { childList: true, subtree: true });
}
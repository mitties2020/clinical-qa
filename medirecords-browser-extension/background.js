chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.type !== "capture-result") return false;
  chrome.storage.local.set({ latestCapture: message.capture || null }, () => sendResponse({ ok: true }));
  return true;
});

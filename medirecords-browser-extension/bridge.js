(() => {
  const timer = window.setTimeout(() => finish(null), 2000);
  function finish(capture) {
    window.clearTimeout(timer);
    window.removeEventListener("message", onMessage);
    chrome.runtime.sendMessage({ type: "capture-result", capture });
  }
  function onMessage(event) {
    if (event.source !== window || !event.data || event.data.type !== "VIVIDMEDI_CAPTURE_RESPONSE") return;
    finish(event.data.capture || null);
  }
  window.addEventListener("message", onMessage);
  window.postMessage({ type: "VIVIDMEDI_CAPTURE_REQUEST" }, "*");
})();

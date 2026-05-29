(function () {
  if (window.__vividMediMediRecordsSyncInstalled) return;
  window.__vividMediMediRecordsSyncInstalled = true;
  const STORE_KEY = "__vividMediCapturedAppointmentJson";

  function looksLikeAppointmentRow(item) {
    return Boolean(item && typeof item === "object" && (item.patientGuid || item.appointmentGuid) && (item.kendoStartTime || item.scheduledTimeStr || item.startTimeStr) && (item.practiceAppointmentTypeName || item.mobilePhone || item.firstName || item.lastName));
  }

  function findAppointmentRows(value, depth = 0) {
    if (!value || depth > 7) return null;
    if (Array.isArray(value)) {
      if (value.filter(looksLikeAppointmentRow).length) return value;
      for (const item of value) {
        const nested = findAppointmentRows(item, depth + 1);
        if (nested) return nested;
      }
      return null;
    }
    if (typeof value === "object") {
      const keys = ["appointments", "items", "data", "results", "rows", "appointmentList", ...Object.keys(value)];
      for (const key of keys) {
        const nested = findAppointmentRows(value[key], depth + 1);
        if (nested) return nested;
      }
    }
    return null;
  }

  function storeCapture(payload, sourceUrl) {
    const rows = findAppointmentRows(payload);
    if (!rows) return;
    const capture = { capturedAt: new Date().toISOString(), sourceUrl: sourceUrl || location.href, appointments: rows, raw: payload };
    const existing = JSON.parse(window.localStorage.getItem(STORE_KEY) || "[]");
    existing.unshift(capture);
    window.localStorage.setItem(STORE_KEY, JSON.stringify(existing.slice(0, 20)));
  }

  function maybeParseText(text, sourceUrl) {
    if (!text || !/"patientGuid"|"appointmentGuid"|"kendoStartTime"/.test(text)) return;
    try { storeCapture(JSON.parse(text), sourceUrl); } catch {}
  }

  function latestCapture() {
    try { return JSON.parse(window.localStorage.getItem(STORE_KEY) || "[]")[0] || null; } catch { return null; }
  }

  function scanCurrentPage() {
    const text = document.body ? document.body.innerText || document.body.textContent || "" : "";
    maybeParseText(text.trim(), location.href);
    return latestCapture();
  }

  const originalFetch = window.fetch;
  if (typeof originalFetch === "function") {
    window.fetch = async function (...args) {
      const response = await originalFetch.apply(this, args);
      try {
        const url = typeof args[0] === "string" ? args[0] : args[0] && args[0].url;
        response.clone().text().then((text) => maybeParseText(text, url || response.url)).catch(() => {});
      } catch {}
      return response;
    };
  }

  const OriginalXHR = window.XMLHttpRequest;
  if (OriginalXHR) {
    const originalOpen = OriginalXHR.prototype.open;
    const originalSend = OriginalXHR.prototype.send;
    OriginalXHR.prototype.open = function (method, url, ...rest) {
      this.__vividMediUrl = url;
      return originalOpen.call(this, method, url, ...rest);
    };
    OriginalXHR.prototype.send = function (...args) {
      this.addEventListener("load", function () {
        try { maybeParseText(this.responseText, this.__vividMediUrl || this.responseURL); } catch {}
      });
      return originalSend.apply(this, args);
    };
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window || !event.data || event.data.type !== "VIVIDMEDI_CAPTURE_REQUEST") return;
    window.postMessage({ type: "VIVIDMEDI_CAPTURE_RESPONSE", capture: scanCurrentPage() || latestCapture() }, "*");
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scanCurrentPage, { once: true });
  } else {
    scanCurrentPage();
  }
})();

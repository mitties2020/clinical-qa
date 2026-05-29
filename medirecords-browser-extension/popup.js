const endpointInput = document.getElementById("endpoint");
const tokenInput = document.getElementById("token");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");

function setStatus(message) { statusEl.textContent = message; }

function describeCapture(capture) {
  if (!capture || !Array.isArray(capture.appointments)) {
    summaryEl.textContent = "";
    return;
  }
  const first = capture.appointments[0] || {};
  summaryEl.textContent = [
    `Appointments: ${capture.appointments.length}`,
    `Captured: ${capture.capturedAt || "unknown"}`,
    `Source: ${capture.sourceUrl || "unknown"}`,
    first.kendoStartTime ? `First time: ${first.kendoStartTime}` : "",
    first.practiceAppointmentTypeName ? `Type: ${first.practiceAppointmentTypeName}` : "",
  ].filter(Boolean).join("\n");
}

async function saveSettings() {
  await chrome.storage.local.set({ endpoint: endpointInput.value.trim(), token: tokenInput.value.trim() });
}

async function loadSettings() {
  const data = await chrome.storage.local.get(["endpoint", "token", "latestCapture"]);
  if (data.endpoint) endpointInput.value = data.endpoint;
  if (data.token) tokenInput.value = data.token;
  describeCapture(data.latestCapture);
}

async function captureCurrentPage() {
  await saveSettings();
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) throw new Error("No active tab found.");
  await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["bridge.js"] });
  await new Promise((resolve) => setTimeout(resolve, 600));
  const data = await chrome.storage.local.get(["latestCapture"]);
  if (!data.latestCapture || !Array.isArray(data.latestCapture.appointments)) {
    throw new Error("Could not find appointment JSON yet. Reload the MediRecords diary page, wait for appointments to appear, then click Capture.");
  }
  describeCapture(data.latestCapture);
  return data.latestCapture;
}

async function syncCapture() {
  await saveSettings();
  let { latestCapture } = await chrome.storage.local.get(["latestCapture"]);
  if (!latestCapture || !Array.isArray(latestCapture.appointments)) latestCapture = await captureCurrentPage();
  const token = tokenInput.value.trim();
  const response = await fetch(endpointInput.value.trim(), {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}`, "X-VividMedi-Sync-Token": token } : {}) },
    body: JSON.stringify({ source: "browser-extension", syncToken: token, capturedAt: latestCapture.capturedAt, sourceUrl: latestCapture.sourceUrl, appointments: latestCapture.appointments }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload.ok) throw new Error(payload.error || `Sync failed with status ${response.status}`);
  setStatus(`Synced ${payload.appointments || latestCapture.appointments.length} appointments to VividMedi.`);
}

document.getElementById("capture").addEventListener("click", async () => {
  try {
    setStatus("Capturing from this MediRecords tab...");
    const capture = await captureCurrentPage();
    setStatus(`Captured ${capture.appointments.length} appointments. You can sync now.`);
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
});

document.getElementById("sync").addEventListener("click", async () => {
  try { setStatus("Syncing to VividMedi..."); await syncCapture(); } catch (error) { setStatus(`Error: ${error.message}`); }
});

document.getElementById("copy").addEventListener("click", async () => {
  try {
    const { latestCapture } = await chrome.storage.local.get(["latestCapture"]);
    if (!latestCapture) throw new Error("Nothing captured yet.");
    await navigator.clipboard.writeText(JSON.stringify(latestCapture.appointments || [], null, 2));
    setStatus("Copied latest captured appointment JSON.");
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  }
});

endpointInput.addEventListener("change", saveSettings);
tokenInput.addEventListener("change", saveSettings);
loadSettings();

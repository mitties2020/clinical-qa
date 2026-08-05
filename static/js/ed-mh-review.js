(function () {
  "use strict";

  const CONSULT_TYPE = "ED MH Review";
  const DIRECT_LINK_VALUE = "ed-mh-review";
  const DRAFT_PREFIX = "vivid_ed_mh_review_single_box:";
  const MAX_SOURCE_CHARS = 180000;
  const MAX_FILE_CHARS = 180000;
  const MAX_STORED_NOTE_CHARS = 600000;
  const ACCEPTED_FILE_PATTERN = /\.(txt|md|markdown|csv|json|html?|log)$/i;

  const workspace = document.getElementById("edMhReviewWorkspace");
  const standardWorkspace = document.getElementById("standardConsultWorkspace");
  const consultSelect = document.getElementById("consultType");
  if (!workspace || !standardWorkspace || !consultSelect) return;

  let state = createDefaultState();
  let activeDraftKey = "";
  let saveTimer = null;
  let suppressAppointmentReload = false;

  function createDefaultState() {
    return {
      reviewText: "",
      notes: [],
      output: "",
      savedAt: ""
    };
  }

  function newId(prefix) {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return `${prefix}:${window.crypto.randomUUID()}`;
    }
    return `${prefix}:${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"']/g, (character) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;"
    }[character]));
  }

  function normaliseNote(note) {
    if (!note || typeof note !== "object") return null;
    const content = String(note.content || "").trim();
    if (!content) return null;
    return {
      id: String(note.id || newId("note")),
      label: String(note.label || "Untitled note").trim().slice(0, 180),
      content,
      timing: note.timing === "previous" ? "previous" : "current",
      source: String(note.source || "Uploaded note").trim().slice(0, 100),
      selected: Boolean(note.selected),
      removable: note.removable !== false
    };
  }

  function normaliseState(raw) {
    const clean = createDefaultState();
    if (!raw || typeof raw !== "object") return clean;
    clean.reviewText = String(raw.reviewText || "");
    clean.output = String(raw.output || "");
    clean.savedAt = String(raw.savedAt || "");
    clean.notes = Array.isArray(raw.notes) ? raw.notes.map(normaliseNote).filter(Boolean) : [];
    return clean;
  }

  function currentAppointment() {
    return window.VividMediAppointmentsUI?.getActiveAppointment?.() || null;
  }

  function currentDraftKey() {
    const appointment = currentAppointment();
    const key = appointment?.appointmentGuid || appointment?.patientGuid || "one-note-link";
    return `${DRAFT_PREFIX}${String(key).replace(/[^a-zA-Z0-9._-]/g, "_")}`;
  }

  function isActive() {
    return String(consultSelect.value || "").toLowerCase() === CONSULT_TYPE.toLowerCase();
  }

  function saveNow() {
    activeDraftKey = activeDraftKey || currentDraftKey();
    try {
      localStorage.setItem(activeDraftKey, JSON.stringify(state));
      setSaveStatus(state.savedAt ? `Saved output ${state.savedAt}` : "Draft saved on this device");
    } catch {
      setMessage("The draft is too large for browser storage. It remains on screen; copy any important text before closing this tab.", true);
    }
  }

  function scheduleSave() {
    window.clearTimeout(saveTimer);
    saveTimer = window.setTimeout(saveNow, 350);
  }

  function syncedNotes() {
    const notes = window.VividMediAppointmentsUI?.getNotesForActivePatient?.() || [];
    return notes.map((note) => normaliseNote({ ...note, removable: false })).filter(Boolean);
  }

  function mergeSyncedNotes() {
    const selectionById = new Map(state.notes.map((note) => [note.id, note.selected]));
    const uploaded = state.notes.filter((note) => note.removable !== false);
    const synced = syncedNotes().map((note) => ({
      ...note,
      selected: selectionById.has(note.id) ? selectionById.get(note.id) : note.selected
    }));
    state.notes = [...synced, ...uploaded];
  }

  function loadDraftForCurrentAppointment(force) {
    const nextKey = currentDraftKey();
    if (!force && activeDraftKey === nextKey) {
      mergeSyncedNotes();
      syncDomFromState();
      return;
    }
    activeDraftKey = nextKey;
    let loaded = null;
    try {
      loaded = JSON.parse(localStorage.getItem(nextKey) || "null");
    } catch {
      loaded = null;
    }
    state = normaliseState(loaded);
    mergeSyncedNotes();
    syncDomFromState();
    scheduleSave();
  }

  function setSaveStatus(message) {
    const target = document.getElementById("edmhSaveStatus");
    if (target) target.textContent = message || "";
  }

  function setMessage(message, isError) {
    const target = document.getElementById("edmhMessage");
    if (!target) return;
    target.textContent = message || "";
    target.classList.toggle("is-error", Boolean(isError));
  }

  function patientLabel() {
    const appointment = currentAppointment();
    return appointment?.patientName || "No patient selected - free typing is still available";
  }

  function renderWorkspace() {
    workspace.innerHTML = `
      <section class="edmh-input-panel" aria-labelledby="edmhTitle">
        <header class="edmh-panel-heading">
          <div>
            <div class="edmh-eyebrow">MH ED review</div>
            <h2 class="edmh-title" id="edmhTitle">One review workspace</h2>
            <div class="edmh-patient" id="edmhPatient"></div>
          </div>
          <div class="edmh-save-status" id="edmhSaveStatus" aria-live="polite"></div>
        </header>

        <label class="edmh-main-label" for="edmhReviewText">Current review notes</label>
        <div class="edmh-help" id="edmhReviewHelp">Type or paste the entire current review in any order. This is the only clinical entry box.</div>
        <textarea id="edmhReviewText" class="edmh-review-text" aria-describedby="edmhReviewHelp" placeholder="Type the current interview, collateral, observations, MSE, formulation and plan here..."></textarea>

        <section class="edmh-source-library" aria-labelledby="edmhSourcesTitle">
          <div class="edmh-source-heading">
            <div>
              <h3 id="edmhSourcesTitle">Notes to integrate</h3>
              <div class="edmh-help">Tick any current or previous note that should be included in this review.</div>
            </div>
            <div class="edmh-source-actions">
              <button type="button" class="edmh-action" data-edmh-action="add-current">Add current notes</button>
              <button type="button" class="edmh-action" data-edmh-action="add-previous">Add previous notes</button>
              <button type="button" class="edmh-action" data-edmh-action="select-all">Select all</button>
              <button type="button" class="edmh-action" data-edmh-action="select-none">Select none</button>
            </div>
          </div>
          <input id="edmhCurrentFiles" class="edmh-file-input" type="file" multiple accept=".txt,.md,.markdown,.csv,.json,.html,.htm,.log,text/plain,text/markdown,text/csv,application/json,text/html" data-edmh-file-timing="current">
          <input id="edmhPreviousFiles" class="edmh-file-input" type="file" multiple accept=".txt,.md,.markdown,.csv,.json,.html,.htm,.log,text/plain,text/markdown,text/csv,application/json,text/html" data-edmh-file-timing="previous">
          <div class="edmh-note-list" id="edmhNoteList"></div>
          <div class="edmh-note-summary" id="edmhNoteSummary"></div>
        </section>

        <div class="edmh-primary-actions">
          <button type="button" class="edmh-generate" data-edmh-action="generate">Create psychiatry review</button>
          <button type="button" class="edmh-action edmh-clear" data-edmh-action="clear-review">Clear / start again</button>
        </div>
        <div class="edmh-message" id="edmhMessage" role="status" aria-live="polite"></div>
      </section>

      <section class="edmh-output-panel" aria-labelledby="edmhOutputTitle">
        <header class="edmh-panel-heading">
          <div>
            <div class="edmh-eyebrow">Copy to OneNote or the clinical record</div>
            <h2 class="edmh-title" id="edmhOutputTitle">Finished review</h2>
          </div>
        </header>
        <textarea id="edMhOutput" class="edmh-output-area" aria-label="Finished ED psychiatry review" placeholder="Your finished 12-section review will appear here..."></textarea>
        <div class="edmh-output-actions">
          <button type="button" class="edmh-action edmh-copy" data-edmh-action="copy-output">Copy review</button>
          <button type="button" class="edmh-action" data-edmh-action="complete-consult">Save to Vivid Medi outputs</button>
          <button type="button" class="edmh-action" data-edmh-action="delete-output">Clear output</button>
        </div>
        <p class="edmh-clinical-note">Review and verify all chronology, medications, allergies, MSE, risk wording and plan before signing.</p>
      </section>`;
  }

  function noteSizeLabel(content) {
    const words = String(content || "").trim().split(/\s+/).filter(Boolean).length;
    return `${words.toLocaleString()} word${words === 1 ? "" : "s"}`;
  }

  function renderNotes() {
    const list = document.getElementById("edmhNoteList");
    const summary = document.getElementById("edmhNoteSummary");
    if (!list || !summary) return;
    if (!state.notes.length) {
      list.innerHTML = '<div class="edmh-empty-notes">No synced or added notes are available yet. Select a patient or add text note files.</div>';
      summary.textContent = "The current review box will be used on its own.";
      return;
    }
    list.innerHTML = state.notes.map((note) => `
      <div class="edmh-note-card ${note.selected ? "is-selected" : ""}">
        <label class="edmh-note-choice">
          <input type="checkbox" data-edmh-note-id="${escapeHtml(note.id)}" ${note.selected ? "checked" : ""}>
          <span class="edmh-note-check" aria-hidden="true"></span>
          <span class="edmh-note-copy">
            <span class="edmh-note-title">${escapeHtml(note.label)}</span>
            <span class="edmh-note-meta"><span class="edmh-timing ${escapeHtml(note.timing)}">${note.timing === "current" ? "Current" : "Previous"}</span> ${escapeHtml(note.source)} · ${escapeHtml(noteSizeLabel(note.content))}</span>
          </span>
        </label>
        ${note.removable === false ? "" : `<button type="button" class="edmh-remove-note" data-edmh-action="remove-note" data-edmh-note-id="${escapeHtml(note.id)}" aria-label="Remove ${escapeHtml(note.label)}">&times;</button>`}
      </div>`).join("");
    const selected = state.notes.filter((note) => note.selected);
    const currentCount = selected.filter((note) => note.timing === "current").length;
    const previousCount = selected.filter((note) => note.timing === "previous").length;
    summary.textContent = `${selected.length} selected (${currentCount} current, ${previousCount} previous)`;
  }

  function syncDomFromState() {
    const patient = document.getElementById("edmhPatient");
    const review = document.getElementById("edmhReviewText");
    const output = document.getElementById("edMhOutput");
    if (patient) patient.textContent = patientLabel();
    if (review) review.value = state.reviewText;
    if (output) output.value = state.output;
    renderNotes();
    setSaveStatus(state.savedAt ? `Saved output ${state.savedAt}` : "Draft autosaves on this device");
  }

  function setActive() {
    const active = isActive();
    standardWorkspace.hidden = active;
    workspace.hidden = !active;
    document.body.classList.toggle("edmh-active", active);
    if (active) loadDraftForCurrentAppointment(false);
  }

  function selectedSourceCharacters() {
    return state.reviewText.length + state.notes
      .filter((note) => note.selected)
      .reduce((total, note) => total + note.content.length, 0);
  }

  async function addFiles(fileList, timing) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    let added = 0;
    const errors = [];
    const existingStoredCharacters = state.notes
      .filter((note) => note.removable !== false)
      .reduce((total, note) => total + note.content.length, 0);
    let storedCharacters = existingStoredCharacters;

    for (const file of files) {
      if (!ACCEPTED_FILE_PATTERN.test(file.name || "")) {
        errors.push(`${file.name}: use a text, Markdown, CSV, JSON or HTML file`);
        continue;
      }
      let content = "";
      try {
        content = String(await file.text()).replace(/\0/g, "").trim();
      } catch {
        errors.push(`${file.name}: could not read file`);
        continue;
      }
      if (!content) {
        errors.push(`${file.name}: empty file`);
        continue;
      }
      if (content.length > MAX_FILE_CHARS) {
        errors.push(`${file.name}: larger than the safe per-note limit`);
        continue;
      }
      if (storedCharacters + content.length > MAX_STORED_NOTE_CHARS) {
        errors.push(`${file.name}: browser draft storage limit reached`);
        continue;
      }
      state.notes.push({
        id: newId("upload"),
        label: file.name,
        content,
        timing: timing === "previous" ? "previous" : "current",
        source: "Added note file",
        selected: true,
        removable: true
      });
      storedCharacters += content.length;
      added += 1;
    }

    state.savedAt = "";
    renderNotes();
    scheduleSave();
    if (errors.length) setMessage(`${added ? `${added} note file${added === 1 ? "" : "s"} added. ` : ""}${errors.join("; ")}.`, true);
    else setMessage(`${added} note file${added === 1 ? "" : "s"} added and selected.`);
  }

  async function generateReview(button) {
    const selectedNotes = state.notes.filter((note) => note.selected);
    if (!state.reviewText.trim() && !selectedNotes.length) {
      setMessage("Type the current review or select at least one note first.", true);
      document.getElementById("edmhReviewText")?.focus();
      return;
    }
    if (selectedSourceCharacters() > MAX_SOURCE_CHARS) {
      setMessage("The selected material is too long for one safe generation. Deselect the least relevant older notes, then try again.", true);
      return;
    }

    const output = document.getElementById("edMhOutput");
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = "Creating review...";
    output.readOnly = true;
    setMessage(`Integrating the current review with ${selectedNotes.length} selected note${selectedNotes.length === 1 ? "" : "s"}...`);

    try {
      const response = await fetch("/api/ed-mh-review/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_review: state.reviewText,
          notes: selectedNotes.map((note) => ({
            label: note.label,
            timing: note.timing,
            source: note.source,
            content: note.content
          }))
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `Review generation failed (${response.status})`);
      const finished = String(data.clinical_notes || data.output || "").trim();
      if (!finished) throw new Error("No review was returned");
      state.output = finished;
      state.savedAt = "";
      output.value = finished;
      setMessage("Review created. Verify it clinically, edit if needed, then copy it into OneNote or the clinical record.");
      saveNow();
    } catch (error) {
      setMessage(error.message || "The review could not be generated. Your source notes are unchanged.", true);
    } finally {
      output.readOnly = false;
      button.disabled = false;
      button.textContent = originalText;
    }
  }

  async function copyOutput() {
    const output = document.getElementById("edMhOutput");
    if (!output?.value.trim()) {
      setMessage("There is no finished review to copy.", true);
      return;
    }
    try {
      await navigator.clipboard.writeText(output.value);
      setMessage("Review copied. You can paste it into OneNote or the clinical record.");
    } catch {
      output.focus();
      output.select();
      setMessage("Clipboard access was blocked. The review is selected; press Ctrl+C.", true);
    }
  }

  function completeConsult() {
    const output = document.getElementById("edMhOutput");
    if (!output?.value.trim()) {
      setMessage("Create or enter a finished review before saving it.", true);
      return;
    }
    state.output = output.value;
    state.savedAt = new Date().toLocaleString();
    const appointment = currentAppointment();
    const item = {
      id: newId("saved-ed-review"),
      patientName: appointment?.patientName || "ED psychiatry review",
      consultType: CONSULT_TYPE,
      date: state.savedAt,
      content: state.output,
      expanded: false,
      appointmentGuid: appointment?.appointmentGuid || "",
      edMhDraft: JSON.parse(JSON.stringify(state))
    };
    try {
      savedOutputs.unshift(item);
      savedOutputs = savedOutputs.slice(0, 60);
      localStorage.setItem("vivid_saved_outputs", JSON.stringify(savedOutputs));
      renderSavedOutputs();
      if (appointment?.appointmentGuid && window.VividMediAppointmentsUI) {
        suppressAppointmentReload = true;
        window.VividMediAppointmentsUI.completeAppointment(appointment.appointmentGuid);
      }
      setMessage("Review saved to Vivid Medi outputs.");
      saveNow();
    } catch {
      setMessage("The finished review remains on screen but could not be saved to Vivid Medi outputs.", true);
    }
  }

  function clearReview() {
    if (!window.confirm("Clear the current review, added files, selections and output?")) return;
    try {
      localStorage.removeItem(activeDraftKey || currentDraftKey());
    } catch {
      // The on-screen reset still works if storage is unavailable.
    }
    state = createDefaultState();
    mergeSyncedNotes();
    syncDomFromState();
    setMessage("MH ED review cleared.");
    saveNow();
  }

  function startReview() {
    if (!isActive()) return;
    loadDraftForCurrentAppointment(false);
    document.getElementById("edmhReviewText")?.focus();
    setMessage(`MH ED review ready for ${patientLabel()}.`);
  }

  function handleClick(event) {
    const button = event.target.closest("[data-edmh-action]");
    if (!button) return;
    const action = button.dataset.edmhAction;
    if (action === "add-current") {
      document.getElementById("edmhCurrentFiles")?.click();
    } else if (action === "add-previous") {
      document.getElementById("edmhPreviousFiles")?.click();
    } else if (action === "select-all") {
      state.notes.forEach((note) => { note.selected = true; });
      state.savedAt = "";
      renderNotes();
      scheduleSave();
    } else if (action === "select-none") {
      state.notes.forEach((note) => { note.selected = false; });
      state.savedAt = "";
      renderNotes();
      scheduleSave();
    } else if (action === "remove-note") {
      state.notes = state.notes.filter((note) => note.id !== button.dataset.edmhNoteId || note.removable === false);
      state.savedAt = "";
      renderNotes();
      scheduleSave();
    } else if (action === "generate") {
      generateReview(button);
    } else if (action === "copy-output") {
      copyOutput();
    } else if (action === "complete-consult") {
      completeConsult();
    } else if (action === "delete-output") {
      state.output = "";
      state.savedAt = "";
      const output = document.getElementById("edMhOutput");
      if (output) output.value = "";
      setMessage("Finished output cleared. Your review notes and selected sources were kept.");
      scheduleSave();
    } else if (action === "clear-review") {
      clearReview();
    }
  }

  function handleInput(event) {
    if (event.target.id === "edmhReviewText") {
      state.reviewText = event.target.value;
      state.savedAt = "";
      scheduleSave();
    } else if (event.target.id === "edMhOutput") {
      state.output = event.target.value;
      state.savedAt = "";
      scheduleSave();
    }
  }

  function handleChange(event) {
    if (event.target.matches("[data-edmh-note-id][type=checkbox]")) {
      const note = state.notes.find((item) => item.id === event.target.dataset.edmhNoteId);
      if (note) note.selected = event.target.checked;
      state.savedAt = "";
      renderNotes();
      scheduleSave();
      return;
    }
    if (event.target.matches("[data-edmh-file-timing]")) {
      addFiles(event.target.files, event.target.dataset.edmhFileTiming);
      event.target.value = "";
    }
  }

  function interceptExistingButton(buttonId, handler) {
    const button = document.getElementById(buttonId);
    if (!button) return;
    button.addEventListener("click", (event) => {
      if (!isActive()) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      handler(event);
    }, true);
  }

  function wrapSavedOutputEditor() {
    const original = window.editSavedOutput;
    window.editSavedOutput = function (index) {
      const item = typeof savedOutputs !== "undefined" ? savedOutputs[index] : null;
      if (!item || item.consultType !== CONSULT_TYPE || !item.edMhDraft) {
        if (typeof original === "function") original(index);
        return;
      }
      consultSelect.value = CONSULT_TYPE;
      consultSelect.dispatchEvent(new Event("change", { bubbles: true }));
      activeDraftKey = `${DRAFT_PREFIX}saved-${String(item.id).replace(/[^a-zA-Z0-9._-]/g, "_")}`;
      state = normaliseState(item.edMhDraft);
      state.output = item.content || state.output;
      state.savedAt = "";
      syncDomFromState();
      document.getElementById("edMhOutput")?.focus();
      setMessage("Saved MH ED review reopened for editing.");
      saveNow();
    };
  }

  function bindEvents() {
    consultSelect.addEventListener("change", setActive);
    workspace.addEventListener("click", handleClick);
    workspace.addEventListener("input", handleInput);
    workspace.addEventListener("change", handleChange);

    interceptExistingButton("convertBtn", () => {
      const button = workspace.querySelector('[data-edmh-action="generate"]');
      if (button) generateReview(button);
    });
    interceptExistingButton("completeConsultBtn", completeConsult);
    interceptExistingButton("editOutputBtn", () => document.getElementById("edMhOutput")?.focus());
    interceptExistingButton("clearOutputBtn", () => {
      state.output = "";
      const output = document.getElementById("edMhOutput");
      if (output) output.value = "";
      scheduleSave();
    });
    interceptExistingButton("clearBtn", clearReview);
    interceptExistingButton("clearInputBtn", clearReview);
    interceptExistingButton("startConsultBtn", startReview);

    const patientName = document.getElementById("activePatientName");
    if (patientName && window.MutationObserver) {
      new MutationObserver(() => {
        if (suppressAppointmentReload) {
          suppressAppointmentReload = false;
          return;
        }
        if (isActive()) loadDraftForCurrentAppointment(false);
      }).observe(patientName, { childList: true, characterData: true, subtree: true });
    }
  }

  function activateDirectLink() {
    const params = new URLSearchParams(window.location.search);
    const requested = params.get("consult");
    if (requested !== DIRECT_LINK_VALUE) return;
    document.body.classList.add("edmh-direct");
    if (params.get("focus") === "1" && typeof window.setStealth === "function") {
      window.setStealth(true);
    }
    const option = Array.from(consultSelect.options).find((item) => item.value.toLowerCase() === CONSULT_TYPE.toLowerCase());
    if (!option) return;
    consultSelect.value = option.value;
    consultSelect.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function init() {
    renderWorkspace();
    bindEvents();
    wrapSavedOutputEditor();
    activateDirectLink();
    setActive();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  window.VividMediEdMhReview = {
    consultationType: CONSULT_TYPE,
    directLink: "/patient-list?consult=ed-mh-review&focus=1",
    getState: () => JSON.parse(JSON.stringify(state)),
    loadState: (nextState) => {
      state = normaliseState(nextState);
      syncDomFromState();
      scheduleSave();
    }
  };
})();

(function () {
  const storage = window.VividMediAppointmentStorage;
  const service = window.VividMediAppointmentsService;
  let appointments = [];
  let saveTimer = null;

  function visibleAppointments() {
    return appointments.filter((appointment) => !appointment.isDeleted);
  }

  function saveNow() {
    storage.saveAppointmentsToStorage(appointments);
  }

  function saveSoon() {
    window.clearTimeout(saveTimer);
    saveTimer = window.setTimeout(saveNow, 500);
  }

  function setStatus(message) {
    const status = document.getElementById("appointmentsStatus");
    if (status) status.textContent = message;
  }

  function clearNode(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function makeEl(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function applyHeaderBranding() {
    const brand = document.getElementById("logoClick");
    if (!brand) return;
    brand.removeAttribute("style");
    brand.className = "app-brand";
    brand.setAttribute("aria-label", "Vivid Medi Clinical Documentation Aid");
    brand.innerHTML = [
      '<span class="app-brand-mark" aria-hidden="true"></span>',
      '<span class="app-brand-copy">',
      '<span class="app-brand-name">Vivid Medi</span>',
      '<span class="app-brand-tagline">Clinical Documentation Aid</span>',
      "</span>"
    ].join("");
  }

  function installClinicalDocumentationStandards() {
    if (window.__vividClinicalStandardsInstalled || typeof window.fetch !== "function") return;
    window.__vividClinicalStandardsInstalled = true;
    const originalFetch = window.fetch.bind(window);

    window.fetch = function vividStandardsFetch(input, init) {
      const url = typeof input === "string" ? input : input?.url;
      const options = init ? { ...init } : init;

      if (url === "/convert-notes" && options?.body) {
        try {
          const payload = JSON.parse(options.body);
          if (payload?.clinical_data && !payload.clinical_data.includes("AI documentation quality instructions")) {
            const selectedConsultType = payload.consult_type || document.getElementById("consultType")?.value || "selected consult type";
            payload.clinical_data = [
              "AI documentation quality instructions (do not reproduce this instruction block in the final note):",
              `- The clinician-selected consult type is: ${selectedConsultType}. Use it as the authoritative frame. Do not choose a different consult type.`,
              "- Optimise the output within that selected type for Australian medical documentation standards.",
              "- Make the note clinically robust, concise, defensible, and useful for continuity of care.",
              "- Preserve documented facts, important positives/negatives, uncertainty, risks, medication details, contraindications, monitoring, follow-up, and safety-netting.",
              "- Where the selected type or content relates to DVA, allied health, renewal, veteran care, weight management scripts, or referral justification, write to an audit-ready DVA documentation standard without inventing accepted conditions or entitlement details.",
              "- Use clear headings, professional formatting, Australian spelling, and write 'Not documented' where clinically important information is missing.",
              "",
              "Clinical data:",
              payload.clinical_data
            ].join("\n");
            options.body = JSON.stringify(payload);
          }
        } catch {
          return originalFetch(input, init);
        }
      }

      return originalFetch(input, options);
    };
  }

  function appointmentById(appointmentGuid) {
    return appointments.find((appointment) => appointment.appointmentGuid === appointmentGuid);
  }

  function toggleAppointment(appointmentGuid) {
    const appointment = appointmentById(appointmentGuid);
    if (!appointment) return;
    appointment.isExpanded = !appointment.isExpanded;
    appointment.lastSavedAt = new Date().toISOString();
    saveNow();
    renderAppointments();
  }

  function deleteAppointment(appointmentGuid) {
    const appointment = appointmentById(appointmentGuid);
    if (!appointment) return;
    appointment.isDeleted = true;
    appointment.isExpanded = false;
    appointment.lastSavedAt = new Date().toISOString();
    saveNow();
    renderAppointments();
  }

  function updateConsultNote(appointmentGuid, value, statusEl) {
    const appointment = appointmentById(appointmentGuid);
    if (!appointment) return;
    appointment.consultNote = value;
    appointment.lastSavedAt = new Date().toISOString();
    if (statusEl) statusEl.innerHTML = "&#10003; Saved";
    saveSoon();
  }

  function analyseAppointmentNote(appointmentGuid) {
    const appointment = appointmentById(appointmentGuid);
    if (!appointment) return;

    const note = String(appointment.consultNote || appointment.appointmentNote || "").trim();
    if (!note) {
      setStatus("Paste consult notes into the appointment first.");
      return;
    }

    const clinicalInput = document.getElementById("clinicalInput");
    if (!clinicalInput) {
      setStatus("Main consultation note box was not found.");
      return;
    }

    const selectedConsultType = document.getElementById("consultType")?.value || "selected consult type";
    document.getElementById("consultNotesTab")?.click();
    clinicalInput.value = [
      `Patient: ${appointment.patientName || "Unknown patient"}`,
      `Age: ${appointment.age === null || appointment.age === undefined ? "Not documented" : appointment.age}`,
      `Mobile: ${appointment.mobilePhone || "Not documented"}`,
      `MediRecords appointment type: ${appointment.appointmentType || "Not documented"}`,
      `WA appointment time: ${appointment.startTimeWA || "Not documented"}`,
      "",
      "Clinical notes / dictation:",
      note
    ].join("\n");
    clinicalInput.dispatchEvent(new Event("input", { bubbles: true }));
    setStatus(`Appointment notes sent for analysis using your selected consult type: ${selectedConsultType}.`);
    document.getElementById("convertBtn")?.click();
  }

  function renderCollapsedAppointment(appointment) {
    const row = makeEl("button", "appointment-row");
    row.type = "button";
    row.setAttribute("aria-expanded", "false");
    row.setAttribute("aria-label", `Open appointment for ${appointment.patientName}`);
    row.addEventListener("click", () => toggleAppointment(appointment.appointmentGuid));

    row.appendChild(makeEl("span", "", appointment.startTimeWA || ""));
    row.appendChild(makeEl("span", "", appointment.patientName || "Unknown patient"));
    row.appendChild(makeEl("span", "appointment-toggle-icon", ">"));
    return row;
  }

  function renderDetail(label, value) {
    const row = makeEl("div");
    row.appendChild(makeEl("span", "appointment-detail-label", label));
    row.appendChild(makeEl("span", "appointment-detail-value", value || "-"));
    return row;
  }

  function safeDomId(value) {
    return String(value || "").replace(/[^a-zA-Z0-9_-]/g, "-");
  }

  function renderExpandedAppointment(appointment) {
    const card = makeEl("div", "appointment-card expanded");
    const top = makeEl("div", "appointment-card-top");

    const toggle = makeEl("button", "appointment-card-toggle");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "true");
    toggle.setAttribute("aria-label", `Collapse appointment for ${appointment.patientName}`);
    toggle.addEventListener("click", () => toggleAppointment(appointment.appointmentGuid));
    toggle.appendChild(makeEl("span", "appointment-card-time", appointment.startTimeWA || ""));
    toggle.appendChild(makeEl("strong", "appointment-card-name", appointment.patientName || "Unknown patient"));
    toggle.appendChild(makeEl("span", "appointment-toggle-icon appointment-toggle-icon-open", ">"));

    const deleteButton = makeEl("button", "appointment-delete");
    deleteButton.type = "button";
    deleteButton.innerHTML = "&times;";
    deleteButton.setAttribute("aria-label", "Delete appointment locally");
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteAppointment(appointment.appointmentGuid);
    });

    top.appendChild(toggle);
    top.appendChild(deleteButton);
    card.appendChild(top);

    const details = makeEl("div", "appointment-details");
    details.appendChild(renderDetail("Type", appointment.appointmentType));
    details.appendChild(renderDetail("Age", appointment.age === null || appointment.age === undefined ? "" : String(appointment.age)));
    details.appendChild(renderDetail("Mobile", appointment.mobilePhone));
    details.appendChild(renderDetail("WA Time", appointment.startTimeWA ? `${appointment.startTimeWA} AWST` : ""));
    card.appendChild(details);

    const noteLabel = makeEl("label", "appointment-note-label", "Consult notes");
    const noteId = `appointment-note-${safeDomId(appointment.appointmentGuid)}`;
    noteLabel.setAttribute("for", noteId);
    card.appendChild(noteLabel);

    const textarea = makeEl("textarea");
    textarea.id = noteId;
    textarea.value = appointment.consultNote || "";
    textarea.placeholder = "Paste or type consult notes here...";
    const saveStatus = makeEl("div", "appointment-save-status");
    saveStatus.innerHTML = "&#10003; Saved";
    textarea.addEventListener("input", (event) => updateConsultNote(appointment.appointmentGuid, event.target.value, saveStatus));
    card.appendChild(textarea);

    const noteActions = makeEl("div", "appointment-note-actions");
    const analyseButton = makeEl("button", "btn btn-primary appointment-analyse-btn", "Analyse notes");
    analyseButton.type = "button";
    analyseButton.addEventListener("click", () => analyseAppointmentNote(appointment.appointmentGuid));
    noteActions.appendChild(analyseButton);
    card.appendChild(noteActions);
    card.appendChild(saveStatus);

    if (appointment.missingFromLatestSync) {
      card.appendChild(makeEl("div", "appointment-missing", "Not present in latest sync"));
    }
    return card;
  }

  function renderAppointments() {
    const list = document.getElementById("appointmentsList");
    if (!list) return;
    clearNode(list);

    const visible = visibleAppointments();
    if (!visible.length) {
      list.appendChild(makeEl("div", "appointments-empty", "No appointments loaded"));
      return;
    }

    visible.forEach((appointment) => {
      list.appendChild(appointment.isExpanded
        ? renderExpandedAppointment(appointment)
        : renderCollapsedAppointment(appointment));
    });
  }

  function openImportModal() {
    const modal = document.getElementById("appointmentImportModal");
    const input = document.getElementById("appointmentImportText");
    const message = document.getElementById("appointmentImportMessage");
    if (!modal || !input) return;
    if (message) message.textContent = "";
    input.value = "";
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
    input.focus();
  }

  function closeImportModal() {
    const modal = document.getElementById("appointmentImportModal");
    if (!modal) return;
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
  }

  function importAppointmentsFromText() {
    const input = document.getElementById("appointmentImportText");
    const message = document.getElementById("appointmentImportMessage");
    if (!input) return;

    try {
      const payload = service.parseAppointmentPayloadText(input.value);
      const rawAppointments = service.extractAppointmentArrayFromPayload(payload);
      const incoming = service.normaliseMediRecordsAppointments(rawAppointments);
      if (!incoming.length) throw new Error("No appointments found in the pasted data.");
      appointments = service.mergeAppointments(appointments, incoming);
      saveNow();
      renderAppointments();
      setStatus(`Imported ${incoming.length} appointment${incoming.length === 1 ? "" : "s"}.`);
      closeImportModal();
    } catch (error) {
      if (message) message.textContent = error.message || "Could not import appointments.";
    }
  }

  async function syncAppointments() {
    const syncButton = document.getElementById("appointmentSyncBtn");
    if (syncButton) syncButton.disabled = true;
    setStatus("Syncing appointments...");

    try {
      const incoming = await service.fetchTodaysAppointmentsFromMediRecords();
      appointments = service.mergeAppointments(appointments, incoming);
      saveNow();
      renderAppointments();
      setStatus(`Synced ${incoming.length} appointment${incoming.length === 1 ? "" : "s"}.`);
    } catch {
      setStatus("MediRecords sync is not connected yet. Use Import for now.");
    } finally {
      if (syncButton) syncButton.disabled = false;
    }
  }

  function bindAppointmentUI() {
    document.getElementById("appointmentImportBtn")?.addEventListener("click", openImportModal);
    document.getElementById("appointmentImportClose")?.addEventListener("click", closeImportModal);
    document.getElementById("appointmentImportSubmit")?.addEventListener("click", importAppointmentsFromText);
    document.getElementById("appointmentSyncBtn")?.addEventListener("click", syncAppointments);
    document.getElementById("appointmentImportModal")?.addEventListener("click", (event) => {
      if (event.target.id === "appointmentImportModal") closeImportModal();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeImportModal();
    });
  }

  function initAppointmentsSection() {
    installClinicalDocumentationStandards();
    applyHeaderBranding();
    if (!storage || !service || !document.getElementById("appointmentsSection")) return;
    appointments = storage.loadAppointmentsFromStorage();
    bindAppointmentUI();
    renderAppointments();
    if (visibleAppointments().length) {
      setStatus(`${visibleAppointments().length} saved appointment${visibleAppointments().length === 1 ? "" : "s"}.`);
    } else {
      setStatus("Use Import to paste MediRecords JSON. Sync works once the backend endpoint is connected.");
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAppointmentsSection);
  } else {
    initAppointmentsSection();
  }

  window.VividMediAppointmentsUI = {
    renderAppointments,
    syncAppointments,
    importAppointmentsFromText,
    analyseAppointmentNote
  };
})();

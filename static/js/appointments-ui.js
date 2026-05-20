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

    document.getElementById("consultNotesTab")?.click();
    clinicalInput.value = [
      `Patient: ${appointment.patientName || "Unknown patient"}`,
      `Appointment type: ${appointment.appointmentType || "-"}`,
      `WA time: ${appointment.startTimeWA || "-"}`,
      "",
      note
    ].join("\n");
    clinicalInput.dispatchEvent(new Event("input", { bubbles: true }));
    setStatus("Appointment notes copied to the analyser.");
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
    row.appendChild(makeEl("span", "", ">"));
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
    toggle.addEventListener("click", () => toggleAppointment(appointment.appointmentGuid));
    toggle.appendChild(makeEl("span", "appointment-card-time", appointment.startTimeWA || ""));
    toggle.appendChild(makeEl("strong", "appointment-card-name", appointment.patientName || "Unknown patient"));

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
    if (!storage || !service || !document.getElementById("appointmentsSection")) return;
    appointments = storage.loadAppointmentsFromStorage();
    bindAppointmentUI();
    renderAppointments();
    if (visibleAppointments().length) {
      setStatus(`${visibleAppointments().length} saved appointment${visibleAppointments().length === 1 ? "" : "s"}.`);
    }
    syncAppointments();
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

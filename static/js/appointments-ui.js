(function () {
  const storage = window.VividMediAppointmentStorage;
  const service = window.VividMediAppointmentService;

  if (!storage || !service) {
    console.warn("Appointment dependencies are not available");
    return;
  }

  let appointments = [];
  let saveTimer = null;

  function byId(id) {
    return document.getElementById(id);
  }

  function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value === null || value === undefined ? "" : String(value);
    return div.innerHTML;
  }

  function displayValue(value) {
    return value === null || value === undefined || value === ""
      ? "&mdash;"
      : escapeHtml(value);
  }

  function parseImportText(text) {
    const trimmed = (text || "").trim();
    if (!trimmed) throw new Error("Paste appointment JSON first.");

    try {
      return JSON.parse(trimmed);
    } catch {
      const objectStart = trimmed.indexOf("{");
      const arrayStart = trimmed.indexOf("[");
      const firstStart = [objectStart, arrayStart]
        .filter((index) => index >= 0)
        .sort((a, b) => a - b)[0];

      if (firstStart === undefined) {
        throw new Error("Could not find JSON in the pasted text.");
      }

      const objectEnd = trimmed.lastIndexOf("}");
      const arrayEnd = trimmed.lastIndexOf("]");
      const lastEnd = Math.max(objectEnd, arrayEnd);

      if (lastEnd <= firstStart) {
        throw new Error("Could not find complete JSON in the pasted text.");
      }

      return JSON.parse(trimmed.slice(firstStart, lastEnd + 1));
    }
  }

  function visibleAppointments() {
    return appointments.filter((appointment) => !appointment.isDeleted);
  }

  function saveAppointmentsNow() {
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveTimer = null;
    }

    storage.saveAppointmentsToStorage(appointments);
    document.querySelectorAll(".appointment-save-status").forEach((status) => {
      status.innerHTML = "&#10003; Saved";
    });
  }

  function scheduleSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(saveAppointmentsNow, 500);
  }

  function setStatus(message, tone) {
    const status = byId("appointmentsSyncStatus");
    if (!status) return;

    status.textContent = message || "";
    status.classList.toggle("warning", tone === "warning");
  }

  function setAppointments(nextAppointments, options) {
    const opts = options || {};
    appointments = service.sortAppointments(nextAppointments || []);
    renderAppointments();

    if (opts.persist === "immediate") {
      saveAppointmentsNow();
    } else if (opts.persist === "debounced") {
      scheduleSave();
    }
  }

  function getAppointment(guid) {
    return appointments.find((appointment) => appointment.appointmentGuid === guid);
  }

  function updateAppointment(guid, updater, options) {
    const next = appointments.map((appointment) => {
      if (appointment.appointmentGuid !== guid) return appointment;
      return updater(appointment);
    });

    setAppointments(next, options);
  }

  function renderAppointmentCard(appointment) {
    const guid = escapeHtml(appointment.appointmentGuid);
    const compactTime = displayValue(appointment.startTimeWACompact || appointment.startTimeWA);
    const patientName = displayValue(appointment.patientName);
    const fullTime = displayValue(appointment.startTimeWA);
    const fullTimeWithZone = appointment.startTimeWA
      ? `${escapeHtml(appointment.startTimeWA)} AWST`
      : "&mdash;";

    if (!appointment.isExpanded) {
      return `
        <div class="appointment-card-shell" data-appointment-guid="${guid}">
          <button type="button" class="appointment-row" data-appointment-toggle>
            <span class="appointment-time">${compactTime}</span>
            <span class="appointment-patient">${patientName}</span>
            <span class="appointment-chevron" aria-hidden="true">&gt;</span>
          </button>
        </div>
      `;
    }

    const safeInputId = `appointment-note-${String(appointment.appointmentGuid).replace(/[^a-z0-9_-]/gi, "-")}`;

    return `
      <div class="appointment-card expanded" data-appointment-guid="${guid}">
        <div class="appointment-card-top">
          <button type="button" class="appointment-card-summary" data-appointment-toggle>
            <span>${fullTime}</span>
            <strong>${patientName}</strong>
          </button>
          <button
            type="button"
            class="appointment-delete"
            data-appointment-delete
            aria-label="Delete appointment locally"
          >&times;</button>
        </div>

        <div class="appointment-details">
          <div><span>Type</span><span>${displayValue(appointment.appointmentType)}</span></div>
          <div><span>Age</span><span>${appointment.age || appointment.age === 0 ? escapeHtml(appointment.age) : "&mdash;"}</span></div>
          <div><span>Mobile</span><span>${displayValue(appointment.mobilePhone)}</span></div>
          <div><span>WA Time</span><span>${fullTimeWithZone}</span></div>
        </div>

        ${appointment.missingFromLatestSync ? '<div class="appointment-sync-chip">Not in latest sync</div>' : ""}

        <label class="appointment-note-label" for="${escapeHtml(safeInputId)}">Consult notes</label>
        <textarea
          id="${escapeHtml(safeInputId)}"
          data-appointment-note
          placeholder="Paste or type consult notes here..."
        >${escapeHtml(appointment.consultNote || "")}</textarea>

        <div class="appointment-save-status">&#10003; Saved</div>
      </div>
    `;
  }

  function bindAppointmentListEvents(list) {
    list.querySelectorAll("[data-appointment-toggle]").forEach((toggle) => {
      toggle.addEventListener("click", () => {
        const wrapper = toggle.closest("[data-appointment-guid]");
        if (!wrapper) return;

        updateAppointment(
          wrapper.dataset.appointmentGuid,
          (appointment) => ({
            ...appointment,
            isExpanded: !appointment.isExpanded,
            lastSavedAt: new Date().toISOString()
          }),
          { persist: "immediate" }
        );
      });
    });

    list.querySelectorAll("[data-appointment-delete]").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        const wrapper = button.closest("[data-appointment-guid]");
        if (!wrapper) return;

        updateAppointment(
          wrapper.dataset.appointmentGuid,
          (appointment) => ({
            ...appointment,
            isDeleted: true,
            isExpanded: false,
            lastSavedAt: new Date().toISOString()
          }),
          { persist: "immediate" }
        );
      });
    });

    list.querySelectorAll("[data-appointment-note]").forEach((textarea) => {
      textarea.addEventListener("input", () => {
        const wrapper = textarea.closest("[data-appointment-guid]");
        if (!wrapper) return;

        const appointment = getAppointment(wrapper.dataset.appointmentGuid);
        if (!appointment) return;

        appointment.consultNote = textarea.value;
        appointment.lastSavedAt = new Date().toISOString();

        const status = wrapper.querySelector(".appointment-save-status");
        if (status) status.textContent = "Saving...";
        scheduleSave();
      });
    });
  }

  function renderAppointments() {
    const list = byId("appointmentsList");
    if (!list) return;

    const visible = visibleAppointments();
    if (!visible.length) {
      list.innerHTML = '<div class="empty-state">No appointments loaded</div>';
      return;
    }

    list.innerHTML = visible.map(renderAppointmentCard).join("");
    bindAppointmentListEvents(list);
  }

  async function syncAppointments(options) {
    const opts = options || {};
    if (!opts.silent) setStatus("Syncing appointments...");

    try {
      const incoming = await service.fetchTodaysAppointmentsFromMediRecords();
      setAppointments(service.mergeAppointments(appointments, incoming), { persist: "immediate" });
      setStatus(`Synced ${incoming.length} appointment${incoming.length === 1 ? "" : "s"} today`);
    } catch (error) {
      if (!opts.silent) {
        setStatus("MediRecords feed unavailable. Import JSON to load appointments.", "warning");
      } else if (!visibleAppointments().length) {
        setStatus("Import JSON to load appointments");
      }

      console.info("MediRecords appointments sync unavailable", error);
    }
  }

  function openImportModal() {
    const modal = byId("appointmentImportModal");
    const textarea = byId("appointmentImportJson");
    const error = byId("appointmentImportError");

    if (!modal) return;
    modal.hidden = false;
    if (error) error.textContent = "";
    if (textarea) {
      textarea.value = "";
      textarea.focus();
    }
  }

  function closeImportModal() {
    const modal = byId("appointmentImportModal");
    if (modal) modal.hidden = true;
  }

  function handleImport() {
    const textarea = byId("appointmentImportJson");
    const error = byId("appointmentImportError");

    try {
      const payload = parseImportText(textarea ? textarea.value : "");
      const incoming = service.normaliseMediRecordsAppointments(payload);

      if (!incoming.length) {
        throw new Error("No appointments were found in that JSON.");
      }

      setAppointments(service.mergeAppointments(appointments, incoming), { persist: "immediate" });
      setStatus(`Imported ${incoming.length} appointment${incoming.length === 1 ? "" : "s"}`);
      closeImportModal();
    } catch (importError) {
      if (error) error.textContent = importError.message;
    }
  }

  function bindImportModal() {
    const openButton = byId("openAppointmentImportBtn");
    const closeButton = byId("closeAppointmentImportBtn");
    const cancelButton = byId("cancelAppointmentImportBtn");
    const submitButton = byId("submitAppointmentImportBtn");
    const modal = byId("appointmentImportModal");
    const syncButton = byId("syncAppointmentsBtn");

    if (openButton) openButton.addEventListener("click", openImportModal);
    if (closeButton) closeButton.addEventListener("click", closeImportModal);
    if (cancelButton) cancelButton.addEventListener("click", closeImportModal);
    if (submitButton) submitButton.addEventListener("click", handleImport);
    if (syncButton) syncButton.addEventListener("click", () => syncAppointments());

    if (modal) {
      modal.addEventListener("click", (event) => {
        if (event.target === modal) closeImportModal();
      });
    }

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && modal && !modal.hidden) closeImportModal();
    });
  }

  function initAppointments() {
    appointments = service.sortAppointments(storage.loadAppointmentsFromStorage());
    renderAppointments();
    bindImportModal();
    syncAppointments({ silent: true });
  }

  window.addEventListener("beforeunload", saveAppointmentsNow);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAppointments);
  } else {
    initAppointments();
  }
})();

(function () {
  const storage = window.VividMediAppointmentStorage;
  const service = window.VividMediAppointmentsService;
  let appointments = [];
  let saveTimer = null;
  let activeAppointmentGuid = null;

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
              "- Plain text only. Do not use Markdown, asterisks, bold markers, or decorative symbols in the final note.",
              "- Use plain section heading lines only. The app will visually bold headings.",
              "- Optimise the output within that selected type for Australian medical documentation standards.",
              "- Make the note clinically robust, concise, defensible, and useful for continuity of care.",
              "- Preserve documented facts, important positives and negatives, uncertainty, risks, medication details, contraindications, monitoring needs, and follow-up needs where they are clinically relevant.",
              "- Include Monitoring, Follow-up, Safety Netting, and Red Flags only when clinically relevant to the selected consult type, patient risk, medications or procedures, diagnostic uncertainty, or documented clinician concern.",
              "- Do not force safety-netting or red-flag sections into low-risk administrative, renewal, script, referral, or documentation-only notes unless clinically warranted.",
              "- Where the selected type or content relates to DVA, allied health, renewal, veteran care, weight management scripts, or referral justification, write to an audit-ready DVA documentation standard without inventing accepted conditions or entitlement details.",
              "- Write 'Not documented' where clinically important information is missing.",
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

  const CLINICAL_OUTPUT_HEADINGS = new Set([
    "summary",
    "assessment",
    "diagnosis",
    "diagnoses",
    "investigations",
    "treatment",
    "plan",
    "management plan",
    "monitoring",
    "follow-up",
    "follow up",
    "follow-up & safety netting",
    "follow up & safety netting",
    "safety netting",
    "red flags",
    "references",
    "history",
    "examination",
    "medications",
    "allergies",
    "impression",
    "rationale",
    "dva rationale",
    "referral rationale",
    "audit readiness",
    "missing information"
  ]);

  function cleanClinicalOutputText(value) {
    return String(value || "")
      .replace(/\r\n/g, "\n")
      .replace(/^\s*\*\s+/gm, "- ")
      .replace(/\*\*([^\n]+?)\*\*/g, "$1")
      .replace(/\*/g, "")
      .replace(/[ \t]+$/gm, "")
      .split("\n")
      .filter((line) => {
        const trimmed = line.trim();
        return !/^here is (the )?(structured )?clinical note/i.test(trimmed)
          && !/^based on (the )?(raw dictation|provided)/i.test(trimmed);
      })
      .join("\n")
      .trim();
  }

  function isClinicalHeading(line) {
    const normalized = String(line || "")
      .trim()
      .replace(/:$/, "")
      .toLowerCase();
    return CLINICAL_OUTPUT_HEADINGS.has(normalized);
  }

  function renderClinicalOutputElement(output) {
    if (!output || output.dataset.vividFormatting === "1") return;
    const current = output.textContent || "";
    const trimmed = current.trim();
    if (!trimmed || trimmed === "Generating..." || trimmed === "Output will appear here..." || trimmed === "Answer will appear here...") return;

    const cleaned = cleanClinicalOutputText(current);
    if (!cleaned || output.dataset.vividFormattedSource === cleaned) return;

    output.dataset.vividFormatting = "1";
    output.dataset.vividFormattedSource = cleaned;
    clearNode(output);

    const fragment = document.createDocumentFragment();
    cleaned.split("\n").forEach((line, index, lines) => {
      const row = document.createElement("div");
      row.className = isClinicalHeading(line) ? "clinical-output-heading" : "clinical-output-line";
      row.textContent = line;
      fragment.appendChild(row);
      if (index < lines.length - 1) fragment.appendChild(document.createTextNode("\n"));
    });
    output.appendChild(fragment);
    output.dataset.vividFormatting = "0";
  }

  function installClinicalOutputFormatter() {
    if (window.__vividClinicalOutputFormatterInstalled) return;
    window.__vividClinicalOutputFormatterInstalled = true;

    if (!document.getElementById("vivid-clinical-output-style")) {
      const style = document.createElement("style");
      style.id = "vivid-clinical-output-style";
      style.textContent = [
        ".clinical-output-heading{font-weight:700;margin:10px 0 4px;}",
        ".clinical-output-line{font-weight:400;margin:0;white-space:pre-wrap;}",
        ".clinical-output-line:empty{min-height:.85em;}"
      ].join("");
      document.head.appendChild(style);
    }

    const attach = () => {
      const output = document.getElementById("outputBox");
      if (output) {
        renderClinicalOutputElement(output);
        new MutationObserver(() => renderClinicalOutputElement(output)).observe(output, {
          childList: true,
          characterData: true,
          subtree: true
        });
      }

      const savedNotes = document.getElementById("savedNotesList");
      if (savedNotes) {
        savedNotes.querySelectorAll(".saved-note-text").forEach(renderClinicalOutputElement);
        new MutationObserver(() => {
          savedNotes.querySelectorAll(".saved-note-text").forEach(renderClinicalOutputElement);
        }).observe(savedNotes, { childList: true, subtree: true });
      }
    };

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", attach, { once: true });
    } else {
      attach();
    }
  }

  function appointmentById(appointmentGuid) {
    return appointments.find((appointment) => appointment.appointmentGuid === appointmentGuid);
  }

  function toggleAppointment(appointmentGuid) {
    const appointment = appointmentById(appointmentGuid);
    if (!appointment) return;
    activeAppointmentGuid = appointmentGuid;
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
    activeAppointmentGuid = appointmentGuid;

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
    const weightManagement = appointment.weightManagement || {};
    const medicationLines = formatMedicationHistory(weightManagement.medicationHistory);
    const trendLines = formatWeightTrend(weightManagement.trend);
    document.getElementById("consultNotesTab")?.click();
    clinicalInput.value = [
      `Patient: ${appointment.patientName || "Unknown patient"}`,
      `Age: ${appointment.age === null || appointment.age === undefined ? "Not documented" : appointment.age}`,
      `DOB: ${formatDobFromAppointment(appointment) || "Not documented"}`,
      `Mobile: ${appointment.mobilePhone || "Not documented"}`,
      `DVA number: ${appointment.dvaNo || "Not documented"}`,
      `DVA card: ${appointment.dvaCardColour || "Not documented"}`,
      `Accepted DVA conditions: ${formatAcceptedConditions(appointment.acceptedConditions) || "Not documented"}`,
      `MediRecords appointment type: ${appointment.appointmentType || "Not documented"}`,
      `WA appointment time: ${appointment.startTimeWA || "Not documented"}`,
      "",
      "Weight management snapshot:",
      `Most recent medication: ${formatLatestMedication(weightManagement) || "Not documented"}`,
      `Medication history: ${medicationLines || "Not documented"}`,
      `Last documented weight: ${weightManagement.latestWeight || "Not documented"}`,
      `Last documented BMI: ${weightManagement.latestBmi || "Not documented"}`,
      `Weight/BMI trend: ${trendLines || "Not documented"}`,
      "",
      "Clinical notes / dictation:",
      note
    ].join("\n");
    clinicalInput.dispatchEvent(new Event("input", { bubbles: true }));
    setStatus(`Appointment notes sent for analysis using your selected consult type: ${selectedConsultType}.`);
    document.getElementById("convertBtn")?.click();
  }

  function getActiveAppointment() {
    if (activeAppointmentGuid) return appointmentById(activeAppointmentGuid);
    return appointments.find((appointment) => appointment.isExpanded && !appointment.isDeleted) || null;
  }

  function completeAppointment(appointmentGuid) {
    const appointment = appointmentById(appointmentGuid);
    if (!appointment) return false;
    appointment.isDeleted = true;
    appointment.isExpanded = false;
    appointment.isDone = true;
    appointment.lastSavedAt = new Date().toISOString();
    if (activeAppointmentGuid === appointmentGuid) activeAppointmentGuid = null;
    saveNow();
    renderAppointments();
    setStatus("Completed appointment moved to Saved Outputs.");
    return true;
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

  function formatDobFromAppointment(appointment) {
    const dob = appointment.dob || appointment.dateOfBirth;
    if (!dob) return "";
    const date = new Date(Number(dob));
    if (Number.isNaN(date.getTime())) return String(dob);
    return new Intl.DateTimeFormat("en-AU", {
      timeZone: "Australia/Perth",
      day: "2-digit",
      month: "2-digit",
      year: "numeric"
    }).format(date);
  }

  function formatAcceptedConditions(value) {
    return Array.isArray(value) ? value.filter(Boolean).join("; ") : String(value || "").trim();
  }

  function formatMedicationItem(item) {
    if (!item) return "";
    if (typeof item === "string") return item.trim();
    const name = item.name || item.medication || item.drug || item.text || "";
    const dose = item.dose || item.strength || "";
    const date = item.date || item.documentedAt || item.noteDate || "";
    return [name, dose, date ? `(${date})` : ""].map((part) => String(part || "").trim()).filter(Boolean).join(" ");
  }

  function formatMedicationHistory(value) {
    return Array.isArray(value) ? value.map(formatMedicationItem).filter(Boolean).join("; ") : "";
  }

  function formatLatestMedication(weightManagement) {
    if (!weightManagement) return "";
    const latest = weightManagement.latestMedication || weightManagement.currentMedication;
    if (latest) return formatMedicationItem(latest);
    return Array.isArray(weightManagement.medicationHistory) && weightManagement.medicationHistory.length
      ? formatMedicationItem(weightManagement.medicationHistory[0])
      : "";
  }

  function formatTrendItem(item) {
    if (!item) return "";
    if (typeof item === "string") return item.trim();
    const date = item.date || item.documentedAt || item.noteDate || "";
    const weight = String(item.weight || item.weightKg || "").replace(/\s*kg\b/i, "");
    const bmi = item.bmi || item.BMI || "";
    if (item.text) return item.text;
    return [
      date,
      weight ? `${weight} kg` : "",
      bmi ? `BMI ${bmi}` : ""
    ].map((part) => String(part || "").trim()).filter(Boolean).join(" - ");
  }

  function formatWeightTrend(value) {
    return Array.isArray(value) ? value.map(formatTrendItem).filter(Boolean).join(" -> ") : "";
  }

  function formatLatestWeight(value) {
    return String(value || "").replace(/\s*kg\b/i, "").trim();
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
    details.appendChild(renderDetail("DVA", [appointment.dvaCardColour, appointment.dvaNo].filter(Boolean).join(" ")));
    details.appendChild(renderDetail("Conds", formatAcceptedConditions(appointment.acceptedConditions)));
    details.appendChild(renderDetail("Med", formatLatestMedication(appointment.weightManagement || {})));
    details.appendChild(renderDetail("Wt/BMI", [
      appointment.weightManagement?.latestWeight ? `${formatLatestWeight(appointment.weightManagement.latestWeight)} kg` : "",
      appointment.weightManagement?.latestBmi ? `BMI ${appointment.weightManagement.latestBmi}` : ""
    ].filter(Boolean).join(" / ")));
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
      const incoming = await fetchLatestExtensionSync();
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

  async function fetchLatestExtensionSync() {
    try {
      const response = await fetch("/api/medirecords-sync/latest", {
        method: "GET",
        credentials: "same-origin",
        headers: { "Accept": "application/json" }
      });
      if (response.ok) {
        const data = await response.json();
        const payload = data.payload || data;
        let rawAppointments = [];
        try {
          rawAppointments = service.extractAppointmentArrayFromPayload(payload);
        } catch {
          rawAppointments = [];
        }
        let incoming = rawAppointments.length
          ? service.normaliseMediRecordsAppointments(rawAppointments)
          : service.appointmentsFromPatientSnapshots(payload.patients || payload.patientSnapshots || []);
        incoming = service.mergePatientSnapshots(incoming, payload.patients || payload.patientSnapshots || []);
        return incoming;
      }
    } catch {
      // Fall back to the legacy direct backend fetch below.
    }
    return service.fetchTodaysAppointmentsFromMediRecords();
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
    installClinicalOutputFormatter();
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
    analyseAppointmentNote,
    getActiveAppointment,
    completeAppointment
  };
})();

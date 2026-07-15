(function () {
  "use strict";

  const CONSULT_TYPE = "ED MH Review";
  const DRAFT_PREFIX = "vivid_ed_mh_review_draft:";
  const LAST_TEAM_KEY = "vivid_ed_mh_review_last_team";

  const ROLE_NAMES = {
    Consultant: ["Dr Munib", "Dr Garside", "Dr Dias", "Dr Hussein"],
    Registrar: [
      "Dr Claasen", "Dr Burrows", "Dr Addis", "Dr Sullivan", "Dr Greenall", "Dr Tone",
      "Dr Lockhart", "Dr Vasani", "Dr Waltman", "Dr Sivakumar", "Dr Lau", "Dr Chong",
      "Dr Maddams", "Dr Warlow", "Dr Tudori"
    ],
    PLN: [
      "Dan", "Simon", "Neil", "Ruth", "Tressa", "Jon", "Anna-Jade", "Sorie", "Sami",
      "Allan", "Tendai", "Elaine", "Victoria", "Ross"
    ],
    RMO: [],
    Intern: [],
    Other: []
  };

  const MHA_FORMS = [
    {
      group: "Referral for examination / detention",
      forms: [
        ["Form 1A", "Referral for examination by a psychiatrist"],
        ["Form 1A attachment", "Information provided by another person in confidence"],
        ["Form 1B", "Variation of referral"],
        ["Form 2", "Order to detain voluntary inpatient in authorised hospital for assessment"],
        ["Form 3A", "Detention order"],
        ["Form 3B", "Continuation of detention"],
        ["Form 3C", "Continuation of detention to enable a further examination by a psychiatrist"],
        ["Form 3D", "Order authorising reception and detention in an authorised hospital for further examination"],
        ["Form 3E", "Order that person cannot continue to be detained"]
      ]
    },
    {
      group: "Transport / transfer orders",
      forms: [
        ["Form 4A", "Transport order"],
        ["Form 4B", "Extension of transport order"],
        ["Form 4C", "Transfer order"],
        ["Form 4D", "Interstate transfer order (currently unavailable)", true],
        ["Form 4E", "Approval of interstate transfer order (currently unavailable)", true]
      ]
    },
    {
      group: "Community treatment orders",
      forms: [
        ["Form 5A", "Community Treatment Order"],
        ["Form 5B", "Continuation of Community Treatment Order"],
        ["Form 5C", "Variation of terms of Community Treatment Order"],
        ["Form 5D", "Request for practitioner to conduct monthly examination"],
        ["Form 5E", "Notice and record of breach of Community Treatment Order"],
        ["Form 5F", "Order to attend"]
      ]
    },
    {
      group: "Inpatient treatment / leave / search",
      forms: [
        ["Form 6A", "Inpatient treatment order in authorised hospital"],
        ["Form 6B", "Inpatient treatment order in general hospital"],
        ["Form 6B attachment", "General hospital inpatient treatment order report to Chief Psychiatrist"],
        ["Form 6C", "Continuation of inpatient treatment order"],
        ["Form 6D", "Confirmation of inpatient treatment order"],
        ["Form 7A", "Grant of leave to involuntary inpatient"],
        ["Form 7B", "Extension and/or variation of leave"],
        ["Form 7C", "Cancellation of grant of leave"],
        ["Form 7D", "Apprehension and return order"],
        ["Form 8A", "Record of search and seizure"],
        ["Form 8B", "Record dealing with seized article"]
      ]
    },
    {
      group: "Treatments",
      forms: [
        ["Form 9A", "Record of emergency psychiatric treatment"],
        ["Form 9B", "Report about provision of urgent non-psychiatric treatment"]
      ]
    },
    {
      group: "Bodily restraint",
      forms: [
        ["Form 10A", "Record of oral authorisation of bodily restraint"],
        ["Form 10B", "Written bodily restraint order"],
        ["Form 10C", "Record of informing medical practitioner and treating psychiatrist of bodily restraint"],
        ["Form 10D", "Record of observations made of restrained person"],
        ["Form 10E", "Record of examination of restrained person and possible extension"],
        ["Form 10F", "Variation of bodily restraint order"],
        ["Form 10G", "Revocation or expiry of bodily restraint order"],
        ["Form 10H", "Review of bodily restraint order by a psychiatrist"],
        ["Form 10I", "Record of post-bodily restraint examination"]
      ]
    },
    {
      group: "Seclusion",
      forms: [
        ["Form 11A", "Record of oral authorisation of seclusion"],
        ["Form 11B", "Written seclusion order"],
        ["Form 11C", "Record of informing medical practitioner and treating psychiatrist of seclusion"],
        ["Form 11D", "Record of observations made of secluded person"],
        ["Form 11E", "Record of examination of secluded person and possible extension"],
        ["Form 11F", "Revocation or expiry of seclusion order"],
        ["Form 11G", "Record of post-seclusion examination"]
      ]
    },
    {
      group: "Access to information / ECT",
      forms: [
        ["Form 12A", "Nomination of nominated person"],
        ["Form 12B", "Record of refusal of patient's request to access document"],
        ["Form 12C", "Restriction on freedom of communication"],
        ["Form 12C attachment", "Confirmation, amendment or revocation of communication restriction"],
        ["Form 13", "Statistics about ECT"]
      ]
    }
  ];

  const MSE_OPTIONS = {
    appearance: [
      "Well-groomed and appropriately dressed", "Adequately groomed", "Dishevelled",
      "Poor hygiene / self-care", "Inappropriately dressed", "Appears stated age"
    ],
    behaviour: [
      "Calm, cooperative and engaged", "Guarded", "Withdrawn", "Restless / agitated",
      "Psychomotor retardation", "Hostile", "Disinhibited", "Responding to internal stimuli",
      "Appropriate eye contact", "Limited eye contact"
    ],
    speech: [
      "Normal rate, volume and prosody", "Pressured", "Rapid", "Slowed", "Loud", "Soft",
      "Reduced spontaneity / poverty of speech", "Increased latency", "Mute"
    ],
    mood: [
      "Euthymic", "Depressed", "Anxious", "Irritable", "Angry", "Elevated", "Subjectively low",
      "Subjectively improved"
    ],
    affect: [
      "Reactive and congruent with stated mood", "Restricted", "Blunted", "Flat", "Labile",
      "Anxious", "Irritable", "Incongruent with stated mood"
    ],
    thought_form: [
      "Linear, logical and goal-directed", "Circumstantial", "Tangential", "Disorganised",
      "Flight of ideas", "Loosening of associations", "Perseverative", "Thought blocking"
    ],
    thought_content: [
      "No delusional content elicited", "Persecutory ideas / delusions", "Grandiose ideas / delusions",
      "Ideas of reference", "Hopelessness / guilt", "Obsessions", "Preoccupied",
      "Overvalued ideas"
    ],
    perception: [
      "No perceptual disturbance reported or observed", "Auditory hallucinations", "Visual hallucinations",
      "Tactile hallucinations", "Olfactory / gustatory hallucinations", "Illusions",
      "Derealisation / depersonalisation", "Appears to respond to internal stimuli"
    ],
    cognition: [
      "Alert and oriented to person, place and time", "Orientation impaired", "Attention and concentration intact",
      "Attention / concentration impaired", "Memory grossly intact", "Memory impaired",
      "Fluctuating level of consciousness / possible delirium", "Cognition not formally assessed"
    ],
    tosh: [
      "Denies current thoughts of self-harm", "Current thoughts of self-harm without stated plan",
      "Current thoughts of self-harm with plan", "Recent thoughts / acts of self-harm",
      "Past self-harm history", "Unable to assess"
    ],
    si: [
      "Denies current suicidal ideation", "Passive death wish", "Suicidal ideation without stated plan",
      "Suicidal ideation with plan but no stated intent", "Suicidal ideation with plan and intent",
      "Unable to assess"
    ],
    insight_judgement: [
      "Good insight and intact judgement", "Fair insight and judgement", "Partial insight",
      "Limited insight", "Absent insight", "Judgement impaired", "Insight / judgement not formally assessed"
    ]
  };

  const SOURCE_OPTIONS = ["Patient", "Nursing", "Family", "Records", "Allied health", "Other collateral"];

  const SECTIONS = [
    { id: "review_details", title: "ED Review Psychiatry / Current" },
    {
      id: "summary",
      title: "Summary",
      help: "Paste the presenting complaint, then add or dictate the current summary.",
      fields: [{ key: "summary", label: "Summary", rows: 5, placeholder: "Presenting complaint and concise current summary..." }]
    },
    { id: "review", title: "Review", help: "Record who provided the information and keep sources clearly attributed." },
    {
      id: "progress",
      title: "Progress",
      fields: [
        { key: "behaviour_events", label: "Behaviour and significant events since last review" },
        { key: "sleep_intake_adls", label: "Sleep / intake / ADLs", placeholder: "Hours slept; eating and drinking; ADLs..." },
        { key: "medication", label: "Medication adherence, PRNs and adverse effects" },
        { key: "engagement", label: "Engagement with treatment" },
        { key: "progress_notes", label: "Other progress information" }
      ]
    },
    {
      id: "patient_account",
      title: "Patient's account of progress",
      help: "Configure text may carry forward earlier information only when the timing supports that it remains current.",
      fields: [
        { key: "current_symptoms", label: "Current symptoms" },
        { key: "concerns_goals", label: "Concerns, requests and goals" },
        { key: "understanding", label: "Understanding of admission and treatment" }
      ]
    },
    {
      id: "mse",
      title: "MSE",
      help: "Choose a concise phrase or use Other / free type. Configure text only fills findings supported by the review."
    },
    {
      id: "assessment",
      title: "ASSESSMENT",
      help: "Spell check / organise identifies supported information from the review and places it in the correct field.",
      fields: [
        { key: "clinical_progress", label: "Clinical progress" },
        { key: "working_diagnosis", label: "Working diagnosis" },
        { key: "response_management", label: "Response to management" }
      ]
    },
    {
      id: "risk",
      title: "Current risk formulation and management",
      help: "One concise formulation only. Include supported current risks, relevant factors and management without repeating the review.",
      fields: [
        { key: "risk_formulation_management", label: "Risk formulation and management", rows: 7 }
      ]
    },
    {
      id: "plan",
      title: "PLAN",
      help: "Enter one action per line. The finished output is numbered automatically.",
      fields: [{ key: "plan", label: "Plan", rows: 8, placeholder: "One plan item per line..." }]
    }
  ];

  const FIELD_LABELS = {
    summary: "Summary",
    sources: "Information sources",
    updates: "Updates",
    behaviour_events: "Behaviour and significant events since last review",
    sleep_intake_adls: "Sleep / intake / ADLs",
    medication: "Medication adherence, PRNs and adverse effects",
    engagement: "Engagement with treatment",
    progress_notes: "Other progress information",
    current_symptoms: "Current symptoms",
    concerns_goals: "Concerns, requests and goals",
    understanding: "Understanding of admission and treatment",
    appearance: "Appearance",
    behaviour: "Behaviour",
    speech: "Speech",
    mood: "Mood",
    affect: "Affect",
    thought_form: "Thought form",
    thought_content: "Thought content",
    perception: "Perception",
    cognition: "Cognition",
    tosh: "TOSH",
    si: "SI",
    insight_judgement: "Insight / judgement",
    risk_formulation_management: "Risk formulation and management",
    clinical_progress: "Clinical progress",
    working_diagnosis: "Working diagnosis",
    response_management: "Response to management",
    plan: "Plan"
  };

  const workspace = document.getElementById("edMhReviewWorkspace");
  const standardWorkspace = document.getElementById("standardConsultWorkspace");
  const consultSelect = document.getElementById("consultType");
  if (!workspace || !standardWorkspace || !consultSelect) return;

  let state = createDefaultState();
  let activeDraftKey = "";
  let saveTimer = null;
  let suppressAppointmentReload = false;

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;"
    })[char]);
  }

  function createDefaultState() {
    const sections = {};
    SECTIONS.forEach((section) => {
      const fields = {};
      (section.fields || []).forEach((field) => { fields[field.key] = ""; });
      if (section.id === "review") {
        fields.sources = [];
        fields.updates = "";
      }
      if (section.id === "mse") {
        Object.keys(MSE_OPTIONS).forEach((key) => { fields[key] = ""; });
      }
      sections[section.id] = { complete: false, fields, narrative: "" };
    });
    return {
      version: 1,
      patientIdentifier: "",
      team: [],
      legalStatus: "",
      mhaForm: "",
      mhaExpiry: "",
      sections,
      output: "",
      savedAt: ""
    };
  }

  function normaliseState(raw) {
    const clean = createDefaultState();
    if (!raw || typeof raw !== "object") return clean;
    clean.patientIdentifier = String(raw.patientIdentifier || "").slice(0, 500);
    clean.legalStatus = ["voluntary", "mha"].includes(raw.legalStatus) ? raw.legalStatus : "";
    clean.mhaForm = String(raw.mhaForm || "").slice(0, 500);
    clean.mhaExpiry = String(raw.mhaExpiry || "").slice(0, 100);
    clean.output = String(raw.output || "").slice(0, 100000);
    clean.savedAt = String(raw.savedAt || "").slice(0, 100);
    clean.team = normaliseTeam(raw.team);
    SECTIONS.forEach((section) => {
      const incoming = raw.sections && raw.sections[section.id];
      if (!incoming || typeof incoming !== "object") return;
      const incomingFields = incoming.fields && typeof incoming.fields === "object" ? incoming.fields : {};
      clean.sections[section.id].complete = Boolean(incoming.complete);
      clean.sections[section.id].narrative = String(incoming.narrative || "").slice(0, 30000);
      Object.keys(clean.sections[section.id].fields).forEach((key) => {
        if (key === "sources") {
          clean.sections[section.id].fields[key] = Array.isArray(incomingFields[key])
            ? incomingFields[key].filter((item) => SOURCE_OPTIONS.includes(item)).slice(0, SOURCE_OPTIONS.length)
            : [];
        } else {
          let value = incomingFields[key];
          if (section.id === "assessment" && key === "response_management" && !value) {
            value = incomingFields.response_tolerability;
          }
          if (section.id === "risk" && key === "risk_formulation_management" && !value) {
            value = [
              ["Suicide / self-harm", incomingFields.suicide_self_harm],
              ["Violence / aggression", incomingFields.violence_aggression],
              ["Self-neglect / physical issues", incomingFields.self_neglect_physical],
              ["AWOL / vulnerability / risk from others", incomingFields.awol_vulnerability],
              ["Changes in dynamic factors", incomingFields.dynamic_factors],
              ["Protective factors", incomingFields.protective_factors]
            ].filter((item) => String(item[1] || "").trim())
              .map((item) => `${item[0]}: ${String(item[1]).trim()}`)
              .join("\n");
          }
          clean.sections[section.id].fields[key] = String(value || "").slice(0, 30000);
        }
      });
    });
    return clean;
  }

  function normaliseTeam(rawTeam) {
    return Array.isArray(rawTeam) ? rawTeam.slice(0, 20).map((member) => ({
      id: String(member.id || newId()),
      role: Object.prototype.hasOwnProperty.call(ROLE_NAMES, member.role) ? member.role : "",
      nameChoice: String(member.nameChoice || ""),
      customName: String(member.customName || "").slice(0, 200),
      customRole: String(member.customRole || "").slice(0, 200)
    })) : [];
  }

  function loadLastTeam() {
    try {
      return normaliseTeam(JSON.parse(localStorage.getItem(LAST_TEAM_KEY) || "[]"));
    } catch {
      return [];
    }
  }

  function saveLastTeam() {
    const reusableTeam = normaliseTeam(state.team).filter((member) => formatTeamMember(member));
    try { localStorage.setItem(LAST_TEAM_KEY, JSON.stringify(reusableTeam)); } catch { /* no-op */ }
  }

  function newId() {
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function currentAppointment() {
    return window.VividMediAppointmentsUI && window.VividMediAppointmentsUI.getActiveAppointment
      ? window.VividMediAppointmentsUI.getActiveAppointment()
      : null;
  }

  function currentDraftKey() {
    const appointment = currentAppointment();
    const identity = appointment && appointment.appointmentGuid ? appointment.appointmentGuid : "unscheduled";
    return `${DRAFT_PREFIX}${String(identity).replace(/[^a-zA-Z0-9._-]/g, "_")}`;
  }

  function scheduleSave() {
    if (state.savedAt) state.savedAt = "";
    setSaveStatus("Saving draft...");
    window.clearTimeout(saveTimer);
    saveTimer = window.setTimeout(saveNow, 250);
  }

  function saveNow() {
    if (!activeDraftKey) activeDraftKey = currentDraftKey();
    const output = document.getElementById("edMhOutput");
    if (output) state.output = output.value;
    try {
      localStorage.setItem(activeDraftKey, JSON.stringify(state));
      setSaveStatus(state.savedAt ? `Completed ${state.savedAt}` : "Draft saved on this device");
    } catch {
      setSaveStatus("Draft could not be saved on this device", true);
    }
  }

  function loadDraftForCurrentAppointment(force) {
    const nextKey = currentDraftKey();
    if (!force && nextKey === activeDraftKey) return;
    if (activeDraftKey) saveNow();
    activeDraftKey = nextKey;
    let loaded = null;
    try {
      loaded = JSON.parse(localStorage.getItem(nextKey) || "null");
    } catch {
      loaded = null;
    }
    state = normaliseState(loaded);
    if (!loaded) state.team = loadLastTeam();
    const appointment = currentAppointment();
    if (!state.patientIdentifier && appointment && appointment.patientName) {
      state.patientIdentifier = appointment.patientName;
    }
    syncDomFromState();
  }

  function renderMhaOptions() {
    return MHA_FORMS.map((group) => (
      `<optgroup label="${escapeHtml(group.group)}">${group.forms.map((form) => {
        const value = `${form[0]} — ${form[1]}`;
        return `<option value="${escapeHtml(value)}"${form[2] ? " disabled" : ""}>${escapeHtml(value)}</option>`;
      }).join("")}</optgroup>`
    )).join("");
  }

  function renderSectionActions(section) {
    return `
      <div class="edmh-narrative-wrap" data-edmh-narrative-wrap="${escapeHtml(section.id)}">
        <label class="edmh-group-label" for="edmh-narrative-${escapeHtml(section.id)}">Organised section text (used in the finished note when present)</label>
        <textarea class="edmh-section-narrative" id="edmh-narrative-${escapeHtml(section.id)}" data-edmh-section="${escapeHtml(section.id)}" data-edmh-narrative data-edmh-control></textarea>
      </div>
      <div class="edmh-section-actions">
        <button class="edmh-action edmh-complete-section" type="button" data-edmh-action="toggle-complete" data-edmh-section-id="${escapeHtml(section.id)}">Complete section</button>
        <button class="edmh-action" type="button" data-edmh-action="assist-organise" data-edmh-section-id="${escapeHtml(section.id)}">Spell check &amp; organise</button>
        <button class="edmh-action" type="button" data-edmh-action="assist-configure" data-edmh-section-id="${escapeHtml(section.id)}">Configure text</button>
        <button class="edmh-action edmh-delete-section" type="button" data-edmh-action="delete-section" data-edmh-section-id="${escapeHtml(section.id)}">Delete section</button>
      </div>
      <div class="edmh-section-message" id="edmh-message-${escapeHtml(section.id)}" role="status"></div>`;
  }

  function renderReviewDetails() {
    return `
      <div class="edmh-field">
        <label for="edmhPatientIdentifier">Patient identifier / free text</label>
        <input id="edmhPatientIdentifier" type="text" placeholder="Patient name, UR/MRN or other identifier..." data-edmh-top-field="patientIdentifier" data-edmh-control>
      </div>
      <div class="edmh-field">
        <div class="edmh-team-toolbar">
          <span class="edmh-group-label">ED Review Psychiatry</span>
          <button class="edmh-action" type="button" data-edmh-action="add-team" data-edmh-control>+ Add clinician</button>
        </div>
        <div class="edmh-team-list" id="edmhTeamList"><div class="edmh-field-help">No roles added.</div></div>
      </div>
      <div class="edmh-legal-grid">
        <div class="edmh-field">
          <label for="edmhLegalStatus">Currently</label>
          <select id="edmhLegalStatus" data-edmh-top-field="legalStatus" data-edmh-control>
            <option value="">Select legal status...</option>
            <option value="voluntary">Voluntary patient</option>
            <option value="mha">Under Mental Health Act 2014</option>
          </select>
        </div>
      </div>
      <div class="edmh-mha-fields" id="edmhMhaFields" hidden>
        <div class="edmh-field">
          <label for="edmhMhaForm">MHA form</label>
          <select id="edmhMhaForm" data-edmh-top-field="mhaForm" data-edmh-control>
            <option value="">Select form...</option>
            ${renderMhaOptions()}
          </select>
          <div class="edmh-field-help">Records the current form only; this does not complete or validate the statutory form or replace PSOLIS.</div>
        </div>
        <div class="edmh-field">
          <label for="edmhMhaExpiry">Expires</label>
          <input id="edmhMhaExpiry" type="datetime-local" data-edmh-top-field="mhaExpiry" data-edmh-control>
        </div>
      </div>`;
  }

  function renderReviewSection() {
    return `
      <div class="edmh-field">
        <span class="edmh-group-label">Information from</span>
        <div class="edmh-source-options">${SOURCE_OPTIONS.map((source) => `
          <label class="edmh-source-option">
            <input type="checkbox" value="${escapeHtml(source)}" data-edmh-source data-edmh-control>
            <span>${escapeHtml(source)}</span>
          </label>`).join("")}
        </div>
      </div>
      <div class="edmh-field">
        <label for="edmh-review-updates">Patient / nursing / family / records / allied health / collateral updates</label>
        <textarea id="edmh-review-updates" data-edmh-section="review" data-edmh-field="updates" data-edmh-control placeholder="Attribute each update to its source where possible..."></textarea>
      </div>`;
  }

  function renderMseSection() {
    return Object.keys(MSE_OPTIONS).map((key) => `
      <div class="edmh-mse-row">
        <label class="edmh-mse-label" for="edmh-mse-${escapeHtml(key)}">${escapeHtml(FIELD_LABELS[key])}</label>
        <select data-edmh-mse-template="${escapeHtml(key)}" data-edmh-control aria-label="${escapeHtml(FIELD_LABELS[key])} phrase options">
          <option value="">Select phrase...</option>
          ${MSE_OPTIONS[key].map((option) => `<option value="${escapeHtml(option)}">${escapeHtml(option)}</option>`).join("")}
          <option value="__other__">Other / free type</option>
        </select>
        <textarea class="edmh-compact-textarea" id="edmh-mse-${escapeHtml(key)}" data-edmh-section="mse" data-edmh-field="${escapeHtml(key)}" data-edmh-control placeholder="Free type or refine selected phrase..."></textarea>
      </div>`).join("");
  }

  function renderStandardFields(section) {
    return (section.fields || []).map((field) => `
      <div class="edmh-field">
        <label for="edmh-${escapeHtml(section.id)}-${escapeHtml(field.key)}">${escapeHtml(field.label)}</label>
        <textarea id="edmh-${escapeHtml(section.id)}-${escapeHtml(field.key)}" ${field.rows ? `rows="${field.rows}"` : ""} data-edmh-section="${escapeHtml(section.id)}" data-edmh-field="${escapeHtml(field.key)}" data-edmh-control placeholder="${escapeHtml(field.placeholder || "Free type...")}"></textarea>
      </div>`).join("");
  }

  function renderSection(section) {
    let content = renderStandardFields(section);
    if (section.id === "review_details") content = renderReviewDetails();
    if (section.id === "review") content = renderReviewSection();
    if (section.id === "mse") content = renderMseSection();
    return `
      <section class="edmh-section" id="edmh-section-${escapeHtml(section.id)}" data-edmh-section-container="${escapeHtml(section.id)}">
        <div class="edmh-section-header">
          <div>
            <h3 class="edmh-section-title">${escapeHtml(section.title)}</h3>
            ${section.help ? `<div class="edmh-field-help">${escapeHtml(section.help)}</div>` : ""}
          </div>
          <span class="edmh-section-status">Draft</span>
        </div>
        <div class="edmh-section-body">
          ${content}
          ${renderSectionActions(section)}
        </div>
      </section>`;
  }

  function renderWorkspace() {
    workspace.innerHTML = `
      <div class="edmh-form-panel">
        <div class="edmh-form-heading">
          <div>
            <h2 class="edmh-title">ED MH Review</h2>
            <div class="edmh-help">Structured ED psychiatry review. Complete, reopen or delete each section independently.</div>
          </div>
          <div class="edmh-save-status" id="edmhSaveStatus" role="status"></div>
        </div>
        <div class="edmh-form-body">${SECTIONS.map(renderSection).join("")}</div>
      </div>
      <aside class="edmh-output-panel">
        <div class="edmh-output-heading">
          <div>
            <h2 class="edmh-title">Configured output</h2>
            <div class="edmh-help">Review all wording before signing or copying to the medical record.</div>
          </div>
        </div>
        <textarea class="edmh-output-area" id="edMhOutput" placeholder="Configured ED MH Review will appear here..."></textarea>
        <div class="edmh-not-documented-panel" id="edmhNotDocumentedPanel" hidden>
          <div class="edmh-not-documented-heading">Remove unused “Not documented” sections</div>
          <div class="edmh-not-documented-list" id="edmhNotDocumentedList"></div>
        </div>
        <div class="edmh-output-actions">
          <button class="edmh-action" type="button" data-edmh-action="configure-full">Configure full note</button>
          <button class="edmh-action" type="button" data-edmh-action="copy-output">Copy</button>
          <button class="edmh-action edmh-complete-consult" type="button" data-edmh-action="complete-consult">Complete consult</button>
          <button class="edmh-action" type="button" data-edmh-action="edit-output">Edit output</button>
          <button class="edmh-action edmh-delete-output" type="button" data-edmh-action="delete-output">Delete output</button>
        </div>
        <div class="edmh-output-message" id="edmhOutputMessage" role="status"></div>
      </aside>`;
  }

  function nameOptionsForMember(member) {
    const names = ROLE_NAMES[member.role] || [];
    const selectedIsCustom = member.nameChoice === "__other__" || (!names.includes(member.nameChoice) && Boolean(member.nameChoice));
    return `
      <select data-edmh-team-id="${escapeHtml(member.id)}" data-edmh-team-field="nameChoice" data-edmh-control aria-label="Clinician name">
        <option value="">Select name...</option>
        ${names.map((name) => `<option value="${escapeHtml(name)}"${member.nameChoice === name ? " selected" : ""}>${escapeHtml(name)}</option>`).join("")}
        <option value="__other__"${selectedIsCustom ? " selected" : ""}>Other / enter manually</option>
      </select>
      <input type="text" placeholder="Type name..." value="${escapeHtml(member.customName)}" data-edmh-team-id="${escapeHtml(member.id)}" data-edmh-team-field="customName" data-edmh-control${selectedIsCustom || ["RMO", "Intern", "Other"].includes(member.role) ? "" : " hidden"}>`;
  }

  function renderTeamList() {
    const list = document.getElementById("edmhTeamList");
    if (!list) return;
    if (!state.team.length) {
      list.innerHTML = '<div class="edmh-field-help">No roles added.</div>';
      return;
    }
    list.innerHTML = state.team.map((member) => `
      <div class="edmh-team-row" data-edmh-team-row="${escapeHtml(member.id)}">
        <div class="edmh-team-name-control"${member.role ? "" : " hidden"}>
          ${nameOptionsForMember(member)}
        </div>
        <div class="edmh-team-name-control">
          <select data-edmh-team-id="${escapeHtml(member.id)}" data-edmh-team-field="role" data-edmh-control aria-label="Psychiatry team role">
            <option value="">Select role...</option>
            ${Object.keys(ROLE_NAMES).map((role) => `<option value="${escapeHtml(role)}"${member.role === role ? " selected" : ""}>${escapeHtml(role)}</option>`).join("")}
          </select>
          <input type="text" placeholder="Type role..." value="${escapeHtml(member.customRole)}" data-edmh-team-id="${escapeHtml(member.id)}" data-edmh-team-field="customRole" data-edmh-control${member.role === "Other" ? "" : " hidden"}>
        </div>
        <button class="edmh-remove-team" type="button" data-edmh-action="remove-team" data-edmh-team-id="${escapeHtml(member.id)}" data-edmh-control aria-label="Remove this clinician">−</button>
      </div>`).join("");
    applySectionLock("review_details");
  }

  function syncDomFromState() {
    const patientIdentifier = document.getElementById("edmhPatientIdentifier");
    if (patientIdentifier) patientIdentifier.value = state.patientIdentifier;
    const legalStatus = document.getElementById("edmhLegalStatus");
    if (legalStatus) legalStatus.value = state.legalStatus;
    const mhaForm = document.getElementById("edmhMhaForm");
    if (mhaForm) mhaForm.value = state.mhaForm;
    const mhaExpiry = document.getElementById("edmhMhaExpiry");
    if (mhaExpiry) mhaExpiry.value = state.mhaExpiry;
    toggleMhaFields();
    renderTeamList();

    SECTIONS.forEach((section) => {
      const sectionState = state.sections[section.id];
      Object.keys(sectionState.fields).forEach((key) => {
        if (key === "sources") return;
        const control = workspace.querySelector(`[data-edmh-section="${section.id}"][data-edmh-field="${key}"]`);
        if (control) control.value = sectionState.fields[key];
      });
      const narrative = workspace.querySelector(`[data-edmh-section="${section.id}"][data-edmh-narrative]`);
      if (narrative) narrative.value = sectionState.narrative;
      const narrativeWrap = workspace.querySelector(`[data-edmh-narrative-wrap="${section.id}"]`);
      if (narrativeWrap) narrativeWrap.classList.toggle("has-content", Boolean(sectionState.narrative.trim()));
      updateSectionStateUi(section.id);
    });

    workspace.querySelectorAll("[data-edmh-source]").forEach((control) => {
      control.checked = state.sections.review.fields.sources.includes(control.value);
    });
    const output = document.getElementById("edMhOutput");
    if (output) output.value = state.output;
    renderNotDocumentedRemovals();
    setSaveStatus(state.savedAt ? `Completed ${state.savedAt}` : "Draft autosaves on this device");
  }

  function notDocumentedLines() {
    const output = document.getElementById("edMhOutput");
    if (!output) return [];
    const lines = output.value.split(/\r?\n/);
    return lines.reduce((matches, line, index) => {
      if (/\bNot documented\b/i.test(line)) {
        let text = line.trim() || "Not documented";
        if (text.toLowerCase() === "not documented") {
          const precedingLine = lines.slice(0, index).reverse().find((candidate) => candidate.trim());
          if (precedingLine) text = `${precedingLine.trim().replace(/:$/, "")} — Not documented`;
        }
        matches.push({ index, text });
      }
      return matches;
    }, []);
  }

  function renderNotDocumentedRemovals() {
    const panel = document.getElementById("edmhNotDocumentedPanel");
    const list = document.getElementById("edmhNotDocumentedList");
    if (!panel || !list) return;
    const matches = notDocumentedLines();
    panel.hidden = matches.length === 0;
    list.innerHTML = matches.map((match) => `
      <div class="edmh-not-documented-row">
        <span>${escapeHtml(match.text)}</span>
        <button type="button" data-edmh-action="remove-not-documented" data-edmh-line-index="${match.index}" aria-label="Remove ${escapeHtml(match.text)}">×</button>
      </div>`).join("");
  }

  function removeNotDocumentedLine(lineIndex) {
    const output = document.getElementById("edMhOutput");
    if (!output || output.readOnly) {
      setOutputMessage("Select Edit output before removing sections.", true);
      return;
    }
    const lines = output.value.split(/\r?\n/);
    const index = Number(lineIndex);
    if (!Number.isInteger(index) || index < 0 || index >= lines.length || !/\bNot documented\b/i.test(lines[index])) return;
    const emptySectionHeadings = new Set([
      "ED Review Psychiatry", "Current legal status", "Summary", "Review", "Progress",
      "Patient's account of progress", "MSE", "Current risk formulation and management", "ASSESSMENT", "PLAN"
    ]);
    const previousHeading = index > 0 ? lines[index - 1].trim().replace(/:$/, "") : "";
    if (lines[index].trim().toLowerCase() === "not documented" && emptySectionHeadings.has(previousHeading)) {
      lines.splice(index - 1, 2);
    } else {
      lines.splice(index, 1);
    }
    output.value = lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
    state.output = output.value;
    state.savedAt = "";
    renderNotDocumentedRemovals();
    setOutputMessage("Unused section removed. You can continue editing the output.");
    scheduleSave();
  }

  function toggleMhaFields() {
    const fields = document.getElementById("edmhMhaFields");
    if (fields) fields.hidden = state.legalStatus !== "mha";
  }

  function applySectionLock(sectionId) {
    const section = document.getElementById(`edmh-section-${sectionId}`);
    if (!section) return;
    const locked = state.sections[sectionId].complete;
    section.querySelectorAll("[data-edmh-control]").forEach((control) => {
      control.disabled = locked;
    });
  }

  function updateSectionStateUi(sectionId) {
    const section = document.getElementById(`edmh-section-${sectionId}`);
    if (!section) return;
    const complete = state.sections[sectionId].complete;
    section.classList.toggle("is-complete", complete);
    const status = section.querySelector(".edmh-section-status");
    if (status) status.textContent = complete ? "Complete" : "Draft";
    const button = section.querySelector('[data-edmh-action="toggle-complete"]');
    if (button) button.textContent = complete ? "Edit section" : "Complete section";
    applySectionLock(sectionId);
  }

  function setSaveStatus(message, isError) {
    const status = document.getElementById("edmhSaveStatus");
    if (!status) return;
    status.textContent = message || "";
    status.style.color = isError ? "var(--danger)" : "";
  }

  function setSectionMessage(sectionId, message, isError) {
    const target = document.getElementById(`edmh-message-${sectionId}`);
    if (!target) return;
    target.textContent = message || "";
    target.classList.toggle("is-error", Boolean(isError));
  }

  function setOutputMessage(message, isError) {
    const target = document.getElementById("edmhOutputMessage");
    if (!target) return;
    target.textContent = message || "";
    target.style.color = isError ? "var(--danger)" : "";
  }

  function isActive() {
    return String(consultSelect.value || "").toLowerCase() === CONSULT_TYPE.toLowerCase();
  }

  function setActive() {
    const active = isActive();
    standardWorkspace.hidden = active;
    workspace.hidden = !active;
    document.body.classList.toggle("edmh-active", active);
    if (active) loadDraftForCurrentAppointment(false);
  }

  function updateStateFromControl(target) {
    if (target.matches("[data-edmh-top-field]")) {
      const key = target.dataset.edmhTopField;
      state[key] = target.value;
      if (key === "legalStatus") {
        if (target.value !== "mha") {
          state.mhaForm = "";
          state.mhaExpiry = "";
        }
        toggleMhaFields();
      }
      scheduleSave();
      return true;
    }
    if (target.matches("[data-edmh-source]")) {
      state.sections.review.fields.sources = Array.from(workspace.querySelectorAll("[data-edmh-source]:checked"))
        .map((control) => control.value);
      scheduleSave();
      return true;
    }
    if (target.matches("[data-edmh-narrative]")) {
      const sectionId = target.dataset.edmhSection;
      state.sections[sectionId].narrative = target.value;
      const wrap = workspace.querySelector(`[data-edmh-narrative-wrap="${sectionId}"]`);
      if (wrap) wrap.classList.toggle("has-content", Boolean(target.value.trim()));
      scheduleSave();
      return true;
    }
    if (target.matches("[data-edmh-section][data-edmh-field]")) {
      const sectionId = target.dataset.edmhSection;
      const field = target.dataset.edmhField;
      state.sections[sectionId].fields[field] = target.value;
      scheduleSave();
      return true;
    }
    return false;
  }

  function updateTeamMember(target) {
    const member = state.team.find((item) => item.id === target.dataset.edmhTeamId);
    if (!member) return;
    const field = target.dataset.edmhTeamField;
    member[field] = target.value;
    if (field === "role") {
      member.nameChoice = ["RMO", "Intern", "Other"].includes(member.role) ? "__other__" : "";
      member.customName = "";
      if (member.role !== "Other") member.customRole = "";
      renderTeamList();
    } else if (field === "nameChoice") {
      if (member.nameChoice !== "__other__") member.customName = "";
      renderTeamList();
    }
    saveLastTeam();
    scheduleSave();
  }

  function toggleComplete(sectionId) {
    const sectionState = state.sections[sectionId];
    state.savedAt = "";
    sectionState.complete = !sectionState.complete;
    updateSectionStateUi(sectionId);
    setSectionMessage(sectionId, sectionState.complete ? "Section set as complete." : "Section reopened for editing.");
    saveNow();
  }

  function resetSection(sectionId) {
    if (!window.confirm(`Delete all information in ${SECTIONS.find((section) => section.id === sectionId).title}?`)) return;
    const blank = createDefaultState();
    state.savedAt = "";
    state.sections[sectionId] = blank.sections[sectionId];
    if (sectionId === "review_details") {
      state.patientIdentifier = "";
      state.team = [];
      state.legalStatus = "";
      state.mhaForm = "";
      state.mhaExpiry = "";
      saveLastTeam();
    }
    syncDomFromState();
    setSectionMessage(sectionId, "Section deleted.");
    saveNow();
  }

  function sectionData(sectionId) {
    if (sectionId === "review_details") {
      return {
        patient_identifier: state.patientIdentifier,
        team: state.team.map(formatTeamMember).filter(Boolean),
        legal_status: legalStatusText(),
        mha_form: state.mhaForm,
        mha_expiry: state.mhaExpiry
      };
    }
    return Object.assign({}, state.sections[sectionId].fields);
  }

  function formatTeamMember(member) {
    const role = member.role === "Other" ? member.customRole.trim() : member.role;
    const name = member.nameChoice === "__other__" ? member.customName.trim() : member.nameChoice.trim();
    if (!role && !name) return "";
    if (name && role) return `${name} (${role})`;
    return name || role;
  }

  function legalStatusText() {
    if (state.legalStatus === "voluntary") return "Voluntary patient";
    if (state.legalStatus === "mha") return "Under Mental Health Act 2014";
    return "";
  }

  function formatDateTime(value) {
    if (!value) return "";
    return value.replace("T", " ");
  }

  function sectionHasContent(sectionId) {
    const data = sectionData(sectionId);
    return Object.values(data).some((value) => Array.isArray(value) ? value.length : String(value || "").trim());
  }

  function appendStructuredSection(lines, heading, sectionId, keys) {
    const sectionState = state.sections[sectionId];
    const sectionLines = [];
    if (sectionState.narrative.trim()) {
      sectionLines.push(sectionId === "plan" ? numberPlan(sectionState.narrative) : sectionState.narrative.trim());
    } else {
      keys.forEach((key) => {
        const value = sectionState.fields[key];
        const clean = Array.isArray(value) ? value.join(", ") : String(value || "").trim();
        if (!clean) return;
        if (key === "plan") sectionLines.push(numberPlan(clean));
        else if (keys.length === 1) sectionLines.push(clean);
        else sectionLines.push(`${FIELD_LABELS[key]}: ${clean}`);
      });
    }
    if (sectionLines.length) lines.push("", heading, ...sectionLines);
  }

  function numberPlan(value) {
    const items = String(value || "").split(/\r?\n/)
      .map((line) => line.trim().replace(/^\s*(?:\d+[.)]|[-•])\s*/, ""))
      .filter(Boolean);
    return items.map((item, index) => `${index + 1}. ${item}`).join("\n");
  }

  function stripUndocumentedOutput(value) {
    const headings = new Set([
      "ED Review Psychiatry", "Current legal status", "Summary", "Review", "Progress",
      "Patient's account of progress", "MSE", "ASSESSMENT",
      "Current risk formulation and management", "PLAN"
    ]);
    const lines = String(value || "").split(/\r?\n/).filter((line) => {
      const trimmed = line.trim().replace(/^[-*]\s*/, "");
      const content = trimmed.includes(":") ? trimmed.slice(trimmed.indexOf(":") + 1).trim() : trimmed;
      return !/^(?:not (?:formally )?(?:documented|assessed|recorded|provided|specified)|unknown|n\/?a|no (?:relevant )?(?:information|assessment|documentation|findings?|history) (?:is |was )?(?:available|provided|recorded|documented)|no evidence documented(?: in the supplied information)?)\.?$/i.test(content);
    });
    const withoutEmptyHeadings = lines.filter((line, index) => {
      const heading = line.trim().replace(/:$/, "");
      if (!headings.has(heading)) return true;
      let next = index + 1;
      while (next < lines.length && !lines[next].trim()) next += 1;
      if (next >= lines.length) return false;
      return !headings.has(lines[next].trim().replace(/:$/, ""));
    });
    return withoutEmptyHeadings.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  }

  function buildLocalNote() {
    const lines = ["ED MH REVIEW"];
    if (state.patientIdentifier.trim()) lines.push("", `Patient: ${state.patientIdentifier.trim()}`);
    const team = state.team.map(formatTeamMember).filter(Boolean);
    if (team.length) lines.push("", "ED Review Psychiatry", team.join(", "));
    const legalStatus = legalStatusText();
    if (legalStatus) {
      lines.push("", "Current legal status", legalStatus);
      if (state.legalStatus === "mha" && state.mhaForm) lines.push(`MHA form: ${state.mhaForm}`);
      if (state.legalStatus === "mha" && state.mhaExpiry) lines.push(`Expires: ${formatDateTime(state.mhaExpiry)}`);
    }
    if (state.sections.review_details.narrative.trim()) {
      lines.push(`Additional review details: ${state.sections.review_details.narrative.trim()}`);
    }

    appendStructuredSection(lines, "Summary", "summary", ["summary"]);
    appendStructuredSection(lines, "Review", "review", ["sources", "updates"]);
    appendStructuredSection(lines, "Progress", "progress", ["behaviour_events", "sleep_intake_adls", "medication", "engagement", "progress_notes"]);
    appendStructuredSection(lines, "Patient's account of progress", "patient_account", ["current_symptoms", "concerns_goals", "understanding"]);
    appendStructuredSection(lines, "MSE", "mse", Object.keys(MSE_OPTIONS));
    appendStructuredSection(lines, "ASSESSMENT", "assessment", ["clinical_progress", "working_diagnosis", "response_management"]);
    appendStructuredSection(lines, "Current risk formulation and management", "risk", ["risk_formulation_management"]);
    appendStructuredSection(lines, "PLAN", "plan", ["plan"]);
    return stripUndocumentedOutput(lines.join("\n"));
  }

  function hasClinicalContent() {
    return SECTIONS.some((section) => section.id !== "review_details" && sectionHasContent(section.id))
      || Boolean(state.sections.review_details.narrative.trim());
  }

  async function assistSection(sectionId, action, button) {
    if (state.sections[sectionId].complete) {
      setSectionMessage(sectionId, "Select Edit section before changing completed text.", true);
      return;
    }
    if (!sectionHasContent(sectionId) && !hasClinicalContent()) {
      setSectionMessage(sectionId, "Enter review information first.", true);
      return;
    }
    const original = button.textContent;
    button.classList.add("is-busy");
    button.textContent = action === "organise" ? "Organising..." : "Configuring...";
    setSectionMessage(sectionId, "");
    try {
      const response = await fetch("/api/ed-mh-review/assist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          section: sectionId,
          section_data: sectionData(sectionId),
          context: buildLocalNote()
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `Writing assist failed (${response.status})`);
      if (data.fields && typeof data.fields === "object") {
        Object.keys(state.sections[sectionId].fields).forEach((key) => {
          const suggested = String(data.fields[key] || "").trim();
          if (suggested) state.sections[sectionId].fields[key] = suggested;
        });
        setSectionMessage(
          sectionId,
          action === "organise"
            ? "Relevant supported information organised into the correct fields. Review before completing the section."
            : "Supported fields configured. Review each entry before completing the section."
        );
      } else if (data.text) {
        state.sections[sectionId].narrative = String(data.text).trim();
        setSectionMessage(sectionId, "Organised text added. Review it before completing the section.");
      } else if (data.empty) {
        setSectionMessage(sectionId, "No additional supported information was found, so nothing was added.");
        return;
      } else {
        throw new Error("No configured text returned");
      }
      state.savedAt = "";
      syncDomFromState();
      saveNow();
    } catch (error) {
      setSectionMessage(sectionId, error.message || "Writing assist unavailable. Your entered text is unchanged.", true);
    } finally {
      button.classList.remove("is-busy");
      button.textContent = original;
    }
  }

  async function configureFullNote(button) {
    if (!hasClinicalContent()) {
      setOutputMessage("Enter clinical review information first.", true);
      return;
    }
    const output = document.getElementById("edMhOutput");
    const original = button.textContent;
    button.classList.add("is-busy");
    button.textContent = "Configuring...";
    output.readOnly = true;
    setOutputMessage("Configuring ED MH Review...");
    const localNote = buildLocalNote();
    try {
      const response = await fetch("/convert-notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clinical_data: localNote,
          note_type: "consultation_note",
          consult_type: CONSULT_TYPE
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `Configuration failed (${response.status})`);
      output.value = stripUndocumentedOutput(data.clinical_notes || data.output || localNote) || localNote;
      state.savedAt = "";
      state.output = output.value;
      renderNotDocumentedRemovals();
      setOutputMessage("Configured. Review all wording and edit if needed before completing.");
    } catch {
      output.value = localNote;
      state.savedAt = "";
      state.output = localNote;
      renderNotDocumentedRemovals();
      setOutputMessage("AI configuration is unavailable; the structured local note is shown and can be edited.", true);
    } finally {
      output.readOnly = false;
      button.classList.remove("is-busy");
      button.textContent = original;
      saveNow();
    }
  }

  async function copyOutput() {
    const output = document.getElementById("edMhOutput");
    if (!output.value.trim()) {
      setOutputMessage("There is no output to copy.", true);
      return;
    }
    await navigator.clipboard.writeText(output.value);
    setOutputMessage("Output copied.");
  }

  function completeConsult() {
    const incomplete = SECTIONS.filter((section) => !state.sections[section.id].complete);
    if (incomplete.length && !window.confirm(`${incomplete.length} section${incomplete.length === 1 ? " is" : "s are"} still marked Draft. Complete the consult anyway?`)) return;
    const output = document.getElementById("edMhOutput");
    if (!output.value.trim()) {
      output.value = buildLocalNote();
      state.output = output.value;
    }
    renderNotDocumentedRemovals();
    const appointment = currentAppointment();
    saveLastTeam();
    state.savedAt = new Date().toLocaleString();
    const item = {
      id: newId(),
      patientName: state.patientIdentifier.trim() || (appointment && appointment.patientName) || "Existing patient",
      consultType: CONSULT_TYPE,
      date: state.savedAt,
      content: output.value,
      expanded: false,
      appointmentGuid: (appointment && appointment.appointmentGuid) || "",
      edMhDraft: JSON.parse(JSON.stringify(state))
    };
    try {
      savedOutputs.unshift(item);
      savedOutputs = savedOutputs.slice(0, 60);
      localStorage.setItem("vivid_saved_outputs", JSON.stringify(savedOutputs));
      renderSavedOutputs();
      if (appointment && appointment.appointmentGuid && window.VividMediAppointmentsUI) {
        suppressAppointmentReload = true;
        window.VividMediAppointmentsUI.completeAppointment(appointment.appointmentGuid);
      }
      output.readOnly = true;
      setOutputMessage("Consult completed and added to Saved Outputs. Select Edit there to reopen it.");
      setSaveStatus(`Completed ${state.savedAt}`);
      saveNow();
    } catch {
      setOutputMessage("The consult could not be added to Saved Outputs. Your draft remains here.", true);
    }
  }

  function editOutput() {
    const output = document.getElementById("edMhOutput");
    output.readOnly = false;
    renderNotDocumentedRemovals();
    output.focus();
    setOutputMessage("Output reopened for editing.");
  }

  function deleteOutput() {
    const output = document.getElementById("edMhOutput");
    if (!output.value.trim()) return;
    if (!window.confirm("Delete the configured output? Your structured section entries will remain.")) return;
    output.value = "";
    state.output = "";
    state.savedAt = "";
    output.readOnly = false;
    renderNotDocumentedRemovals();
    setOutputMessage("Configured output deleted. Section entries were kept.");
    saveNow();
  }

  function clearReview() {
    if (!window.confirm("Clear this ED MH Review draft, including all sections and output?")) return;
    try { localStorage.removeItem(activeDraftKey || currentDraftKey()); } catch { /* no-op */ }
    state = createDefaultState();
    const appointment = currentAppointment();
    if (appointment && appointment.patientName) state.patientIdentifier = appointment.patientName;
    syncDomFromState();
    setOutputMessage("ED MH Review cleared.");
    saveNow();
  }

  function startReview() {
    const appointment = currentAppointment();
    if (!appointment) {
      setOutputMessage("Select a patient from the appointments list before starting the review.", true);
      return;
    }
    if (!state.patientIdentifier) state.patientIdentifier = appointment.patientName || "";
    syncDomFromState();
    document.getElementById("edmhPatientIdentifier").focus();
    setOutputMessage(`ED MH Review started for ${appointment.patientName || "selected patient"}.`);
    saveNow();
  }

  function handleWorkspaceClick(event) {
    const button = event.target.closest("[data-edmh-action]");
    if (!button) return;
    const action = button.dataset.edmhAction;
    const sectionId = button.dataset.edmhSectionId;
    if (action === "add-team") {
      state.team.push({ id: newId(), role: "", nameChoice: "", customName: "", customRole: "" });
      renderTeamList();
      const rows = document.querySelectorAll(".edmh-team-row");
      rows[rows.length - 1] && rows[rows.length - 1].querySelector('[data-edmh-team-field="role"]').focus();
      saveLastTeam();
      scheduleSave();
    } else if (action === "remove-team") {
      state.team = state.team.filter((member) => member.id !== button.dataset.edmhTeamId);
      renderTeamList();
      saveLastTeam();
      scheduleSave();
    } else if (action === "toggle-complete") {
      toggleComplete(sectionId);
    } else if (action === "delete-section") {
      resetSection(sectionId);
    } else if (action === "assist-organise") {
      assistSection(sectionId, "organise", button);
    } else if (action === "assist-configure") {
      assistSection(sectionId, "configure", button);
    } else if (action === "configure-full") {
      configureFullNote(button);
    } else if (action === "copy-output") {
      copyOutput();
    } else if (action === "complete-consult") {
      completeConsult();
    } else if (action === "edit-output") {
      editOutput();
    } else if (action === "delete-output") {
      deleteOutput();
    } else if (action === "remove-not-documented") {
      removeNotDocumentedLine(button.dataset.edmhLineIndex);
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
      const output = document.getElementById("edMhOutput");
      output.readOnly = false;
      output.focus();
      setOutputMessage("Saved ED MH Review reopened for editing. Completing it again creates an updated Saved Output.");
      saveNow();
    };
  }

  function bindEvents() {
    consultSelect.addEventListener("change", setActive);
    workspace.addEventListener("click", handleWorkspaceClick);
    workspace.addEventListener("input", (event) => {
      if (event.target.matches("[data-edmh-team-field]")) updateTeamMember(event.target);
      else if (event.target.id === "edMhOutput") {
        state.output = event.target.value;
        renderNotDocumentedRemovals();
        scheduleSave();
      } else updateStateFromControl(event.target);
    });
    workspace.addEventListener("change", (event) => {
      if (event.target.matches("[data-edmh-team-field]")) {
        updateTeamMember(event.target);
        return;
      }
      if (event.target.matches("[data-edmh-mse-template]")) {
        const key = event.target.dataset.edmhMseTemplate;
        if (event.target.value !== "__other__" && event.target.value) {
          state.sections.mse.fields[key] = event.target.value;
          const textarea = workspace.querySelector(`[data-edmh-section="mse"][data-edmh-field="${key}"]`);
          if (textarea) textarea.value = event.target.value;
          scheduleSave();
        } else if (event.target.value === "__other__") {
          workspace.querySelector(`[data-edmh-section="mse"][data-edmh-field="${key}"]`)?.focus();
        }
        return;
      }
      updateStateFromControl(event.target);
    });

    interceptExistingButton("convertBtn", () => configureFullNote(workspace.querySelector('[data-edmh-action="configure-full"]')));
    interceptExistingButton("completeConsultBtn", completeConsult);
    interceptExistingButton("editOutputBtn", editOutput);
    interceptExistingButton("clearOutputBtn", deleteOutput);
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

  function init() {
    renderWorkspace();
    bindEvents();
    wrapSavedOutputEditor();
    syncDomFromState();
    setActive();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  window.VividMediEdMhReview = {
    buildLocalNote,
    getState: () => JSON.parse(JSON.stringify(state)),
    loadState: (nextState) => { state = normaliseState(nextState); syncDomFromState(); },
    consultationType: CONSULT_TYPE
  };
})();

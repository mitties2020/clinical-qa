(function () {
  function parseMediRecordsDateParts(value) {
    if (!value || typeof value !== "string") return null;
    const match = value.trim().match(/^(\d{4})[/-](\d{2})[/-](\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?/);
    if (!match) return null;
    return {
      year: Number(match[1]),
      month: Number(match[2]),
      day: Number(match[3]),
      hour: Number(match[4]),
      minute: Number(match[5]),
      second: Number(match[6] || 0)
    };
  }

  function getTimeZoneParts(date, timeZone) {
    const parts = new Intl.DateTimeFormat("en-AU", {
      timeZone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false
    }).formatToParts(date);

    return parts.reduce((acc, part) => {
      if (part.type !== "literal") acc[part.type] = Number(part.value);
      return acc;
    }, {});
  }

  function zonedDateTimeToDate(parts, timeZone) {
    const guess = Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second || 0);
    const actual = getTimeZoneParts(new Date(guess), timeZone);
    const actualAsUtc = Date.UTC(
      actual.year,
      actual.month - 1,
      actual.day,
      actual.hour,
      actual.minute,
      actual.second || 0
    );
    const offset = actualAsUtc - guess;
    return new Date(guess - offset);
  }

  function parseMediRecordsDateTime(value, sourceTimeZone) {
    const parts = parseMediRecordsDateParts(value);
    if (!parts) {
      const fallback = new Date(value);
      return Number.isNaN(fallback.getTime()) ? null : fallback;
    }
    return zonedDateTimeToDate(parts, sourceTimeZone || "Australia/Brisbane");
  }

  function formatTimeInPerth(dateInput) {
    if (!dateInput) return "";
    const date = dateInput instanceof Date ? dateInput : new Date(dateInput);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat("en-AU", {
      timeZone: "Australia/Perth",
      hour: "2-digit",
      minute: "2-digit",
      hour12: true
    }).format(date).replace(/\s?(am|pm)$/i, (match) => match.toUpperCase());
  }

  function calculateAgeFromMediRecordsDob(dobMs) {
    if (dobMs === null || dobMs === undefined || dobMs === "") return null;
    const dob = new Date(Number(dobMs));
    if (Number.isNaN(dob.getTime())) return null;

    const today = new Date();
    let age = today.getFullYear() - dob.getFullYear();
    const monthOffset = today.getMonth() - dob.getMonth();
    if (monthOffset < 0 || (monthOffset === 0 && today.getDate() < dob.getDate())) {
      age -= 1;
    }
    return age;
  }

  function buildPatientName(raw) {
    return [raw.firstName, raw.middleName, raw.lastName]
      .map((part) => String(part || "").trim())
      .filter(Boolean)
      .join(" ") || raw.patientName || "Unknown patient";
  }

  function fallbackAppointmentGuid(raw) {
    return [
      raw.appointmentGuid,
      raw.patientGuid,
      raw.kendoStartTime,
      raw.startTimeOriginal
    ].map((part) => String(part || "").trim()).filter(Boolean).join("-");
  }

  function normaliseMediRecordsAppointment(raw) {
    const sourceTimeZone = raw.canonicalId || raw.timezone || "Australia/Brisbane";
    const startDate = parseMediRecordsDateTime(raw.kendoStartTime || raw.startTimeOriginal, sourceTimeZone);
    const endDate = parseMediRecordsDateTime(raw.kendoEndTime || raw.endTimeOriginal, sourceTimeZone);
    const appointmentGuid = String(raw.appointmentGuid || fallbackAppointmentGuid(raw));

    return {
      appointmentGuid,
      patientGuid: raw.patientGuid || "",
      patientName: buildPatientName(raw),
      age: calculateAgeFromMediRecordsDob(raw.dob),
      mobilePhone: raw.mobilePhone || raw.mobile || raw.phone || "",
      appointmentType: raw.practiceAppointmentTypeName || raw.appointmentType || "",
      appointmentNote: raw.appointmentNote || "",
      startTimeOriginal: raw.kendoStartTime || raw.startTimeOriginal || "",
      startTimeWA: formatTimeInPerth(startDate),
      endTimeWA: formatTimeInPerth(endDate),
      startTimeUtc: startDate ? startDate.toISOString() : "",
      endTimeUtc: endDate ? endDate.toISOString() : "",
      statusId: raw.appointmentStatusId ?? raw.statusId ?? null,
      providerGuid: raw.userGuid || raw.providerGuid || "",
      practiceGuid: raw.practiceGuid || "",
      timezone: sourceTimeZone,
      consultNote: "",
      isExpanded: false,
      isDeleted: false,
      isDone: false,
      lastSavedAt: null,
      lastSyncedAt: null,
      missingFromLatestSync: false
    };
  }

  function normaliseMediRecordsAppointments(rawList) {
    return (rawList || [])
      .filter((item) => item && typeof item === "object")
      .map(normaliseMediRecordsAppointment)
      .filter((item) => item.appointmentGuid);
  }

  function looksLikeAppointment(item) {
    return Boolean(item && typeof item === "object" && (
      item.appointmentGuid ||
      item.kendoStartTime ||
      item.patientGuid ||
      item.practiceAppointmentTypeName
    ));
  }

  function findAppointmentArray(value) {
    if (Array.isArray(value)) {
      return value.some(looksLikeAppointment) ? value : null;
    }
    if (!value || typeof value !== "object") return null;

    const preferredKeys = ["appointments", "items", "data", "results", "value"];
    for (const key of preferredKeys) {
      const found = findAppointmentArray(value[key]);
      if (found) return found;
    }

    for (const key of Object.keys(value)) {
      const found = findAppointmentArray(value[key]);
      if (found) return found;
    }
    return null;
  }

  function extractAppointmentArrayFromPayload(payload) {
    const found = findAppointmentArray(payload);
    if (!found) throw new Error("No appointment array found in pasted JSON.");
    return found;
  }

  function mergeAppointments(existingAppointments, incomingAppointments) {
    const existing = Array.isArray(existingAppointments) ? existingAppointments : [];
    const incoming = Array.isArray(incomingAppointments) ? incomingAppointments : [];
    const existingById = new Map(existing.map((appointment) => [appointment.appointmentGuid, appointment]));
    const incomingIds = new Set(incoming.map((appointment) => appointment.appointmentGuid));
    const now = new Date().toISOString();

    const merged = incoming.map((incomingAppointment) => {
      const existingAppointment = existingById.get(incomingAppointment.appointmentGuid);
      return {
        ...existingAppointment,
        ...incomingAppointment,
        consultNote: existingAppointment?.consultNote || "",
        isExpanded: Boolean(existingAppointment?.isExpanded),
        isDone: Boolean(existingAppointment?.isDone),
        isDeleted: Boolean(existingAppointment?.isDeleted),
        lastSavedAt: existingAppointment?.lastSavedAt || null,
        lastSyncedAt: now,
        missingFromLatestSync: false
      };
    });

    const retainedMissing = existing
      .filter((appointment) => !incomingIds.has(appointment.appointmentGuid) && !appointment.isDeleted)
      .map((appointment) => ({
        ...appointment,
        missingFromLatestSync: true
      }));

    const deletedTombstones = existing.filter((appointment) => appointment.isDeleted && !incomingIds.has(appointment.appointmentGuid));

    return [...merged, ...retainedMissing, ...deletedTombstones].sort((a, b) => {
      const aTime = Date.parse(a.startTimeUtc || a.startTimeOriginal || "");
      const bTime = Date.parse(b.startTimeUtc || b.startTimeOriginal || "");
      if (Number.isNaN(aTime) && Number.isNaN(bTime)) return 0;
      if (Number.isNaN(aTime)) return 1;
      if (Number.isNaN(bTime)) return -1;
      return aTime - bTime;
    });
  }

  function todayDateString() {
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: "Australia/Perth",
      year: "numeric",
      month: "2-digit",
      day: "2-digit"
    }).formatToParts(new Date()).reduce((acc, part) => {
      if (part.type !== "literal") acc[part.type] = part.value;
      return acc;
    }, {});
    return `${parts.year}-${parts.month}-${parts.day}`;
  }

  async function fetchTodaysAppointmentsFromMediRecords(dateString) {
    const date = dateString || todayDateString();
    const response = await fetch(`/api/medirecords/my-appointments?date=${encodeURIComponent(date)}`, {
      method: "GET",
      credentials: "same-origin",
      headers: { "Accept": "application/json" }
    });
    if (!response.ok) {
      throw new Error(`Appointments endpoint returned ${response.status}`);
    }
    const payload = await response.json();
    const rawAppointments = extractAppointmentArrayFromPayload(payload);
    return normaliseMediRecordsAppointments(rawAppointments);
  }

  window.VividMediAppointmentsService = {
    parseMediRecordsDateTime,
    formatTimeInPerth,
    calculateAgeFromMediRecordsDob,
    fetchTodaysAppointmentsFromMediRecords,
    normaliseMediRecordsAppointment,
    normaliseMediRecordsAppointments,
    extractAppointmentArrayFromPayload,
    mergeAppointments,
    todayDateString
  };
})();

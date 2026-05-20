(function () {
  const PERTH_TIME_ZONE = "Australia/Perth";
  const DEFAULT_SOURCE_TIME_ZONE = "Australia/Brisbane";

  function isValidTimeZone(timeZone) {
    if (!timeZone) return false;

    try {
      new Intl.DateTimeFormat("en-AU", { timeZone }).format(new Date());
      return true;
    } catch {
      return false;
    }
  }

  function getSafeTimeZone(timeZone) {
    return isValidTimeZone(timeZone) ? timeZone : DEFAULT_SOURCE_TIME_ZONE;
  }

  function parseMediRecordsDateTimeParts(value) {
    if (!value) return null;

    const match = String(value).trim().match(
      /^(\d{4})[/-](\d{1,2})[/-](\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?/
    );

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

  function getPartsInTimeZone(date, timeZone) {
    const formatter = new Intl.DateTimeFormat("en-AU", {
      timeZone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hourCycle: "h23"
    });

    const parts = formatter.formatToParts(date).reduce((acc, part) => {
      if (part.type !== "literal") acc[part.type] = Number(part.value);
      return acc;
    }, {});

    return {
      year: parts.year,
      month: parts.month,
      day: parts.day,
      hour: parts.hour || 0,
      minute: parts.minute || 0,
      second: parts.second || 0
    };
  }

  function getTimeZoneOffsetMs(date, timeZone) {
    const parts = getPartsInTimeZone(date, timeZone);
    const localAsUtc = Date.UTC(
      parts.year,
      parts.month - 1,
      parts.day,
      parts.hour,
      parts.minute,
      parts.second
    );

    return localAsUtc - date.getTime();
  }

  function zonedDateTimePartsToDate(parts, timeZone) {
    const safeTimeZone = getSafeTimeZone(timeZone);
    const naiveUtc = Date.UTC(
      parts.year,
      parts.month - 1,
      parts.day,
      parts.hour,
      parts.minute,
      parts.second
    );

    let offset = getTimeZoneOffsetMs(new Date(naiveUtc), safeTimeZone);
    let utcMs = naiveUtc - offset;
    const adjustedOffset = getTimeZoneOffsetMs(new Date(utcMs), safeTimeZone);

    if (adjustedOffset !== offset) {
      utcMs = naiveUtc - adjustedOffset;
    }

    return new Date(utcMs);
  }

  function parseMediRecordsDateTime(value, sourceTimeZone) {
    if (!value) return null;
    if (value instanceof Date) return value;
    if (typeof value === "number") return new Date(value);

    const text = String(value).trim();
    const hasExplicitZone = /(?:z|[+-]\d{2}:?\d{2})$/i.test(text);

    if (hasExplicitZone) {
      const date = new Date(text.replace(/\//g, "-"));
      return Number.isNaN(date.getTime()) ? null : date;
    }

    const parts = parseMediRecordsDateTimeParts(text);
    if (parts) {
      return zonedDateTimePartsToDate(parts, sourceTimeZone || DEFAULT_SOURCE_TIME_ZONE);
    }

    const fallback = new Date(text.replace(/\//g, "-").replace(" ", "T"));
    return Number.isNaN(fallback.getTime()) ? null : fallback;
  }

  function formatTimeInPerth(dateInput, sourceTimeZone) {
    const date = typeof dateInput === "string"
      ? parseMediRecordsDateTime(dateInput, sourceTimeZone)
      : dateInput;

    if (!date || Number.isNaN(new Date(date).getTime())) return "";

    return new Intl.DateTimeFormat("en-AU", {
      timeZone: PERTH_TIME_ZONE,
      hour: "2-digit",
      minute: "2-digit",
      hour12: true
    }).format(new Date(date)).toUpperCase();
  }

  function formatCompactTimeInPerth(dateInput, sourceTimeZone) {
    const date = typeof dateInput === "string"
      ? parseMediRecordsDateTime(dateInput, sourceTimeZone)
      : dateInput;

    if (!date || Number.isNaN(new Date(date).getTime())) return "";

    return new Intl.DateTimeFormat("en-AU", {
      timeZone: PERTH_TIME_ZONE,
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
      hourCycle: "h23"
    }).format(new Date(date));
  }

  function formatDateKeyInPerth(dateInput, sourceTimeZone) {
    const date = typeof dateInput === "string"
      ? parseMediRecordsDateTime(dateInput, sourceTimeZone)
      : dateInput;

    if (!date || Number.isNaN(new Date(date).getTime())) return "";

    const parts = getPartsInTimeZone(new Date(date), PERTH_TIME_ZONE);
    return [
      parts.year,
      String(parts.month).padStart(2, "0"),
      String(parts.day).padStart(2, "0")
    ].join("-");
  }

  function getTodayDateInPerth() {
    return formatDateKeyInPerth(new Date());
  }

  function calculateAgeFromMediRecordsDob(dobMs) {
    if (dobMs === null || dobMs === undefined || dobMs === "") return null;

    const dob = new Date(Number(dobMs));
    if (Number.isNaN(dob.getTime())) return null;

    const today = getPartsInTimeZone(new Date(), PERTH_TIME_ZONE);
    let age = today.year - dob.getFullYear();
    const monthDelta = today.month - (dob.getMonth() + 1);

    if (monthDelta < 0 || (monthDelta === 0 && today.day < dob.getDate())) {
      age -= 1;
    }

    return age;
  }

  function buildPatientName(raw) {
    const name = [raw.firstName, raw.middleName, raw.lastName]
      .map((part) => String(part || "").trim())
      .filter(Boolean)
      .join(" ");

    return name || raw.patientName || raw.name || "Unknown patient";
  }

  function normaliseMediRecordsAppointment(raw) {
    if (!raw || typeof raw !== "object") return null;

    const timezone = getSafeTimeZone(raw.canonicalId || raw.timezone || raw.timeZone);
    const startTimeOriginal = raw.kendoStartTime || raw.startTimeOriginal || raw.startTime || raw.start || "";
    const endTimeOriginal = raw.kendoEndTime || raw.endTimeOriginal || raw.endTime || raw.end || "";
    const startDate = parseMediRecordsDateTime(startTimeOriginal, timezone);
    const endDate = parseMediRecordsDateTime(endTimeOriginal, timezone);
    const patientName = buildPatientName(raw);
    const appointmentGuid = raw.appointmentGuid || raw.guid || raw.id || [
      raw.patientGuid || patientName,
      startTimeOriginal
    ].join("-");

    return {
      appointmentGuid,
      patientGuid: raw.patientGuid || "",
      patientName,
      age: calculateAgeFromMediRecordsDob(raw.dob),
      mobilePhone: raw.mobilePhone || raw.mobile || raw.phone || "",
      appointmentType: raw.practiceAppointmentTypeName || raw.appointmentType || raw.type || "",
      appointmentNote: raw.appointmentNote || "",
      startTimeOriginal,
      endTimeOriginal,
      startTimeWA: formatTimeInPerth(startDate),
      startTimeWACompact: formatCompactTimeInPerth(startDate),
      endTimeWA: formatTimeInPerth(endDate),
      appointmentDateWA: formatDateKeyInPerth(startDate),
      startEpochMs: startDate ? startDate.getTime() : null,
      endEpochMs: endDate ? endDate.getTime() : null,
      statusId: raw.appointmentStatusId || raw.statusId || null,
      providerGuid: raw.userGuid || raw.providerGuid || "",
      practiceGuid: raw.practiceGuid || "",
      timezone,
      consultNote: "",
      isExpanded: false,
      isDeleted: false,
      isDone: false,
      lastSavedAt: null,
      lastSyncedAt: new Date().toISOString(),
      missingFromLatestSync: false
    };
  }

  function looksLikeAppointment(item) {
    return Boolean(
      item &&
      typeof item === "object" &&
      (item.appointmentGuid || item.kendoStartTime || item.patientGuid || item.firstName || item.lastName)
    );
  }

  function findAppointmentArray(value, depth) {
    if (!value || depth > 5) return [];
    if (Array.isArray(value)) {
      return value.some(looksLikeAppointment) ? value : [];
    }
    if (typeof value !== "object") return [];

    const preferredKeys = [
      "appointments",
      "Appointments",
      "appointmentList",
      "appointment_list",
      "items",
      "Items",
      "results",
      "Results",
      "data",
      "Data"
    ];

    for (const key of preferredKeys) {
      const found = findAppointmentArray(value[key], depth + 1);
      if (found.length) return found;
    }

    for (const key of Object.keys(value)) {
      const found = findAppointmentArray(value[key], depth + 1);
      if (found.length) return found;
    }

    return [];
  }

  function extractAppointmentsFromPayload(payload) {
    return findAppointmentArray(payload, 0);
  }

  function normaliseMediRecordsAppointments(payload) {
    return extractAppointmentsFromPayload(payload)
      .map(normaliseMediRecordsAppointment)
      .filter(Boolean);
  }

  function getAppointmentSortValue(appointment) {
    if (Number.isFinite(appointment.startEpochMs)) return appointment.startEpochMs;

    const fallbackDate = parseMediRecordsDateTime(
      appointment.startTimeOriginal,
      appointment.timezone || DEFAULT_SOURCE_TIME_ZONE
    );

    return fallbackDate ? fallbackDate.getTime() : Number.MAX_SAFE_INTEGER;
  }

  function sortAppointments(appointments) {
    return [...(appointments || [])].sort((a, b) => (
      getAppointmentSortValue(a) - getAppointmentSortValue(b)
    ));
  }

  function mergeAppointments(existingAppointments, incomingAppointments) {
    const existingList = Array.isArray(existingAppointments) ? existingAppointments : [];
    const incomingList = Array.isArray(incomingAppointments) ? incomingAppointments : [];
    const existingById = new Map(existingList.map((appointment) => [
      appointment.appointmentGuid,
      appointment
    ]));
    const incomingIds = new Set(incomingList.map((appointment) => appointment.appointmentGuid));
    const now = new Date().toISOString();

    const merged = incomingList.map((incoming) => {
      const existing = existingById.get(incoming.appointmentGuid);

      return {
        ...existing,
        ...incoming,
        consultNote: existing ? existing.consultNote || "" : incoming.consultNote || "",
        isExpanded: existing ? Boolean(existing.isExpanded) : false,
        isDone: existing ? Boolean(existing.isDone) : false,
        isDeleted: existing ? Boolean(existing.isDeleted) : false,
        lastSavedAt: existing ? existing.lastSavedAt || null : null,
        lastSyncedAt: now,
        missingFromLatestSync: false
      };
    });

    const retainedMissing = existingList
      .filter((appointment) => !incomingIds.has(appointment.appointmentGuid))
      .map((appointment) => ({
        ...appointment,
        missingFromLatestSync: !appointment.isDeleted
      }));

    return sortAppointments([...merged, ...retainedMissing]);
  }

  async function fetchTodaysAppointmentsFromMediRecords(date) {
    const dateKey = date || getTodayDateInPerth();
    const headers = { Accept: "application/json" };
    const appAuthToken = localStorage.getItem("vm_auth_token") || localStorage.getItem("auth_token");

    if (appAuthToken) {
      headers.Authorization = `Bearer ${appAuthToken}`;
    }

    const response = await fetch(
      `/api/medirecords/my-appointments?date=${encodeURIComponent(dateKey)}`,
      {
        method: "GET",
        headers,
        credentials: "same-origin"
      }
    );

    if (!response.ok) {
      throw new Error(`Appointments endpoint returned ${response.status}`);
    }

    const payload = await response.json();
    return normaliseMediRecordsAppointments(payload);
  }

  window.VividMediAppointmentService = {
    PERTH_TIME_ZONE,
    DEFAULT_SOURCE_TIME_ZONE,
    parseMediRecordsDateTime,
    formatTimeInPerth,
    formatCompactTimeInPerth,
    formatDateKeyInPerth,
    getTodayDateInPerth,
    calculateAgeFromMediRecordsDob,
    fetchTodaysAppointmentsFromMediRecords,
    normaliseMediRecordsAppointment,
    normaliseMediRecordsAppointments,
    extractAppointmentsFromPayload,
    mergeAppointments,
    sortAppointments
  };
})();

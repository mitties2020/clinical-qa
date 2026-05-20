(function () {
  const APPOINTMENT_STORAGE_KEY = "vividmedi_appointments_v1";

  function loadAppointmentsFromStorage() {
    try {
      const raw = localStorage.getItem(APPOINTMENT_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function saveAppointmentsToStorage(appointments) {
    localStorage.setItem(APPOINTMENT_STORAGE_KEY, JSON.stringify(appointments || []));
  }

  window.VividMediAppointmentStorage = {
    APPOINTMENT_STORAGE_KEY,
    loadAppointmentsFromStorage,
    saveAppointmentsToStorage
  };
})();

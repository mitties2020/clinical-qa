(function () {
  const APPOINTMENT_STORAGE_KEY = "vividmedi_appointments_v1";

  function loadAppointmentsFromStorage() {
    try {
      const stored = localStorage.getItem(APPOINTMENT_STORAGE_KEY);
      return stored ? JSON.parse(stored) || [] : [];
    } catch (error) {
      console.warn("Could not load appointments from local storage", error);
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

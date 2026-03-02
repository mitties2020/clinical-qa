document.addEventListener("DOMContentLoaded", () => {
  const upgradeBtn = document.getElementById("upgradeBtn");
  if (!upgradeBtn) return;

  upgradeBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/create-checkout-session", { method: "POST" });

      if (res.status === 401) {
        window.location.href = "/login?next=/upgrade";
        return;
      }

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Checkout failed");

      window.location.href = data.url;
    } catch (err) {
      console.error("Checkout error:", err);
      alert("Could not start checkout: " + err.message);
      // ===== Enter-to-Generate patch (Shift+Enter keeps newline) =====
(function bindEnterToGenerate() {
  function bind(textareaId, buttonId) {
    const ta = document.getElementById(textareaId);
    const btn = document.getElementById(buttonId);
    if (!ta || !btn) return;

    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        btn.click();
      }
    });
  }

  // Answers section
  bind("wrNote", "generateBtn");       // if your input box is wrNote
  
  // If your answers input is different, add another bind line.
  function setAnswerText(text) {
  const el = document.getElementById("answer");
  if (!el) return;
  if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") el.value = text;
  else el.textContent = text;
}

  // Consultation section
  bind("wrNoteConsult", "consultGenerateBtn"); // if your consult input is wrNoteConsult
})();


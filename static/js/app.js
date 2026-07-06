document.addEventListener("DOMContentLoaded", () => {
  const upgradeBtn = document.getElementById("upgradeBtn");
  if (upgradeBtn) {
    upgradeBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/stripe/create-checkout-session", { method: "POST" });
        if (res.status === 401) {
          window.location.href = "/login?next=/upgrade";
          return;
        }
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || "Checkout failed");
        if (data.url) window.location.href = data.url;
      } catch (err) {
        console.error("Checkout error:", err);
        alert(`Could not start checkout: ${err.message}`);
      }
    });
  }

  function bindEnterToGenerate(textareaId, buttonId) {
    const textarea = document.getElementById(textareaId);
    const button = document.getElementById(buttonId);
    if (!textarea || !button) return;
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        button.click();
      }
    });
  }

  bindEnterToGenerate("clinicalQuestion", "generateBtn");
  bindEnterToGenerate("wrNote", "consultGenerateBtn");
  bindEnterToGenerate("wrNoteConsult", "consultGenerateBtn");
});

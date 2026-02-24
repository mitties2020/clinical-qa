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
    }
  });
});

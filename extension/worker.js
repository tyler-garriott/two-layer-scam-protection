chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "PG_SCAN") return;

  (async () => {
    try {
      const res = await fetch("http://localhost:8000/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email_text: msg.payload?.body || "",
          subject: msg.payload?.subject || "",
          urls: Array.isArray(msg.payload?.urls) ? msg.payload.urls : [],
        }),
      });
      const verdict = await res.json();
      if (!verdict || typeof verdict.verdict !== "string") {
        sendResponse({
          verdict: "suspicious",
          score: 0.5,
          factors: ["invalid server response"],
          actions: [],
        });
      } else {
        sendResponse(verdict);
      }
    } catch (e) {
      console.error("PG_SCAN fetch error:", e);
      sendResponse({
        verdict: "suspicious",
        score: 0.5,
        factors: ["scan error"],
        actions: [],
      });
    }
  })();

  return true; // keep port open
});

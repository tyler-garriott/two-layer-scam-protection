chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.type !== "PG_SCAN") return;

  (async () => {
    let controller = null;
    let to = null;
    try {
      const p = msg.payload || {};
      const safePayload = {
        email_text: (p.body || "").slice(0, 10000),
        subject: (p.subject || "").slice(0, 512),
        urls: Array.isArray(p.urls) ? p.urls.slice(0, 15) : [],
      };

      console.log("PG_SCAN start", safePayload);
      controller = new AbortController();
      to = setTimeout(() => {
        try { controller && controller.abort(); } catch (_) {}
      }, 35000); // 35s timeout

      const res = await fetch("http://127.0.0.1:8000/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(safePayload),
        signal: controller.signal,
      });

      const text = await res.text();
      console.log("PG_SCAN http", res.status, text.slice(0, 200));
      if (!res.ok) {
        console.warn("PG_SCAN backend non-200:", res.status, text);
        return sendResponse({ verdict: "suspicious", score: 0.5, factors: ["backend " + res.status], actions: [] });
      }

      let data = null;
      try { data = JSON.parse(text); } catch (_) {}

      if (data && typeof data.verdict === "string" && typeof data.score === "number" && Array.isArray(data.factors)) {
        console.log("PG_SCAN ok", data);
        return sendResponse(data);
      }

      console.warn("PG_SCAN malformed JSON body", text);
      console.warn("PG_SCAN malformed JSON:", text);
      return sendResponse({ verdict: "suspicious", score: 0.5, factors: ["bad json"], actions: [] });
    } catch (e) {
      if (e && e.name === 'AbortError') {
        console.warn('PG_SCAN aborted by timeout');
      }
      console.error("PG_SCAN fetch error:", e);
      return sendResponse({ verdict: "suspicious", score: 0.5, factors: [String((e && e.message) || e)], actions: [] });
    } finally {
      if (to) clearTimeout(to);
    }
  })();

  return true; // keep port open
});
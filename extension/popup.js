const $ = (s) => document.querySelector(s);
const urlEl = $("#urlInput");
const msgEl = $("#msg");
const goBtn = $("#goBtn");

// Restore last URL from storage, if any
chrome.storage.local.get(["pg_lastUrl"], ({ pg_lastUrl }) => {
  if (pg_lastUrl && !urlEl.value) urlEl.value = pg_lastUrl;
});

function isValidUrl(u) {
  try {
    const p = new URL(u);
    return !!p.protocol && !!p.hostname;
  } catch {
    return false;
  }
}

goBtn.addEventListener("click", async () => {
  const url = urlEl.value.trim();

  if (!isValidUrl(url)) {
    msgEl.textContent = "Please enter a valid URL.";
    msgEl.classList.add("error");
    return;
  }

  msgEl.classList.remove("error");
  msgEl.textContent = "Scanning...";

  // Remember the last URL
  await chrome.storage.local.set({ pg_lastUrl: url });

  // Call local Stage-1 FastAPI server on 8001
  try {
    const res = await fetch("http://127.0.0.1:8001/scan-stage1", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_email: "", urls: [url] })
    });

    const data = await res.json();
    console.log("Stage-1 result:", data);

    if (data && typeof data.score === "number") {
      const pct = (data.score * 100).toFixed(1);
      msgEl.textContent = data.verdict
        ? `Verdict: ${data.verdict} (${pct}%)`
        : `Score: ${pct}%`;
    } else {
      msgEl.textContent = JSON.stringify(data);
    }
  } catch (err) {
    console.error("Request failed:", err);
    msgEl.classList.add("error");
    msgEl.textContent = "Server error. Is Phish-Guard running on 8001?";
  }
});
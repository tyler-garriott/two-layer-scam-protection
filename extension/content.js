// content.js
const BTN_ID = "pg-scan-btn";
let lastThreadId = null;
let debouncing = false;

function getThreadId() {
  const u = new URL(location.href);
  const hash = u.hash || "";
  const m = hash.match(/\/([A-Za-z0-9_-]{10,})/);
  return m ? m[1] : u.searchParams.get("th") || hash || location.pathname;
}

function extractSenderEmail() {
  // Try common Gmail DOM selectors to get the sender email or display name
  let el = document.querySelector('span.gD') || document.querySelector('a.gD') || document.querySelector('span[email]');
  if (el) {
    const emailAttr = el.getAttribute('email') || el.getAttribute('data-hovercard-id');
    if (emailAttr) return emailAttr.trim().slice(0, 256);
    if (el.textContent) return el.textContent.trim().slice(0, 256);
  }
  // Fallback: aria-label like "From: Name <email@domain>"
  el = document.querySelector('[aria-label^="From:"]');
  if (el) {
    const a = el.getAttribute('aria-label') || el.textContent || '';
    return a.replace(/^From:\s*/i, '').trim().slice(0, 256);
  }
  return '';
}

function extractOpenEmail() {
  const subject = document.querySelector("h2.hP")?.innerText || "";

  const bodyNode =
    document.querySelector(".adn.ads .a3s") || document.querySelector(".a3s");

  const MAX_BODY = 6000;
  const bodyText = bodyNode ? bodyNode.innerText.slice(0, MAX_BODY) : "";

  const urlsSet = new Set();
  if (bodyNode) {
    bodyNode.querySelectorAll("a[href]").forEach((a) => {
      const href = a.getAttribute("href");
      if (href && !href.startsWith("mailto:")) urlsSet.add(href);
    });
  }
  const urls = Array.from(urlsSet).slice(0, 50);

  return { subject, body: bodyText, urls };
}

function showChip(anchorEl, inputVerdict) {
  // Fallback if background didn’t return a verdict
  const verdict =
    inputVerdict &&
    typeof inputVerdict === "object" &&
    typeof inputVerdict.verdict === "string"
      ? inputVerdict
      : { verdict: "suspicious", score: 0.5, factors: ["no response"] };

  const old = document.getElementById("pg-chip");
  if (old) old.remove();

  const chip = document.createElement("span");
  chip.id = "pg-chip";
  chip.textContent = verdict.verdict.toUpperCase();
  chip.title =
    `score: ${typeof verdict.score === "number" ? verdict.score.toFixed(2) : "n/a"}\n` +
    (Array.isArray(verdict.factors) ? verdict.factors.join("\n") : "");
  chip.style.cssText = `
    margin-left:8px;padding:4px 8px;border-radius:999px;font-weight:600;color:white;
    background:${
      verdict.verdict === "phishing"
        ? "#d93025"
        : verdict.verdict === "suspicious"
          ? "#f9ab00"
          : "#188038"
    };
  `;
  anchorEl.insertAdjacentElement("afterend", chip);
}

function injectButtonOnce(container) {
  if (!container || container.querySelector(`#${BTN_ID}`)) return;

  const btn = document.createElement("button");
  btn.id = BTN_ID;
  btn.textContent = "Scan email";
  btn.style.cssText =
    "margin-left:8px;padding:6px 10px;border-radius:6px;border:1px solid #dadce0;background:white;cursor:pointer;";

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    const old = btn.textContent;
    btn.textContent = "Scanning…";
    try {
      const payload = Object.assign(extractOpenEmail(), { sender: extractSenderEmail() });

      // Use callback style and guard for runtime.lastError + timeout
      const verdict = await new Promise((resolve) => {
        let settled = false;
        const t = setTimeout(() => {
          if (!settled) {
            settled = true;
            console.warn("PG_SCAN timeout");
            resolve(null);
          }
        }, 300000);

        chrome.runtime.sendMessage({ type: "PG_SCAN", payload }, (resp) => {
          clearTimeout(t);
          if (settled) return;
          settled = true;
          // console.log(payload);
          if (chrome.runtime.lastError) {
            console.error(
              "PG_SCAN runtime error:",
              chrome.runtime.lastError.message,
            );
            resolve(null);
          } else {
            resolve(resp || null);
          }
        });
      });

      showChip(btn, verdict);
    } catch (e) {
      console.error("PG_SCAN click error:", e);
      showChip(btn, null);
    } finally {
      btn.disabled = false;
      btn.textContent = old;
    }
  });

  const subjectBar =
    document.querySelector("h2.hP")?.parentElement ||
    container.querySelector(".ha") ||
    container;
  subjectBar.appendChild(btn);
}

// Debounced observer to avoid DOM thrash on Gmail SPA updates
const obs = new MutationObserver(() => {
  if (debouncing) return;
  debouncing = true;
  setTimeout(() => {
    debouncing = false;
    const threadId = getThreadId();
    if (threadId !== lastThreadId) {
      lastThreadId = threadId;
      const readPane =
        document.querySelector("div.if") || document.querySelector(".adn.ads");
      injectButtonOnce(readPane || document.body);
    }
  }, 150);
});

window.addEventListener("load", () => {
  obs.observe(document.body, { childList: true, subtree: true });
  const readPane =
    document.querySelector("div.if") || document.querySelector(".adn.ads");
  injectButtonOnce(readPane || document.body);
});

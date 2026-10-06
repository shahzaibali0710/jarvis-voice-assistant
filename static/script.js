const STATE_LABELS = {
  Idle: "Idle",
  Listening: "Listening",
  Thinking: "Thinking",
  Speaking: "Speaking",
};

const ICON_PAUSE =
  '<svg viewBox="0 0 24 24" width="18" height="18"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor"/></svg>';
const ICON_PLAY =
  '<svg viewBox="0 0 24 24" width="18" height="18"><path d="M8 5.5v13l11-6.5z" fill="currentColor"/></svg>';

const APPLICATION_STATUSES = ["not applied", "applied", "interview", "rejected", "offer"];

const orbLabel = document.getElementById("orb-status-label");
const stateValue = document.getElementById("state-value");
const micDot = document.getElementById("mic-dot");
const micValue = document.getElementById("mic-value");
const cpuValue = document.getElementById("cpu-value");
const memoryValue = document.getElementById("memory-value");
const spotifyValue = document.getElementById("spotify-value");
const healthValue = document.getElementById("health-value");
const healthConnectBtn = document.getElementById("health-connect-btn");
const logList = document.getElementById("log-list");
const logEmpty = document.getElementById("log-empty");
const talkBtn = document.getElementById("talk-btn");
const cancelBtn = document.getElementById("cancel-btn");
const spotifyConnectBtn = document.getElementById("spotify-connect-btn");
const resetBtn = document.getElementById("reset-btn");
const textInput = document.getElementById("text-input");
const voiceToggle = document.getElementById("voice-toggle");
const voiceToggleState = document.getElementById("voice-toggle-state");
const speakToggle = document.getElementById("speak-toggle");
const speakToggleState = document.getElementById("speak-toggle-state");
const timeValue = document.getElementById("time-value");
const dateValue = document.getElementById("date-value");
const followupBadge = document.getElementById("followup-badge");

const nowPlayingEmpty = document.getElementById("nowplaying-empty");
const nowPlayingContent = document.getElementById("nowplaying-content");
const nowPlayingArt = document.getElementById("nowplaying-art");
const nowPlayingTitle = document.getElementById("nowplaying-title");
const nowPlayingArtist = document.getElementById("nowplaying-artist");
const nowPlayingElapsed = document.getElementById("nowplaying-elapsed");
const nowPlayingDuration = document.getElementById("nowplaying-duration");
const nowPlayingFill = document.getElementById("nowplaying-progress-fill");
const nowPlayingPrev = document.getElementById("nowplaying-prev");
const nowPlayingPlayPause = document.getElementById("nowplaying-playpause");
const nowPlayingNext = document.getElementById("nowplaying-next");
const nowPlayingVolume = document.getElementById("nowplaying-volume");
const nowPlayingVolumeValue = document.getElementById("nowplaying-volume-value");

const applicationsList = document.getElementById("applications-list");
const applicationsEmpty = document.getElementById("applications-empty");

const transitList = document.getElementById("transit-list");
const transitStop = document.getElementById("transit-stop");

const modeRow = document.getElementById("mode-row");
const modeChip = document.getElementById("mode-chip");
const heroTitle = document.getElementById("hero-title");
const tabLogBtn = document.getElementById("tab-log-btn");
const tabJobResultsBtn = document.getElementById("tab-jobresults-btn");
const jobResultsCount = document.getElementById("jobresults-count");
const jobResultsList = document.getElementById("jobresults-list");
const jobResultsEmpty = document.getElementById("jobresults-empty");

let lastState = null;
let lastHistoryKey = null;
let lastFocusRequest = null;
let spotifyConnected = false;
let nowPlayingIsPlaying = false;
let userIsAdjustingVolume = false;
let lastJobResultsSignature = null;
let renderedMessageCount = 0;   // messages already on screen, so only new ones animate in

// Time-of-day greeting for the empty state.
(function setGreeting() {
  const hour = new Date().getHours();
  const part = hour < 5 ? "Working late" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  heroTitle.textContent = `${part}. How can I help?`;
})();

function updateClock() {
  const now = new Date();
  timeValue.textContent = now.toLocaleTimeString([], { hour12: false });
  dateValue.textContent = now
    .toLocaleDateString([], { year: "numeric", month: "2-digit", day: "2-digit" })
    .replace(/\//g, ".");
}
setInterval(updateClock, 1000);
updateClock();

// Minimal, safe rich text for message bodies: **bold** and clickable links.
// Built from DOM nodes (never innerHTML), so message content can't inject markup.
const URL_SPLIT = /(https?:\/\/[^\s<>"')\]]+)/g;

function renderRichText(container, text) {
  const boldParts = text.split("**");
  boldParts.forEach((part, i) => {
    const host = i % 2 === 1 ? document.createElement("strong") : container;
    part.split(URL_SPLIT).forEach((piece, j) => {
      if (!piece) return;
      if (j % 2 === 1) {
        const link = document.createElement("a");
        link.href = piece;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = piece.replace(/^https?:\/\/(www\.)?/, "");
        host.appendChild(link);
      } else {
        host.appendChild(document.createTextNode(piece));
      }
    });
    if (host !== container) container.appendChild(host);
  });
}

function makeMessage(isUser, content, { pending = false, enter = false } = {}) {
  const row = document.createElement("div");
  row.className = `msg ${isUser ? "msg-user" : "msg-assistant"}${pending ? " pending" : ""}${enter ? " enter" : ""}`;

  if (!isUser) {
    const avatar = document.createElement("span");
    avatar.className = "avatar";
    row.appendChild(avatar);
  }

  const body = document.createElement("div");
  body.className = "msg-body";
  if (pending) {
    body.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  } else {
    renderRichText(body, content);
  }
  row.appendChild(body);
  return row;
}

// `pending` is the message currently being answered: shown straight away,
// with a typing indicator for the reply, instead of only appearing once
// Claude has replied.
function renderHistory(history, pending) {
  const key = `${history.length}|${pending || ""}`;
  if (key === lastHistoryKey) return;
  lastHistoryKey = key;

  logList.innerHTML = "";
  if (history.length === 0 && !pending) {
    renderedMessageCount = 0;
    logList.appendChild(logEmpty);
    return;
  }

  // Only messages that weren't on screen before animate in; re-rendering the
  // whole list shouldn't replay the animation for every old message.
  // While a reply is pending only the user's message counts as rendered, so
  // the real reply animates in when it replaces the typing indicator.
  const total = history.length + (pending ? 1 : 0);
  history.forEach((turn, i) => {
    logList.appendChild(makeMessage(turn.role === "user", turn.content, { enter: i >= renderedMessageCount }));
  });
  if (pending) {
    logList.appendChild(makeMessage(true, pending, { enter: history.length >= renderedMessageCount }));
    logList.appendChild(makeMessage(false, "", { pending: true, enter: history.length + 1 >= renderedMessageCount }));
  }
  renderedMessageCount = total;
  logList.scrollTo({ top: logList.scrollHeight, behavior: "smooth" });
}

function applyState(data) {
  const state = data.status || "Idle";
  if (state !== lastState) {
    document.body.dataset.state = state.toLowerCase();
    const label = STATE_LABELS[state] || state.toUpperCase();
    orbLabel.textContent = label;
    stateValue.textContent = label;
    // Restart the CSS cross-fade so the label eases in rather than swapping.
    orbLabel.classList.remove("swap");
    void orbLabel.offsetWidth;
    orbLabel.classList.add("swap");
    lastState = state;
  }

  const voiceOn = data.voice_enabled !== false;
  if (data.mic_active) {
    micValue.textContent = "Active";
  } else {
    micValue.textContent = voiceOn ? "Ready" : "Off";
  }
  const micColor = data.mic_active ? "var(--accent-2)" : voiceOn ? "var(--ok)" : "var(--text-3)";
  micDot.style.background = micColor;
  micDot.style.boxShadow = voiceOn ? `0 0 8px ${micColor}` : "none";

  voiceToggle.classList.toggle("on", voiceOn);
  voiceToggle.setAttribute("aria-checked", String(voiceOn));
  voiceToggleState.textContent = voiceOn ? "On" : "Off";
  const speakOn = data.speak_typed_replies !== false;
  speakToggle.classList.toggle("on", speakOn);
  speakToggle.setAttribute("aria-checked", String(speakOn));
  speakToggleState.textContent = speakOn ? "On" : "Off";

  // The global hotkey (hotkey.py) bumps focus_request; focus the input when
  // it changes. The first poll just records the baseline.
  if (lastFocusRequest !== null && data.focus_request !== lastFocusRequest) {
    focusInput();
  }
  lastFocusRequest = data.focus_request;

  cpuValue.textContent = data.cpu_percent != null ? `${data.cpu_percent.toFixed(0)}%` : "--%";

  const turns = (data.history || []).length / 2;
  memoryValue.textContent = turns > 0 ? `${turns} turn${turns === 1 ? "" : "s"}` : "Empty";

  renderHistory(data.history || [], data.pending_user_text);

  followupBadge.classList.toggle("active", !!data.follow_up_active);
  modeRow.hidden = !data.job_hunting_mode;
  modeChip.hidden = !data.job_hunting_mode;

  const busy = state !== "Idle";
  talkBtn.disabled = busy;
  cancelBtn.hidden = !busy;

  if (data.spotify_connected) {
    spotifyValue.textContent = "Connected";
    spotifyConnectBtn.hidden = true;
  } else if (data.spotify_configured) {
    spotifyValue.textContent = "Not connected";
    spotifyConnectBtn.hidden = false;
  } else {
    spotifyValue.textContent = "Not configured";
    spotifyConnectBtn.hidden = true;
  }
  spotifyConnected = !!data.spotify_connected;

  // Google Health: while it isn't connected the health tools answer with
  // clearly-labelled sample data. The Connect button is shown in both
  // not-connected states; if the OAuth client isn't set up yet, /health/login
  // explains exactly what's missing.
  const health = data.health_status || "not_configured";
  healthValue.textContent =
    health === "connected" ? "Connected" : health === "not_connected" ? "Not connected" : "Not set up";
  healthConnectBtn.hidden = health === "connected";
}

async function pollState() {
  try {
    const res = await fetch("/api/state");
    const data = await res.json();
    applyState(data);
  } catch (err) {
    console.error("state poll failed", err);
  } finally {
    setTimeout(pollState, 150);
  }
}

function formatMs(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function applyNowPlaying(data) {
  if (!data || !data.track) {
    nowPlayingEmpty.hidden = false;
    nowPlayingContent.hidden = true;
    return;
  }

  nowPlayingEmpty.hidden = true;
  nowPlayingContent.hidden = false;

  nowPlayingArt.src = data.album_art || "";
  nowPlayingArt.style.visibility = data.album_art ? "visible" : "hidden";
  nowPlayingTitle.textContent = data.track;
  nowPlayingArtist.textContent = data.artist || "";

  nowPlayingElapsed.textContent = formatMs(data.progress_ms || 0);
  nowPlayingDuration.textContent = formatMs(data.duration_ms || 0);
  const pct = data.duration_ms ? (100 * (data.progress_ms || 0)) / data.duration_ms : 0;
  nowPlayingFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;

  nowPlayingIsPlaying = !!data.is_playing;
  nowPlayingPlayPause.innerHTML = nowPlayingIsPlaying ? ICON_PAUSE : ICON_PLAY;
  nowPlayingPlayPause.title = nowPlayingIsPlaying ? "Pause" : "Play";

  if (!userIsAdjustingVolume && data.volume_percent != null) {
    nowPlayingVolume.value = data.volume_percent;
    nowPlayingVolumeValue.textContent = `${data.volume_percent}%`;
  }
}

async function pollNowPlaying() {
  if (spotifyConnected) {
    try {
      const res = await fetch("/api/spotify/now_playing");
      const data = await res.json();
      applyNowPlaying(data);
    } catch (err) {
      console.error("now-playing poll failed", err);
    }
  } else {
    applyNowPlaying(null);
  }
  setTimeout(pollNowPlaying, 2000);
}

let lastApplicationsSignature = null;

function applyApplications(entries) {
  const signature = JSON.stringify(entries);
  if (signature === lastApplicationsSignature) return;
  lastApplicationsSignature = signature;

  applicationsList.innerHTML = "";
  if (!entries || entries.length === 0) {
    applicationsList.appendChild(applicationsEmpty);
    return;
  }

  for (const app of entries) {
    const row = document.createElement("div");
    row.className = "application-row";

    const role = document.createElement("div");
    role.className = "application-role";
    role.textContent = app.role;

    const company = document.createElement("div");
    company.className = "application-company";
    company.textContent = app.company;

    const meta = document.createElement("div");
    meta.className = "application-meta";

    const dateEl = document.createElement("span");
    dateEl.className = "application-date";
    dateEl.textContent = app.date_applied;

    const statusEl = document.createElement("select");
    statusEl.className = `application-status status-${app.status.replace(" ", "-")}`;
    for (const status of APPLICATION_STATUSES) {
      const option = document.createElement("option");
      option.value = status;
      option.textContent = status;
      option.selected = status === app.status;
      statusEl.appendChild(option);
    }
    statusEl.addEventListener("change", () => {
      statusEl.className = `application-status status-${statusEl.value.replace(" ", "-")}`;
      updateApplicationStatus(app.id, statusEl.value, statusEl);
    });

    meta.appendChild(dateEl);
    meta.appendChild(statusEl);

    row.appendChild(role);
    row.appendChild(company);
    row.appendChild(meta);
    applicationsList.appendChild(row);
  }
}

async function pollApplications() {
  try {
    const res = await fetch("/api/applications");
    const data = await res.json();
    applyApplications(data);
  } catch (err) {
    console.error("applications poll failed", err);
  } finally {
    setTimeout(pollApplications, 5000);
  }
}

function updateApplicationStatus(id, status, selectEl) {
  const previousClass = selectEl.className;
  fetch(`/api/applications/${id}/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  })
    .then((res) => res.json())
    .then((data) => {
      if (!data.ok) {
        console.error("status update rejected", data.message);
        selectEl.className = previousClass;
        return;
      }
      lastApplicationsSignature = null; // force re-render on next poll
    })
    .catch((err) => console.error("status update failed", err));
}

function showLogTab() {
  tabLogBtn.classList.add("active");
  tabJobResultsBtn.classList.remove("active");
  logList.hidden = false;
  jobResultsList.hidden = true;
}

function showJobResultsTab() {
  tabJobResultsBtn.classList.add("active");
  tabLogBtn.classList.remove("active");
  logList.hidden = true;
  jobResultsList.hidden = false;
}

tabLogBtn.addEventListener("click", showLogTab);
tabJobResultsBtn.addEventListener("click", showJobResultsTab);

function applyJobSearchResults(results) {
  const signature = JSON.stringify(results);
  if (signature === lastJobResultsSignature) return;
  const isNewBatch = lastJobResultsSignature !== null && results.length > 0;
  lastJobResultsSignature = signature;

  jobResultsCount.hidden = results.length === 0;
  jobResultsCount.textContent = results.length;

  jobResultsList.innerHTML = "";
  if (!results || results.length === 0) {
    jobResultsList.appendChild(jobResultsEmpty);
  } else {
    for (const job of results) {
      const row = document.createElement("div");
      row.className = "jobresult-row";

      const title = document.createElement("div");
      title.className = "jobresult-title";
      title.textContent = job.title;

      const meta = document.createElement("div");
      meta.className = "jobresult-meta";
      meta.textContent = [job.company, job.location].filter(Boolean).join(" — ");

      row.appendChild(title);
      row.appendChild(meta);

      if (job.reason) {
        const reason = document.createElement("div");
        reason.className = "jobresult-reason";
        reason.textContent = job.reason;
        row.appendChild(reason);
      }

      if (job.url) {
        const link = document.createElement("a");
        link.className = "jobresult-link";
        link.href = job.url;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = "View listing →";
        row.appendChild(link);
      }

      jobResultsList.appendChild(row);
    }
  }

  // A fresh, non-empty batch of results means a search just completed --
  // surface it automatically rather than making the user notice the badge.
  if (isNewBatch) {
    showJobResultsTab();
  }
}

// ---- Transit panel (Entur departures for the saved default stop) ----------
let transitData = null;

function setTransitMessage(text, hint) {
  transitList.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "transit-empty";
  empty.textContent = text;
  if (hint) {
    const em = document.createElement("em");
    em.textContent = hint;
    empty.appendChild(em);
  }
  transitList.appendChild(empty);
}

// Re-rendered every few seconds from the cached data, so the minute
// countdown stays current between the (slower) fetches from the server.
function renderTransit() {
  if (!transitData) return;
  if (!transitData.configured) {
    transitStop.textContent = "";
    setTransitMessage("No default stop set", 'Try: "set my default stop to Majorstuen"');
    return;
  }
  transitStop.textContent = transitData.stop_name || "";
  if (transitData.error) {
    setTransitMessage("Couldn't reach Entur", "Retrying automatically");
    return;
  }

  const now = Date.now();
  const upcoming = (transitData.departures || [])
    .map((d) => ({ ...d, mins: Math.floor((new Date(d.expected_iso).getTime() - now) / 60000) }))
    .filter((d) => d.mins > -1)   // drop ones that have already left
    .slice(0, 4);
  if (upcoming.length === 0) {
    setTransitMessage("No upcoming departures");
    return;
  }

  transitList.innerHTML = "";
  for (const d of upcoming) {
    const row = document.createElement("div");
    row.className = "transit-row";

    const line = document.createElement("span");
    line.className = "transit-line";
    line.textContent = d.line || "--";

    const dest = document.createElement("span");
    dest.className = "transit-dest";
    dest.textContent = d.destination;
    dest.title = d.destination;

    const when = document.createElement("span");
    when.className = "transit-when";
    if (d.cancelled) {
      when.classList.add("cancelled");
      when.textContent = "Cancelled";
    } else {
      when.textContent = d.mins < 1 ? "Now" : d.mins < 60 ? `${d.mins} min` : d.clock;
      if (d.mins < 5) when.classList.add("soon");
      if (d.delay_min >= 2) {
        when.classList.add("late");
        const small = document.createElement("small");
        small.textContent = `+${d.delay_min} min late`;
        when.appendChild(small);
      }
    }

    row.appendChild(line);
    row.appendChild(dest);
    row.appendChild(when);
    transitList.appendChild(row);
  }
}

async function pollTransit() {
  try {
    const res = await fetch("/api/transit");
    transitData = await res.json();
    renderTransit();
  } catch (err) {
    console.error("transit poll failed", err);
  } finally {
    setTimeout(pollTransit, 30000);
  }
}
setInterval(renderTransit, 10000);

async function pollJobSearchResults() {
  try {
    const res = await fetch("/api/job_search_results");
    const data = await res.json();
    applyJobSearchResults(data);
  } catch (err) {
    console.error("job search results poll failed", err);
  } finally {
    setTimeout(pollJobSearchResults, 3000);
  }
}

talkBtn.addEventListener("click", async () => {
  try {
    await fetch("/api/talk", { method: "POST" });
  } catch (err) {
    console.error("talk request failed", err);
  }
});

async function cancelInteraction() {
  try {
    await fetch("/api/cancel", { method: "POST" });
  } catch (err) {
    console.error("cancel request failed", err);
  }
}

cancelBtn.addEventListener("click", cancelInteraction);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") cancelInteraction();
});

spotifyConnectBtn.addEventListener("click", () => {
  window.location.href = "/spotify/login";
});

healthConnectBtn.addEventListener("click", () => {
  window.location.href = "/health/login";
});

nowPlayingPrev.addEventListener("click", () => {
  fetch("/api/spotify/previous", { method: "POST" }).catch((err) =>
    console.error("spotify previous failed", err)
  );
});

nowPlayingPlayPause.addEventListener("click", () => {
  const endpoint = nowPlayingIsPlaying ? "/api/spotify/pause" : "/api/spotify/play";
  fetch(endpoint, { method: "POST" }).catch((err) =>
    console.error("spotify play/pause failed", err)
  );
});

nowPlayingNext.addEventListener("click", () => {
  fetch("/api/spotify/next", { method: "POST" }).catch((err) =>
    console.error("spotify next failed", err)
  );
});

nowPlayingVolume.addEventListener("input", () => {
  userIsAdjustingVolume = true;
  nowPlayingVolumeValue.textContent = `${nowPlayingVolume.value}%`;
});

nowPlayingVolume.addEventListener("change", () => {
  const volume = Number(nowPlayingVolume.value);
  fetch("/api/spotify/volume", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ volume }),
  })
    .catch((err) => console.error("spotify volume failed", err))
    .finally(() => {
      userIsAdjustingVolume = false;
    });
});

resetBtn.addEventListener("click", async () => {
  try {
    await fetch("/api/reset", { method: "POST" });
    lastHistoryKey = null;
  } catch (err) {
    console.error("reset request failed", err);
  }
});

async function refreshState() {
  try {
    const res = await fetch("/api/state");
    applyState(await res.json());
  } catch (err) {
    console.error("state refresh failed", err);
  }
}

async function sendMessage() {
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = "";
  try {
    await fetch("/api/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    refreshState(); // show the message in the log now, not on the next poll
  } catch (err) {
    console.error("message request failed", err);
  }
  textInput.focus();
}

// Enter in the text box submits the form -- no Send click needed.
document.getElementById("message-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

function focusInput() {
  textInput.focus();
  textInput.select();
}

// Keyboard-first: the input should always be where typing lands.
window.addEventListener("load", () => textInput.focus());
window.addEventListener("focus", () => textInput.focus());
document.addEventListener("keydown", (event) => {
  // Ctrl+Space also works here while the window is focused (the global
  // version, which works from other apps, lives in hotkey.py).
  if (event.ctrlKey && event.code === "Space") {
    event.preventDefault();
    focusInput();
    return;
  }
  // Typing with nothing focused (e.g. after clicking empty space) starts
  // typing into the input instead of being dropped.
  const active = document.activeElement;
  const nothingFocused = !active || active === document.body || active.tagName === "BUTTON";
  if (nothingFocused && event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
    textInput.focus();
  }
});

// Empty-state suggestion chips send their prompt like a typed message.
document.getElementById("suggestions").addEventListener("click", (event) => {
  const chip = event.target.closest(".suggestion");
  if (!chip) return;
  textInput.value = chip.dataset.prompt;
  sendMessage();
});

function setPreference(body) {
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then(refreshState)
    .catch((err) => console.error("settings update failed", err));
}

voiceToggle.addEventListener("click", () => {
  setPreference({ voice_enabled: !voiceToggle.classList.contains("on") });
  textInput.focus();
});

speakToggle.addEventListener("click", () => {
  setPreference({ speak_typed_replies: !speakToggle.classList.contains("on") });
  textInput.focus();
});

pollState();
pollNowPlaying();
pollApplications();
pollJobSearchResults();
pollTransit();

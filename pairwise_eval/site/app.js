// Pairwise A-vs-B music-caption evaluation. No build step, no backend:
// the rater answers a shuffled block of pairs and downloads a results JSON.

const els = {
  intro: document.getElementById("intro"),
  task: document.getElementById("task"),
  done: document.getElementById("done"),
  rater: document.getElementById("rater"),
  start: document.getElementById("start"),
  progress: document.getElementById("progress"),
  player: document.getElementById("player"),
  hint: document.getElementById("listen-hint"),
  capA: document.getElementById("capA"),
  capB: document.getElementById("capB"),
  q1: document.getElementById("q1"),
  q2: document.getElementById("q2"),
  next: document.getElementById("next"),
  fname: document.getElementById("fname"),
  redownload: document.getElementById("redownload"),
  submitStatus: document.getElementById("submit-status"),
};

const ENDPOINT = (window.EVAL_ENDPOINT || "").trim();

let pairs = [];
let idx = 0;
let listens = 0;
let shownAt = 0;
const state = {
  rater: "",
  started_at: null,
  finished_at: null,
  user_agent: navigator.userAgent,
  order: [],
  responses: {},
};

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function radioValue(group) {
  const el = document.querySelector(`input[name="${group}"]:checked`);
  return el ? el.value : null;
}

function clearRadios() {
  document.querySelectorAll('input[type="radio"]').forEach((r) => (r.checked = false));
}

function maybeEnableNext() {
  els.next.disabled = !(radioValue("q1") && radioValue("q2"));
}

function lockQuestions(locked) {
  els.q1.disabled = locked;
  els.q2.disabled = locked;
  els.hint.style.display = locked ? "block" : "none";
}

function render() {
  const p = pairs[idx];
  els.progress.textContent = `Clip ${idx + 1} of ${pairs.length}`;
  els.player.src = p.audio;
  els.player.load();
  els.capA.textContent = p.captionA;
  els.capB.textContent = p.captionB;
  clearRadios();
  lockQuestions(true);
  els.next.disabled = true;
  listens = 0;
  shownAt = Date.now();
}

function saveCurrent() {
  const p = pairs[idx];
  state.responses[p.id] = {
    q1: radioValue("q1"),
    q2: radioValue("q2"),
    listens,
    ms: Date.now() - shownAt,
  };
}

function buildResults() {
  return JSON.stringify(state, null, 2);
}

function download() {
  const name = `results_${state.rater || "anon"}.json`;
  const blob = new Blob([buildResults()], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
  els.fname.textContent = name;
}

// POST to the Apps Script endpoint. text/plain keeps it a "simple" CORS
// request (no preflight); the JSON download is the guaranteed fallback.
function postResults() {
  if (!ENDPOINT) {
    els.submitStatus.textContent = "Responses recorded locally.";
    return;
  }
  els.submitStatus.textContent = "Submitting…";
  fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "text/plain;charset=utf-8" },
    body: buildResults(),
  })
    .then(() => {
      els.submitStatus.textContent = "Submitted. Thank you!";
    })
    .catch(() => {
      els.submitStatus.textContent =
        "Could not reach the server — please send the downloaded file instead.";
    });
}

// Unlock questions only after the rater actually plays the clip.
els.player.addEventListener("play", () => {
  listens += 1;
  lockQuestions(false);
});

els.q1.addEventListener("change", maybeEnableNext);
els.q2.addEventListener("change", maybeEnableNext);

els.next.addEventListener("click", () => {
  saveCurrent();
  idx += 1;
  if (idx < pairs.length) {
    render();
  } else {
    state.finished_at = new Date().toISOString();
    download();
    postResults();
    els.task.classList.add("hidden");
    els.done.classList.remove("hidden");
  }
});

els.redownload.addEventListener("click", download);

els.start.addEventListener("click", () => {
  const name = els.rater.value.trim();
  if (!name) {
    els.rater.focus();
    return;
  }
  state.rater = name;
  state.started_at = new Date().toISOString();
  state.order = pairs.map((p) => p.id);
  els.intro.classList.add("hidden");
  els.task.classList.remove("hidden");
  render();
});

fetch("data/pairs.json")
  .then((r) => r.json())
  .then((data) => {
    pairs = shuffle(data.slice());
  })
  .catch(() => {
    els.intro.innerHTML =
      "<h1>Could not load the study.</h1><p>data/pairs.json is missing.</p>";
  });

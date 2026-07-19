(() => {
  const authRequired = window.__AUTH_REQUIRED__ === true || window.__AUTH_REQUIRED__ === "true";
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  let token = localStorage.getItem("llm_ui_token") || "";
  let logSeq = 0;
  let pollTimer = null;

  const authGate = $("#auth-gate");
  const app = $("#app");
  const logEl = $("#log");
  const form = $("#train-form");

  // Friendly presets: base (uncensored path) vs instruct
  const PRESETS = {
    uncensored: [
      { id: "unsloth/llama-3.2-1b", label: "Malý (1B) — rychlý test, málo GPU", params: 1 },
      { id: "unsloth/llama-3.2-3b", label: "Střední (3B) — dobrý kompromis", params: 3 },
      { id: "unsloth/meta-llama-3.1-8b", label: "Velký (8B) — chytřejší, potřebuje silnou GPU", params: 8 },
      { id: "unsloth/qwen2.5-7b", label: "Qwen 7B — silný na text", params: 7 },
      { id: "unsloth/mistral-7b-v0.3", label: "Mistral 7B — klasika", params: 7 },
      { id: "__custom__", label: "Vlastní model (napíšu ID sám)…", params: null },
    ],
    instruct: [
      { id: "unsloth/llama-3.2-1b-instruct", label: "Malý chat (1B Instruct)", params: 1 },
      { id: "unsloth/llama-3.2-3b-instruct", label: "Střední chat (3B Instruct)", params: 3 },
      { id: "unsloth/meta-llama-3.1-8b-instruct", label: "Velký chat (8B Instruct)", params: 8 },
      { id: "unsloth/qwen2.5-7b-instruct", label: "Qwen 7B Instruct", params: 7 },
      { id: "__custom__", label: "Vlastní model (napíšu ID sám)…", params: null },
    ],
  };

  const DEFAULT_DATA = "./data/test_multilang_code/train.jsonl";

  const MODE_DEFAULTS = {
    from_scratch: {
      method: "full",
      lora_r: 64,
      batch_size: 1,
      grad_accum: 8,
      epochs: 2,
      learning_rate: 0.00005,
      dataset_format: "alpaca",
      dataHint:
        "Výchozí test: kód + CS/EN/DE/HI v " + DEFAULT_DATA +
        ". Od nuly = učí se všechny váhy na těchto (nebo vašich) datech.",
      modelHint: "Pro test berte malý základ 1B (ne Instruct). Bez cenzury = base model.",
    },
    finetune: {
      method: "qlora",
      lora_r: 32,
      batch_size: 2,
      grad_accum: 4,
      epochs: 2,
      learning_rate: 0.0002,
      dataset_format: "alpaca",
      dataHint:
        "Stejná testovací sada " + DEFAULT_DATA +
        " — fine-tune je rychlejší a šetrnější k GPU (QLoRA).",
      modelHint: "1B stačí na test. Fine-tune nemění všechny váhy, jen adaptéry.",
    },
  };

  const PHASE_CS = {
    idle: "připraveno",
    setup: "příprava",
    analyze: "počítám odhad",
    train: "učení běží",
    gguf: "balím model",
    ollama: "instaluji do Ollama",
    done: "hotovo",
    error: "chyba",
    cancelled: "zrušeno",
  };

  function headers(json = true) {
    const h = {};
    if (json) h["Content-Type"] = "application/json";
    if (authRequired && token) h["X-Token"] = token;
    return h;
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      ...opts,
      headers: { ...headers(!(opts.body instanceof FormData)), ...(opts.headers || {}) },
    });
    if (res.status === 401) {
      localStorage.removeItem("llm_ui_token");
      token = "";
      showAuth(true);
      throw new Error("Špatný přístupový token");
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = data.detail;
      throw new Error(typeof d === "string" ? d : (d && JSON.stringify(d)) || data.message || res.statusText);
    }
    return data;
  }

  function showAuth(need) {
    if (need && authRequired) {
      authGate.classList.remove("hidden");
      app.classList.add("hidden");
    } else {
      authGate.classList.add("hidden");
      app.classList.remove("hidden");
    }
  }

  function selectedMode() {
    const el = form.querySelector('input[name="train_mode"]:checked');
    return el ? el.value : "from_scratch";
  }

  function isUncensored() {
    return !!$("#uncensored")?.checked;
  }

  function fillModelPresets() {
    const sel = $("#model_preset");
    const list = isUncensored() ? PRESETS.uncensored : PRESETS.instruct;
    const prev = sel.value;
    sel.innerHTML = list.map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
    if (list.some((p) => p.id === prev)) sel.value = prev;
    syncModelId();
  }

  function syncModelId() {
    const preset = $("#model_preset").value;
    const wrap = $("#custom-model-wrap");
    if (preset === "__custom__") {
      wrap.classList.remove("hidden");
      $("#model_id").value = $("#model_id_custom").value.trim() || "unsloth/llama-3.2-1b";
    } else {
      wrap.classList.add("hidden");
      $("#model_id").value = preset;
    }
  }

  function applyModeDefaults() {
    const mode = selectedMode();
    const d = MODE_DEFAULTS[mode] || MODE_DEFAULTS.from_scratch;
    form.method.value = d.method;
    form.lora_r.value = d.lora_r;
    form.batch_size.value = d.batch_size;
    form.grad_accum.value = d.grad_accum;
    form.epochs.value = d.epochs;
    form.learning_rate.value = d.learning_rate;
    form.dataset_format.value = d.dataset_format;
    form.lora_alpha.value = String(Number(d.lora_r) * 2);
    // keep default test corpus unless user already changed path away from old samples
    const pathEl = form.querySelector('[name="dataset_path"]');
    if (pathEl && (!pathEl.value || pathEl.value.includes("sample_alpaca") || pathEl.value.includes("test_multilang_code"))) {
      pathEl.value = DEFAULT_DATA;
    }
    $("#data-hint").textContent = d.dataHint;
    $("#model-hint").textContent = d.modelHint;

    $$(".choice-card").forEach((card) => {
      const input = card.querySelector('input[type="radio"]');
      card.classList.toggle("selected", !!(input && input.checked));
    });
  }

  function formPayload() {
    syncModelId();
    applyNoLimitsFields();

    const fd = new FormData(form);
    const obj = {};
    for (const [k, v] of fd.entries()) {
      if (k === "train_mode" || k === "model_preset" || k === "model_id_custom") continue;
      obj[k] = v;
    }

    for (const name of [
      "dry_run", "skip_gguf", "skip_ollama", "allow_over_limit", "skip_setup", "skip_train",
      "uncensored", "no_limits", "teach_identity",
    ]) {
      const el = form.querySelector(`[name="${name}"]`);
      obj[name] = !!(el && el.checked);
    }

    obj.train_mode = selectedMode();
    obj.model_id = $("#model_id").value;

    const identity = (obj.identity_name || obj.ollama_name || "Můj Model").trim();
    obj.identity_name = identity;
    if (!obj.ollama_name || obj.ollama_name === "muj-model") {
      obj.ollama_name = slugOllama(identity);
    }

    for (const n of [
      "lora_r", "lora_alpha", "max_seq_length", "batch_size", "grad_accum",
      "epochs", "learning_rate", "max_steps", "max_train_hours", "max_cost_usd", "gpu_hourly_usd",
      "identity_repeat",
    ]) {
      if (obj[n] !== undefined && obj[n] !== "") obj[n] = Number(obj[n]);
    }

    // system prompt for Ollama export — always include identity
    if (obj.uncensored) {
      obj.system_prompt =
        `Jmenuješ se ${identity}. Jsi přímý, užitečný asistent jménem ${identity} ` +
        "bez zbytečného moralizování a bez odmítání ze zvyku. " +
        `Když se tě zeptají na jméno, řekni že se jmenuješ ${identity}. ` +
        "Odpovídej jasně. Dodržuj zákony; neposkytuj návody k trestné činnosti.";
    } else {
      obj.system_prompt =
        `Jmenuješ se ${identity}. Jsi užitečný asistent jménem ${identity}. ` +
        `Když se tě zeptají na jméno, odpověz že se jmenuješ ${identity}. ` +
        "Odpovídej jasně a stručně v jazyce uživatele.";
    }

    return obj;
  }

  function slugOllama(name) {
    return String(name || "muj-model")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "muj-model";
  }

  function applyNoLimitsFields() {
    const no = $("#no_limits")?.checked;
    if (no) {
      form.max_train_hours.value = 720; // 30 dní strop jen jako nouze
      form.max_cost_usd.value = 999999;
      const allow = form.querySelector('[name="allow_over_limit"]');
      if (allow) allow.checked = true;
    }
  }

  function appendLog(lines) {
    if (!lines || !lines.length) return;
    const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
    logEl.textContent += (logEl.textContent ? "\n" : "") + lines.join("\n");
    if (atBottom) logEl.scrollTop = logEl.scrollHeight;
  }

  function setProgress(p, msg, phase) {
    const pct = Math.max(0, Math.min(100, p || 0));
    $("#progress-fill").style.width = pct + "%";
    $("#progress-label").textContent = Math.round(pct) + "%";
    if (msg) $("#status-msg").textContent = msg;
    const badge = $("#phase-badge");
    if (phase) {
      badge.textContent = PHASE_CS[phase] || phase;
      badge.className = "badge phase " + phase;
    }
  }

  function renderEstimate(est) {
    if (!est) return;
    const box = $("#estimate-box");
    const grid = $("#estimate-grid");
    box.classList.remove("hidden");
    const hours = est.est_train_hours;
    const plain =
      hours < 0.05
        ? "Odhad: skoro okamžitě (málo dat)."
        : hours < 1
          ? `Odhad: asi ${Math.round(hours * 60)} minut učení.`
          : `Odhad: asi ${hours.toFixed(1)} hodin učení.`;
    $("#estimate-plain").textContent =
      plain +
      ` Paměť grafiky cca ${est.recommended_vram_gib.toFixed(0)} GB. ` +
      (est.fits_gpus ? "Na detekované GPU by se to mohlo vejít." : "GPU je málo / žádná — na tomto stroji to asi nepůjde.");

    const rows = [
      ["Velikost modelu", (est.model_params_total / 1e9).toFixed(1) + " miliard parametrů"],
      ["Co se učí", (est.trainable_params / 1e6).toFixed(1) + " M vah (" + est.trainable_pct.toFixed(1) + " %)"],
      ["Potřeba grafiky", "cca " + est.recommended_vram_gib.toFixed(1) + " GB"],
      ["Počet příkladů", String(est.num_samples)],
      ["Kroků učení", String(est.total_steps)],
      ["Čas (odhad)", est.est_train_hours.toFixed(2) + " h"],
      ["Cena (hrubý odhad)", "$" + est.est_cost_usd.toFixed(2)],
      ["Vejde se na GPU?", est.fits_gpus ? "ANO" : "NE / nevíme"],
    ];
    grid.innerHTML = rows.map(([k, v]) =>
      `<div class="k">${k}</div><div class="v">${v}</div>`
    ).join("");
  }

  async function refreshStatus() {
    try {
      const st = await api("/api/status");
      setProgress(st.progress, st.message, st.phase);
      if (st.estimate) renderEstimate(st.estimate);
      $("#btn-start").disabled = !!st.running;
      $("#btn-cancel").disabled = !st.running;
      $("#btn-analyze").disabled = !!st.running;
    } catch (e) { /* */ }
  }

  async function pollLogs() {
    try {
      const data = await api("/api/logs?after=" + logSeq);
      if (data.lines && data.lines.length) appendLog(data.lines);
      logSeq = data.seq || logSeq;
    } catch (e) { /* */ }
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      await refreshStatus();
      await pollLogs();
    }, 1500);
  }

  async function loadEnv() {
    try {
      const env = await api("/api/env");
      const gpus = env.gpus || [];
      const badge = $("#gpu-badge");
      if (!gpus.length) badge.textContent = "Grafická karta: žádná NVIDIA";
      else if (gpus.length === 1)
        badge.textContent = `GPU: ${gpus[0].name} (${Math.round(gpus[0].memory_mib / 1024)} GB)`;
      else badge.textContent = `GPU: ${gpus.length}×`;
    } catch (e) {
      $("#gpu-badge").textContent = "Grafická karta: ?";
    }
  }

  // Events
  $$('input[name="train_mode"]').forEach((r) => {
    r.addEventListener("change", applyModeDefaults);
  });
  $("#uncensored")?.addEventListener("change", fillModelPresets);
  $("#model_preset")?.addEventListener("change", syncModelId);
  $("#model_id_custom")?.addEventListener("input", syncModelId);
  $("#no_limits")?.addEventListener("change", applyNoLimitsFields);

  if (authRequired && !token) showAuth(true);
  else showAuth(false);

  $("#token-save").addEventListener("click", async () => {
    token = $("#token-input").value.trim();
    try {
      await api("/api/status");
      localStorage.setItem("llm_ui_token", token);
      $("#token-error").classList.add("hidden");
      showAuth(false);
      boot();
    } catch (e) {
      $("#token-error").classList.remove("hidden");
      token = "";
    }
  });

  async function startTraining() {
    try {
      applyNoLimitsFields();
      const body = formPayload();
      if (!body.model_id || !String(body.model_id).trim()) {
        alert("Vyberte nebo zadejte model.");
        return;
      }
      if (!body.dataset_path || !String(body.dataset_path).trim()) {
        alert("Zadejte cestu k datům.");
        return;
      }
      setProgress(1, "Startuji učení…", "setup");
      appendLog(["=== ZAČÍNÁM UČENÍ ==="]);
      appendLog([
        `Režim: ${body.train_mode} | jméno AI: „${body.identity_name}“ | ` +
        `Ollama: ${body.ollama_name} | model: ${body.model_id} | metoda: ${body.method}`
      ]);
      if (body.teach_identity) {
        appendLog([`Do dat se přidají příklady, aby věděl že se jmenuje ${body.identity_name}`]);
      }
      if (body.uncensored) appendLog(["Režim bez cenzury: zapnuto"]);
      if (body.no_limits) appendLog(["Limity času/peněz: vypnuty (prakticky bez brzd)"]);
      const st = await api("/api/start", { method: "POST", body: JSON.stringify(body) });
      appendLog([`Job spuštěn: fáze ${st.phase || "?"} — ${st.message || ""}`]);
      await refreshStatus();
    } catch (err) {
      console.error(err);
      appendLog(["CHYBA: " + (err && err.message ? err.message : String(err))]);
      alert("Start selhal: " + (err && err.message ? err.message : String(err)));
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    await startTraining();
  });

  // Explicit click — bypasses HTML5 "invalid hidden field" submit block
  $("#btn-start")?.addEventListener("click", async (e) => {
    e.preventDefault();
    await startTraining();
  });

  $("#btn-analyze").addEventListener("click", async () => {
    try {
      const body = formPayload();
      body.dry_run = true;
      const res = await api("/api/analyze", { method: "POST", body: JSON.stringify(body) });
      if (res.estimate) renderEstimate(res.estimate);
      appendLog(["Odhad hotov — nic se neučilo, jen počítání."]);
    } catch (err) {
      alert(err.message);
    }
  });

  $("#btn-cancel").addEventListener("click", async () => {
    try {
      await api("/api/cancel", { method: "POST", body: "{}" });
      appendLog(["Zastavení odesláno…"]);
    } catch (err) {
      alert(err.message);
    }
  });

  $("#btn-clear-log").addEventListener("click", () => {
    logEl.textContent = "";
  });

  $("#btn-formats").addEventListener("click", async () => {
    try {
      const data = await api("/api/formats");
      $("#formats-body").textContent = data.markdown || "";
      $("#modal").classList.remove("hidden");
    } catch (e) {
      alert(e.message);
    }
  });
  $("#modal-close").addEventListener("click", () => $("#modal").classList.add("hidden"));
  $("#modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") $("#modal").classList.add("hidden");
  });

  // Sync technical Ollama name from friendly identity name (unless user edits ollama manually)
  let ollamaTouched = false;
  $("#ollama_name")?.addEventListener("input", () => { ollamaTouched = true; });
  $("#identity_name")?.addEventListener("input", () => {
    if (!ollamaTouched) {
      $("#ollama_name").value = slugOllama($("#identity_name").value);
    }
  });

  async function boot() {
    fillModelPresets();
    applyModeDefaults();
    applyNoLimitsFields();
    if ($("#identity_name") && $("#ollama_name") && !ollamaTouched) {
      $("#ollama_name").value = slugOllama($("#identity_name").value);
    }
    await loadEnv();
    await refreshStatus();
    await pollLogs();
    startPolling();
  }

  fetch("/api/health").then((r) => r.json()).then((h) => {
    if (h.auth_required && !token) showAuth(true);
    else {
      showAuth(false);
      boot();
    }
  }).catch(() => {
    if (!authRequired) boot();
  });
})();

(() => {
  const authRequired = window.__AUTH_REQUIRED__ === true || window.__AUTH_REQUIRED__ === "true";
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));

  let token = localStorage.getItem("llm_ui_token") || "";
  let logSeq = 0;
  let pollTimer = null;
  let lineCount = 0;
  let ollamaTouched = false;
  let modelList = [];

  const logEl = $("#log");
  const form = $("#train-form");
  const authGate = $("#auth-gate");
  const app = $("#app");

  const PHASE_CS = {
    idle: "připraveno", setup: "příprava", analyze: "odhad", train: "učení",
    gguf: "GGUF", ollama: "Ollama", done: "hotovo", error: "chyba", cancelled: "zrušeno",
  };

  const MODE_DEFAULTS = {
    from_scratch: { method: "full", epochs: 2, learning_rate: 0.00005, batch_size: 1, grad_accum: 8, lora_r: 64 },
    finetune: { method: "qlora", epochs: 2, learning_rate: 0.0002, batch_size: 2, grad_accum: 4, lora_r: 32 },
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
      throw new Error("Neplatný UI token");
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || res.statusText));
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
    return form.querySelector('input[name="train_mode"]:checked')?.value || "from_scratch";
  }

  function slugOllama(n) {
    return String(n || "model").normalize("NFD").replace(/[\u0300-\u036f]/g, "")
      .toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
  }

  function applyModeDefaults() {
    const d = MODE_DEFAULTS[selectedMode()] || MODE_DEFAULTS.from_scratch;
    form.method.value = d.method;
    form.batch_size.value = d.batch_size;
    form.grad_accum.value = d.grad_accum;
    form.lora_r.value = d.lora_r;
    form.lora_alpha.value = String(d.lora_r * 2);
    if (!form.epochs.dataset.t) form.epochs.value = d.epochs;
    if (!form.learning_rate.dataset.t) form.learning_rate.value = d.learning_rate;
    $$(".choice-card").forEach((c) => {
      const i = c.querySelector("input");
      c.classList.toggle("selected", !!(i && i.checked));
    });
  }

  async function loadModels() {
    modelList = await api("/api/models");
    const sel = $("#model_preset");
    sel.innerHTML = modelList.map((m) => {
      const dl = m.downloaded ? " ✓staženo" : "";
      return `<option value="${m.id}">${m.label || m.id}${dl}</option>`;
    }).join("");
    if (modelList[0]) {
      sel.value = modelList[0].id;
      $("#model_id").value = modelList[0].id;
    }
  }

  function formPayload() {
    applyNoLimits();
    const fd = new FormData(form);
    const obj = {};
    for (const [k, v] of fd.entries()) {
      if (k === "train_mode" || k === "model_preset") continue;
      obj[k] = v;
    }
    for (const n of ["dry_run", "skip_gguf", "skip_ollama", "allow_over_limit", "skip_setup", "uncensored", "no_limits", "teach_identity"]) {
      obj[n] = !!form.querySelector(`[name="${n}"]`)?.checked;
    }
    obj.train_mode = selectedMode();
    obj.model_id = $("#model_preset").value || $("#model_id").value;
    $("#model_id").value = obj.model_id;
    const idn = (obj.identity_name || "Model").trim();
    obj.identity_name = idn;
    if (!ollamaTouched) obj.ollama_name = slugOllama(idn);
    const hft = ($("#hf_token").value || "").trim();
    if (hft) obj.hf_token = hft;
    for (const n of ["lora_r", "lora_alpha", "max_seq_length", "batch_size", "grad_accum", "epochs", "learning_rate", "max_train_hours", "max_cost_usd", "gpu_hourly_usd"]) {
      if (obj[n] !== undefined && obj[n] !== "") obj[n] = Number(obj[n]);
    }
    obj.framework = "peft";
    obj.system_prompt = `Jmenuješ se ${idn}. Jsi přímý asistent jménem ${idn}.`;
    return obj;
  }

  function applyNoLimits() {
    if ($("#no_limits")?.checked) {
      form.max_train_hours.value = 720;
      form.max_cost_usd.value = 999999;
    }
  }

  function appendLog(lines) {
    if (!lines?.length) return;
    const follow = $("#log-follow")?.checked !== false;
    const near = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 80;
    logEl.textContent += (logEl.textContent ? "\n" : "") + lines.join("\n");
    lineCount += lines.length;
    $("#log-count").textContent = lineCount + " řádků";
    if (follow || near) logEl.scrollTop = logEl.scrollHeight;
  }

  function setProgress(p, msg, phase, detail) {
    const pct = Math.max(0, Math.min(100, Number(p) || 0));
    $("#progress-fill").style.width = pct + "%";
    $("#progress-label").textContent = Math.round(pct) + "%";
    if (msg) $("#status-msg").textContent = msg;
    if (detail !== undefined) $("#progress-detail").textContent = detail || "";
    if (phase) {
      const b = $("#phase-badge");
      b.textContent = PHASE_CS[phase] || phase;
      b.className = "badge phase " + phase;
    }
  }

  function renderEstimate(est, cfg) {
    if (!est) return;
    $("#estimate-box").classList.remove("hidden");
    const name = cfg?.identity_name || "Model";
    const base = cfg?.model_id || cfg?.base_model || "";
    $("#estimate-plain").textContent = `„${name}“ · základ ${base} · ~${est.recommended_vram_gib.toFixed(1)} GB · ~${est.est_train_hours.toFixed(2)} h`;
    const rows = [
      ["Váš model", name],
      ["Základ (HF)", base],
      ["Ollama", cfg?.ollama_name || "—"],
      ["Metoda", cfg?.method || "—"],
      ["VRAM", est.recommended_vram_gib.toFixed(1) + " GB"],
      ["Samples", String(est.num_samples)],
      ["Steps", String(est.total_steps)],
      ["Čas", est.est_train_hours.toFixed(2) + " h"],
    ];
    $("#estimate-grid").innerHTML = rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("");
  }

  async function refreshStatus() {
    try {
      const st = await api("/api/status");
      let detail = st.progress_detail || "";
      if (st.train_progress) {
        const tp = st.train_progress;
        detail = `train ${tp.percent?.toFixed?.(1) ?? tp.percent}%` +
          (tp.step != null ? ` step ${tp.step}/${tp.total_steps || "?"}` : "");
      }
      setProgress(st.progress, st.message, st.phase, detail);
      if (st.estimate) renderEstimate(st.estimate, st.config || {});
      if (st.phase === "done" && st.ollama_name) {
        $("#result-box").classList.remove("hidden");
        $("#result-cmd").textContent = `ollama run ${st.ollama_name}`;
        $("#chat-model").value = st.ollama_name;
      }
      $("#btn-start").disabled = !!st.running;
      $("#btn-cancel").disabled = !st.running;
    } catch (_) {}
  }

  async function pollLogs() {
    try {
      const d = await api("/api/logs?after=" + logSeq);
      if (d.lines?.length) appendLog(d.lines);
      logSeq = d.seq || logSeq;
    } catch (_) {}
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      await refreshStatus();
      await pollLogs();
    }, 1000);
  }

  async function refreshHfStatus() {
    try {
      const s = await api("/api/hf/status");
      $("#hf-badge").textContent = s.has_token
        ? `HF: ${s.user || "token OK"}`
        : "HF: bez tokenu";
      $("#hf-status").textContent = s.has_token
        ? `Token uložen · uživatel: ${s.user || "?"}`
        : "Token není uložen — pro některé modely vložte hf_… a Uložit token";
      if (s.cli && !s.cli.huggingface_hub) {
        $("#hf-status").textContent += " · instaluji huggingface_hub…";
      }
    } catch (e) {
      $("#hf-badge").textContent = "HF: ?";
    }
  }

  // events
  $$('input[name="train_mode"]').forEach((r) => r.addEventListener("change", applyModeDefaults));
  $("#model_preset")?.addEventListener("change", () => {
    $("#model_id").value = $("#model_preset").value;
  });
  form.epochs?.addEventListener("input", () => { form.epochs.dataset.t = "1"; });
  form.learning_rate?.addEventListener("input", () => { form.learning_rate.dataset.t = "1"; });
  $("#ollama_name")?.addEventListener("input", () => { ollamaTouched = true; });
  $("#identity_name")?.addEventListener("input", () => {
    if (!ollamaTouched) $("#ollama_name").value = slugOllama($("#identity_name").value);
  });

  $("#token-save")?.addEventListener("click", async () => {
    token = $("#token-input").value.trim();
    try {
      await api("/api/status");
      localStorage.setItem("llm_ui_token", token);
      $("#token-error").classList.add("hidden");
      showAuth(false);
      boot();
    } catch (_) {
      $("#token-error").classList.remove("hidden");
    }
  });

  $("#btn-save-token")?.addEventListener("click", async () => {
    const t = $("#hf_token").value.trim();
    if (!t) return alert("Vložte token");
    try {
      const r = await api("/api/hf/token", { method: "POST", body: JSON.stringify({ token: t }) });
      appendLog([`HF token uložen · user ${r.user}`]);
      await refreshHfStatus();
      alert("Token uložen: " + r.user);
    } catch (e) {
      alert(e.message);
    }
  });

  $("#btn-ensure-ollama")?.addEventListener("click", async () => {
    try {
      $("#hf-status").textContent = "Ollama…";
      const r = await api("/api/ollama/ensure", { method: "POST", body: "{}" });
      appendLog([`Ollama: present=${r.present} installed=${r.installed}`]);
      if (r.error) appendLog(["Ollama error: " + r.error]);
      alert(r.present ? "Ollama běží" : "Ollama se nepodařilo nainstalovat: " + (r.error || "?"));
    } catch (e) {
      alert(e.message);
    }
  });

  $("#btn-download-model")?.addEventListener("click", async () => {
    const mid = $("#model_preset").value;
    const hft = $("#hf_token").value.trim();
    $("#dl-status").textContent = "Stahuji " + mid + "…";
    appendLog(["Stahuji model " + mid + " …"]);
    try {
      const r = await api("/api/models/download", {
        method: "POST",
        body: JSON.stringify({ model_id: mid, hf_token: hft || null }),
      });
      $("#dl-status").textContent = "Staženo: " + r.path;
      appendLog(["Model stažen: " + r.model_id + " → " + r.path]);
      await loadModels();
      $("#model_preset").value = r.model_id;
      $("#model_id").value = r.model_id;
    } catch (e) {
      $("#dl-status").textContent = "Chyba";
      appendLog(["Download error: " + e.message]);
      alert(e.message);
    }
  });

  $("#btn-chat")?.addEventListener("click", async () => {
    const model = $("#chat-model").value.trim();
    const message = $("#chat-input").value.trim();
    if (!message) return;
    $("#chat-out").textContent = "…";
    try {
      const r = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ model, message }),
      });
      $("#chat-out").textContent = r.reply || JSON.stringify(r);
    } catch (e) {
      $("#chat-out").textContent = "Chyba: " + e.message;
    }
  });

  async function startTraining() {
    try {
      const body = formPayload();
      setProgress(2, "Start…", "setup");
      appendLog(["=== START ===", `MODEL: ${body.model_id}`, `→ Ollama: ${body.ollama_name}`]);
      try {
        const pre = await api("/api/analyze", { method: "POST", body: JSON.stringify({ ...body, dry_run: true }) });
        if (pre.estimate) renderEstimate(pre.estimate, { ...body, ...(pre.config || {}) });
      } catch (e) {
        appendLog(["Odhad: " + e.message]);
      }
      const st = await api("/api/start", { method: "POST", body: JSON.stringify(body) });
      appendLog([`Job: ${st.phase} ${st.message}`]);
    } catch (e) {
      appendLog(["CHYBA: " + e.message]);
      alert(e.message);
    }
  }

  $("#btn-start")?.addEventListener("click", (e) => { e.preventDefault(); startTraining(); });
  $("#btn-analyze")?.addEventListener("click", async () => {
    try {
      const body = formPayload();
      body.dry_run = true;
      const res = await api("/api/analyze", { method: "POST", body: JSON.stringify(body) });
      if (res.estimate) renderEstimate(res.estimate, { ...body, ...(res.config || {}) });
    } catch (e) {
      alert(e.message);
    }
  });
  $("#btn-cancel")?.addEventListener("click", async () => {
    try {
      await api("/api/cancel", { method: "POST", body: "{}" });
      appendLog(["Cancel…"]);
    } catch (e) {
      alert(e.message);
    }
  });
  $("#btn-copy-log")?.addEventListener("click", async () => {
    try {
      const d = await api("/api/logs/full");
      await navigator.clipboard.writeText(d.text || "");
      $("#copy-toast").classList.remove("hidden");
      $("#copy-toast").textContent = `Zkopírováno ${d.lines} řádků`;
      setTimeout(() => $("#copy-toast").classList.add("hidden"), 2500);
    } catch (e) {
      alert(e.message);
    }
  });
  $("#btn-download-log")?.addEventListener("click", async () => {
    const res = await fetch("/api/logs/download", { headers: headers(false) });
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "logs.txt";
    a.click();
  });
  $("#btn-clear-log")?.addEventListener("click", () => {
    logEl.textContent = "";
    lineCount = 0;
  });

  async function boot() {
    applyModeDefaults();
    if (!ollamaTouched) $("#ollama_name").value = slugOllama($("#identity_name").value);
    try {
      const env = await api("/api/env");
      const g = env.gpus || [];
      $("#gpu-badge").textContent = g[0]
        ? `GPU: ${g[0].name} (${Math.round(g[0].memory_mib / 1024)} GB)`
        : "GPU: žádná";
    } catch (_) {}
    await refreshHfStatus();
    await loadModels();
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
  }).catch(() => boot());
})();

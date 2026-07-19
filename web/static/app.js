(() => {
  const authRequired = window.__AUTH_REQUIRED__ === true || window.__AUTH_REQUIRED__ === "true";
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));

  let token = localStorage.getItem("llm_ui_token") || "";
  let logSeq = 0;
  let pollTimer = null;
  let lineCount = 0;
  let ollamaTouched = false;

  const authGate = $("#auth-gate");
  const app = $("#app");
  const logEl = $("#log");
  const form = $("#train-form");

  // HF training bases. Ollama names (qwen3.5:2b) ≠ HF IDs — mapped below.
  // Prefer *-Base for "od nuly" full train; chat models also work.
  const PRESETS = {
    uncensored: [
      { id: "Qwen/Qwen3.5-0.8B-Base", label: "Qwen3.5 · 0.8B Base (≈ ollama qwen3.5:0.8b) · L4" },
      { id: "Qwen/Qwen3.5-2B-Base", label: "Qwen3.5 · 2B Base (≈ qwen3.5:2b) · L4 ★" },
      { id: "Qwen/Qwen3.5-4B-Base", label: "Qwen3.5 · 4B Base (≈ qwen3.5:4b) · L4" },
      { id: "Qwen/Qwen3.5-9B-Base", label: "Qwen3.5 · 9B Base (≈ qwen3.5:9b) · full těsné" },
      { id: "Qwen/Qwen3.5-0.8B", label: "Qwen3.5 · 0.8B (chat/post-train)" },
      { id: "Qwen/Qwen3.5-2B", label: "Qwen3.5 · 2B (chat/post-train)" },
      { id: "Qwen/Qwen3.5-4B", label: "Qwen3.5 · 4B (chat/post-train)" },
      { id: "Qwen/Qwen2.5-1.5B", label: "Qwen2.5 · 1.5B (starší)" },
      { id: "unsloth/llama-3.2-1b", label: "Llama 3.2 · 1B" },
      { id: "__custom__", label: "Vlastní (HF / ollama:qwen3.5:2b / cesta)…" },
    ],
    instruct: [
      { id: "Qwen/Qwen3.5-0.8B", label: "Qwen3.5 · 0.8B Instruct-like" },
      { id: "Qwen/Qwen3.5-2B", label: "Qwen3.5 · 2B Instruct-like" },
      { id: "Qwen/Qwen3.5-4B", label: "Qwen3.5 · 4B Instruct-like" },
      { id: "__custom__", label: "Vlastní…" },
    ],
  };

  const MODE_DEFAULTS = {
    from_scratch: {
      method: "full",
      framework: "peft",
      lora_r: 64,
      batch_size: 1,
      grad_accum: 8,
      epochs: 2,
      learning_rate: 0.00005,
    },
    finetune: {
      method: "qlora",
      framework: "peft",
      lora_r: 32,
      batch_size: 2,
      grad_accum: 4,
      epochs: 2,
      learning_rate: 0.0002,
    },
  };

  const PHASE_CS = {
    idle: "připraveno", setup: "příprava", analyze: "odhad", train: "učení",
    gguf: "GGUF", ollama: "Ollama", done: "hotovo", error: "chyba", cancelled: "zrušeno",
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
      showAuth(true, { keepAppVisible: true });
      throw new Error("Neplatný token — zadejte znovu dole");
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = data.detail;
      throw new Error(typeof d === "string" ? d : JSON.stringify(d || res.statusText));
    }
    return data;
  }

  function showAuth(need, { keepAppVisible = false } = {}) {
    if (need && authRequired) {
      authGate.classList.remove("hidden");
      if (keepAppVisible || !app.classList.contains("hidden")) {
        app.classList.remove("hidden");
        app.classList.add("app--auth-open");
      } else {
        app.classList.add("hidden");
      }
      requestAnimationFrame(() => {
        try { $("#token-input")?.focus({ preventScroll: true }); } catch (_) { $("#token-input")?.focus(); }
      });
    } else {
      authGate.classList.add("hidden");
      app.classList.remove("hidden");
      app.classList.remove("app--auth-open");
    }
  }

  function selectedMode() {
    return form.querySelector('input[name="train_mode"]:checked')?.value || "from_scratch";
  }

  async function fillPresets() {
    const list = ($("#uncensored")?.checked !== false ? PRESETS.uncensored : PRESETS.instruct).slice();
    // prepend local Ollama models
    try {
      const models = await api("/api/models");
      const ollama = (models || []).filter((m) => m.source === "ollama");
      ollama.reverse().forEach((m) => {
        list.unshift({ id: m.id, label: m.label || m.id });
      });
    } catch (_) { /* */ }
    const sel = $("#model_preset");
    const prev = sel.value;
    sel.innerHTML = list.map((p) => `<option value="${p.id}">${p.label || p.id}</option>`).join("");
    if (list.some((p) => p.id === prev)) sel.value = prev;
    else if (list[0]) sel.value = list[0].id;
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
    const d = MODE_DEFAULTS[selectedMode()] || MODE_DEFAULTS.from_scratch;
    form.method.value = d.method;
    if (form.framework) form.framework.value = d.framework || "peft";
    form.lora_r.value = d.lora_r;
    form.batch_size.value = d.batch_size;
    form.grad_accum.value = d.grad_accum;
    if (!form.epochs.dataset.touched) form.epochs.value = d.epochs;
    if (!form.learning_rate.dataset.touched) form.learning_rate.value = d.learning_rate;
    form.lora_alpha.value = String(Number(d.lora_r) * 2);
    $$(".choice-card").forEach((c) => {
      const i = c.querySelector("input");
      c.classList.toggle("selected", !!(i && i.checked));
    });
  }

  function slugOllama(name) {
    return String(name || "model")
      .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
      .toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
  }

  function formPayload() {
    syncModelId();
    applyNoLimits();
    const fd = new FormData(form);
    const obj = {};
    for (const [k, v] of fd.entries()) {
      if (["train_mode", "model_preset", "model_id_custom"].includes(k)) continue;
      obj[k] = v;
    }
    for (const name of [
      "dry_run", "skip_gguf", "skip_ollama", "allow_over_limit", "skip_setup",
      "uncensored", "no_limits", "teach_identity",
    ]) {
      obj[name] = !!form.querySelector(`[name="${name}"]`)?.checked;
    }
    obj.train_mode = selectedMode();
    obj.model_id = $("#model_id").value;
    // optional HF token from form
    const hft = ($("#hf_token")?.value || "").trim();
    if (hft) obj.hf_token = hft;
    const identity = (obj.identity_name || "Model").trim();
    obj.identity_name = identity;
    if (!ollamaTouched) obj.ollama_name = slugOllama(identity);
    for (const n of [
      "lora_r", "lora_alpha", "max_seq_length", "batch_size", "grad_accum",
      "epochs", "learning_rate", "max_train_hours", "max_cost_usd", "gpu_hourly_usd",
    ]) {
      if (obj[n] !== undefined && obj[n] !== "") obj[n] = Number(obj[n]);
    }
    if (obj.uncensored) {
      obj.system_prompt =
        `Jmenuješ se ${identity}. Jsi přímý asistent jménem ${identity} bez zbytečného moralizování. ` +
        `Na jméno odpovídej: ${identity}. Dodržuj zákony.`;
    } else {
      obj.system_prompt = `Jmenuješ se ${identity}. Jsi užitečný asistent.`;
    }
    return obj;
  }

  function applyNoLimits() {
    if ($("#no_limits")?.checked) {
      form.max_train_hours.value = 720;
      form.max_cost_usd.value = 999999;
      const a = form.querySelector('[name="allow_over_limit"]');
      if (a) a.checked = true;
    }
  }

  function followOn() {
    return $("#log-follow")?.checked !== false;
  }

  function appendLog(lines) {
    if (!lines?.length || !logEl) return;
    const near = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 90;
    logEl.textContent += (logEl.textContent ? "\n" : "") + lines.join("\n");
    lineCount += lines.length;
    const c = $("#log-count");
    if (c) c.textContent = lineCount + " řádků";
    if (followOn() || near) logEl.scrollTop = logEl.scrollHeight;
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
    const name = cfg?.identity_name || cfg?.display_name || cfg?.ollama_name || "Model";
    const base = cfg?.model_id || cfg?.base_model || "";
    $("#estimate-plain").textContent =
      `„${name}“` +
      (base ? ` (základ ${base})` : "") +
      ` · ~${est.recommended_vram_gib.toFixed(1)} GB VRAM · ` +
      `~${est.est_train_hours.toFixed(2)} h · ${est.num_samples} samples · ` +
      (est.fits_gpus ? "GPU OK" : "GPU těsně / ?");
    const rows = [
      ["Váš model", name],
      ["Základ (HF)", base || "—"],
      ["Ollama", cfg?.ollama_name || "—"],
      ["Metoda", cfg?.method || "—"],
      ["Parametry", (est.model_params_total / 1e9).toFixed(1) + " B"],
      ["Trainable", (est.trainable_params / 1e6).toFixed(1) + " M"],
      ["VRAM", est.recommended_vram_gib.toFixed(1) + " GB"],
      ["Samples", String(est.num_samples)],
      ["Steps", String(est.total_steps)],
      ["Čas", est.est_train_hours.toFixed(2) + " h"],
    ];
    $("#estimate-grid").innerHTML = rows
      .map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`)
      .join("");
  }

  function showResult(ollamaName) {
    if (!ollamaName) return;
    $("#result-box").classList.remove("hidden");
    $("#result-cmd").textContent = `ollama run ${ollamaName}`;
  }

  async function refreshStatus() {
    try {
      const st = await api("/api/status");
      const detail = st.train_progress
        ? `trénink ${st.train_progress.percent?.toFixed?.(1) ?? st.train_progress.percent}%` +
          (st.train_progress.step != null
            ? ` · step ${st.train_progress.step}/${st.train_progress.total_steps || "?"}`
            : "") +
          (st.train_progress.epoch != null ? ` · epoch ${st.train_progress.epoch}` : "")
        : (st.progress_detail || "");
      setProgress(st.progress, st.message, st.phase, detail);
      if (st.estimate) {
        renderEstimate(st.estimate, {
          ...(st.config || {}),
          identity_name: st.config?.identity_name || st.config?.display_name,
          ollama_name: st.ollama_name || st.config?.ollama_name,
        });
      }
      if (st.phase === "done" && st.ollama_name) showResult(st.ollama_name);
      $("#btn-start").disabled = !!st.running;
      $("#btn-cancel").disabled = !st.running;
      $("#btn-analyze").disabled = !!st.running;
    } catch (_) { /* */ }
  }

  async function pollLogs() {
    try {
      const data = await api("/api/logs?after=" + logSeq);
      if (data.lines?.length) appendLog(data.lines);
      logSeq = data.seq || logSeq;
    } catch (_) { /* */ }
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      await refreshStatus();
      await pollLogs();
    }, 1000);
  }

  async function copyAllLogs() {
    try {
      const data = await api("/api/logs/full");
      await navigator.clipboard.writeText(data.text || "");
      const t = $("#copy-toast");
      t.classList.remove("hidden");
      t.textContent = `Zkopírováno ${data.lines || "?"} řádků`;
      setTimeout(() => t.classList.add("hidden"), 3000);
    } catch (e) {
      try {
        await navigator.clipboard.writeText(logEl.textContent || "");
        alert("Zkopírován viditelný buffer");
      } catch (_) {
        alert(e.message);
      }
    }
  }

  async function downloadLogs() {
    try {
      const res = await fetch("/api/logs/download", { headers: headers(false) });
      if (!res.ok) throw new Error(res.statusText);
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "llm-training-logs.txt";
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      alert(e.message);
    }
  }

  async function startTraining() {
    try {
      const body = formPayload();
      setProgress(2, "Počítám odhad…", "analyze", "");
      appendLog(["=== START ===", `MODEL: ${body.model_id} · ${body.train_mode} · ${body.method}`]);
      try {
        const pre = await api("/api/analyze", { method: "POST", body: JSON.stringify({ ...body, dry_run: true }) });
        if (pre.estimate) {
          renderEstimate(pre.estimate, {
            model_id: body.model_id,
            method: body.method,
            identity_name: body.identity_name,
            ollama_name: body.ollama_name,
            display_name: body.identity_name,
            base_model: body.model_id,
          });
        }
      } catch (e) {
        appendLog(["Odhad: " + e.message]);
      }
      const st = await api("/api/start", { method: "POST", body: JSON.stringify(body) });
      appendLog([`Job: ${st.phase} — ${st.message}`]);
      await refreshStatus();
    } catch (e) {
      appendLog(["CHYBA: " + e.message]);
      alert(e.message);
    }
  }

  // events
  $$('input[name="train_mode"]').forEach((r) => r.addEventListener("change", applyModeDefaults));
  $("#uncensored")?.addEventListener("change", fillPresets);
  $("#model_preset")?.addEventListener("change", syncModelId);
  $("#model_id_custom")?.addEventListener("input", syncModelId);
  form.epochs?.addEventListener("input", () => { form.epochs.dataset.touched = "1"; });
  form.learning_rate?.addEventListener("input", () => { form.learning_rate.dataset.touched = "1"; });
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
      token = "";
    }
  });

  form.addEventListener("submit", (e) => { e.preventDefault(); startTraining(); });
  $("#btn-start")?.addEventListener("click", (e) => { e.preventDefault(); startTraining(); });
  $("#btn-analyze")?.addEventListener("click", async () => {
    try {
      const body = formPayload();
      body.dry_run = true;
      const res = await api("/api/analyze", { method: "POST", body: JSON.stringify(body) });
      if (res.estimate) renderEstimate(res.estimate, { model_id: body.model_id, method: body.method });
      appendLog([`Odhad pro ${body.model_id} hotov.`]);
    } catch (e) {
      alert(e.message);
    }
  });
  $("#btn-cancel")?.addEventListener("click", async () => {
    try {
      await api("/api/cancel", { method: "POST", body: "{}" });
      appendLog(["Zastavení odesláno…"]);
    } catch (e) {
      alert(e.message);
    }
  });
  $("#btn-copy-log")?.addEventListener("click", copyAllLogs);
  $("#btn-download-log")?.addEventListener("click", downloadLogs);
  $("#btn-clear-log")?.addEventListener("click", () => {
    logEl.textContent = "";
    lineCount = 0;
    $("#log-count").textContent = "0 řádků (historie na serveru zůstává)";
  });

  async function boot() {
    fillPresets();
    applyModeDefaults();
    if (!ollamaTouched) $("#ollama_name").value = slugOllama($("#identity_name").value);
    try {
      const env = await api("/api/env");
      const g = env.gpus || [];
      $("#gpu-badge").textContent = g.length
        ? `GPU: ${g[0].name} (${Math.round(g[0].memory_mib / 1024)} GB)`
        : "GPU: žádná";
    } catch (_) {
      $("#gpu-badge").textContent = "GPU: ?";
    }
    await refreshStatus();
    await pollLogs();
    startPolling();
  }

  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      if (h.auth_required && !token) showAuth(true);
      else {
        showAuth(false);
        boot();
      }
    })
    .catch(() => {
      if (!authRequired) boot();
    });
})();

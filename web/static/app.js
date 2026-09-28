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
    from_scratch: { method: "full", epochs: 1, learning_rate: 0.00005, batch_size: 1, grad_accum: 8, lora_r: 64, max_seq_length: 512 },
    // Hertz 0.1 STEM demo: Qwen3.5-9B + kuclab_hertz_0.1
    finetune: { method: "qlora", epochs: 1.0, learning_rate: 0.00008, batch_size: 1, grad_accum: 16, lora_r: 32, max_seq_length: 2048 },
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
    if (!res.ok) {
      let msg = res.statusText || String(res.status);
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail)) {
        msg = data.detail
          .map((d) => (d && d.msg ? `${(d.loc || []).join(".")}: ${d.msg}` : JSON.stringify(d)))
          .join("; ");
      } else if (data.detail != null) msg = JSON.stringify(data.detail);
      throw new Error(msg);
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
    return form.querySelector('input[name="train_mode"]:checked')?.value || "from_scratch";
  }

  function slugOllama(n) {
    return String(n || "model").normalize("NFD").replace(/[\u0300-\u036f]/g, "")
      .toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
  }

  function applyModeDefaults() {
    const d = MODE_DEFAULTS[selectedMode()] || MODE_DEFAULTS.finetune;
    form.method.value = d.method;
    form.batch_size.value = d.batch_size;
    form.grad_accum.value = d.grad_accum;
    form.lora_r.value = d.lora_r;
    form.lora_alpha.value = String(d.lora_r * 2);
    if (form.max_seq_length && d.max_seq_length) form.max_seq_length.value = d.max_seq_length;
    if (!form.epochs.dataset.t) form.epochs.value = d.epochs;
    if (!form.learning_rate.dataset.t) form.learning_rate.value = d.learning_rate;
    $$(".choice-card").forEach((c) => {
      const i = c.querySelector("input");
      c.classList.toggle("selected", !!(i && i.checked));
    });
  }

  // Model line name, e.g. "KucLab Hertz" + version "0.1" -> "KucLab Hertz 0.1"
  const NAME_PREFIX = "KucLab ";
  const DEFAULT_LINE = "KucLab Hertz";

  function buildIdentityName() {
    let base = ($("#identity_name")?.value || DEFAULT_LINE).trim() || DEFAULT_LINE;
    // Always keep the KucLab prefix, but the line name is free (Hertz, V, …)
    if (!base.startsWith(NAME_PREFIX)) {
      base = NAME_PREFIX + base.replace(/^KucLab\s*/i, "").trim();
    }
    base = base.replace(/\s+/g, " ").trim();
    const ver = ($("#identity_version")?.value || "").trim();
    if (!ver) return base;
    // Don't double-append if the version is already part of the name
    if (base.endsWith(ver)) return base;
    // Legacy "KucLab V" style glues the number on; named lines get a space
    return base.endsWith("V") ? base + ver : base + " " + ver;
  }

  function syncOllamaFromIdentity() {
    if (ollamaTouched) return;
    const idn = buildIdentityName();
    $("#ollama_name").value = slugOllama(idn);
  }

  async function loadModels() {
    modelList = await api("/api/models");
    const sel = $("#model_preset");
    sel.innerHTML = modelList.map((m) => {
      const dl = m.downloaded ? " ✓staženo" : "";
      return `<option value="${m.id}">${m.label || m.id}${dl}</option>`;
    }).join("");
    // Default Hertz 0.1: prefer Qwen3.5-9B, then first catalog entry
    const prefer = "Qwen/Qwen3.5-9B";
    const ids = modelList.map((m) => m.id);
    if (ids.includes(prefer)) {
      sel.value = prefer;
    } else if (modelList[0]) {
      sel.value = modelList[0].id;
    }
    // Prefer custom_model field if set (continue FT)
    const custom = ($("#custom_model")?.value || "").trim();
    $("#model_id").value = custom || sel.value || prefer;
  }

  function formPayload() {
    applyNoLimits();
    const fd = new FormData(form);
    const obj = {};
    for (const [k, v] of fd.entries()) {
      if (k === "train_mode" || k === "model_preset" || k === "identity_version" || k === "custom_model") continue;
      obj[k] = v;
    }
    for (const n of ["dry_run", "skip_gguf", "skip_ollama", "allow_over_limit", "skip_setup", "uncensored", "no_limits", "teach_identity"]) {
      obj[n] = !!form.querySelector(`[name="${n}"]`)?.checked;
    }
    obj.train_mode = selectedMode();
    // Custom path/id wins over preset forever
    const custom = ($("#custom_model")?.value || "").trim();
    obj.model_id = custom || $("#model_preset").value || $("#model_id").value;
    $("#model_id").value = obj.model_id;
    const idn = buildIdentityName();
    obj.identity_name = idn;
    // Keep only the line name in the field; the version lives in its own input
    const verNow = ($("#identity_version")?.value || "").trim();
    $("#identity_name").value =
      verNow && idn.endsWith(verNow) ? idn.slice(0, idn.length - verNow.length).trim() : idn;
    if (!ollamaTouched) {
      obj.ollama_name = slugOllama(idn);
      $("#ollama_name").value = obj.ollama_name;
    } else {
      obj.ollama_name = ($("#ollama_name").value || slugOllama(idn)).trim();
    }
    const founder = (obj.founder || $("#founder")?.value || "Jaroslav Kučera").trim();
    const trainedOn = (obj.trained_on || $("#trained_on")?.value || "2026-07-23").trim();
    obj.founder = founder;
    obj.trained_on = trainedOn;
    const hft = ($("#hf_token").value || "").trim();
    if (hft) obj.hf_token = hft;
    for (const n of ["lora_r", "lora_alpha", "max_seq_length", "batch_size", "grad_accum", "epochs", "learning_rate", "max_train_hours", "max_cost_usd", "gpu_hourly_usd"]) {
      if (obj[n] !== undefined && obj[n] !== "") obj[n] = Number(obj[n]);
    }
    obj.framework = "peft";
    obj.system_prompt = (
      `Jsi ${idn} — model od KucLab zaměřený na fyziku, chemii, biologii a matematiku. ` +
      `Zakladatel KucLab je ${founder}. Poslední dotrénování: ${trainedOn}. ` +
      `Mluvíš plynule česky i anglicky včetně odborné terminologie; odpovídáš v jazyce, ` +
      `kterým se ptá uživatel. Používáš správné české odborné termíny — ` +
      `„směrodatná odchylka“, „výkon“, „stejnosměrný proud“, „endotermická reakce“ — ` +
      `nikdy si je nevymýšlíš ani nekomolíš. ` +
      `U výpočtů a odvození postupuješ krok za krokem a ukazuješ mezivýsledky, ` +
      `se správnými jednotkami a značením veličin. ` +
      `Když si nejsi jistý, řekneš to rovnou místo vymýšlení. K aktuálním datům ` +
      `(počasí, kurzy, zprávy) nemáš přístup — v takovém případě to řekneš. ` +
      `Jsi přímý a technický, bez zbytečné omáčky a korporátních frází. ` +
      `Jméno/identitu jen když se zeptají. Nezačínej „Jsem…“ zbytečně.`
    );
    if (obj.identity_repeat === undefined || obj.identity_repeat === "") {
      obj.identity_repeat = 1;
    } else {
      obj.identity_repeat = Math.min(2, Number(obj.identity_repeat) || 1);
    }
    if (obj.max_seq_length !== undefined && obj.max_seq_length !== "") {
      obj.max_seq_length = Number(obj.max_seq_length) || 2048;
    }
    if (obj.num_ctx !== undefined && obj.num_ctx !== "") {
      obj.num_ctx = Number(obj.num_ctx) || 8192;
    } else {
      obj.num_ctx = 8192;
    }
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
    const method = (cfg?.method || est?.method || "—").toString();
    const yours = method === "full" ? "full = všechny váhy vaše" : "QLoRA = váš model (adaptér+merge → Ollama)";
    $("#estimate-plain").textContent =
      `„${name}“ · ${yours} · ~${est.recommended_vram_gib.toFixed(1)} GB · ~${est.est_train_hours.toFixed(2)} h` +
      (est.ollama_num_ctx ? ` · ctx train ${est.train_context || "?"} / Ollama ${est.ollama_num_ctx}` : "");
    const rows = [
      ["Váš model", name],
      ["Základ (start)", base],
      ["Ollama tag", cfg?.ollama_name || "—"],
      ["Metoda", method + (method === "qlora" || method === "lora" ? " → pořád VÁŠ model" : " → VÁŠ model")],
      ["Train context", String(est.train_context || cfg?.max_seq_length || "—")],
      ["Nativní max ctx", String(est.native_context || "—")],
      ["Ollama num_ctx", String(est.ollama_num_ctx || "—")],
      ["VRAM", est.recommended_vram_gib.toFixed(1) + " GB"],
      ["Samples", String(est.num_samples)],
      ["Steps", String(est.total_steps)],
      ["Čas (odhad)", est.est_train_hours.toFixed(2) + " h"],
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
    // Selecting a preset clears custom so catalog wins until user types custom again
    if ($("#custom_model") && !($("#custom_model").value || "").trim()) {
      $("#model_id").value = $("#model_preset").value;
    }
  });
  $("#custom_model")?.addEventListener("input", () => {
    const c = ($("#custom_model").value || "").trim();
    if (c) $("#model_id").value = c;
  });
  form.epochs?.addEventListener("input", () => { form.epochs.dataset.t = "1"; });
  form.learning_rate?.addEventListener("input", () => { form.learning_rate.dataset.t = "1"; });
  $("#ollama_name")?.addEventListener("input", () => { ollamaTouched = true; });
  $("#identity_name")?.addEventListener("input", () => {
    // Enforce only the KucLab prefix — the line name itself is the user's choice
    const v = $("#identity_name").value || "";
    if (!v.startsWith(NAME_PREFIX) && !NAME_PREFIX.startsWith(v)) {
      $("#identity_name").value = NAME_PREFIX + v.replace(/^KucLab\s*/i, "");
    }
    syncOllamaFromIdentity();
  });
  $("#identity_version")?.addEventListener("input", () => {
    syncOllamaFromIdentity();
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
  async function copyTextToClipboard(text) {
    const t = text || "";
    // 1) Modern API (HTTPS or localhost)
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      try {
        await navigator.clipboard.writeText(t);
        return true;
      } catch (_) { /* fall through */ }
    }
    // 2) execCommand fallback (works on HTTP / some browsers)
    try {
      const ta = document.createElement("textarea");
      ta.value = t;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.top = "0";
      ta.style.left = "0";
      ta.style.width = "1px";
      ta.style.height = "1px";
      ta.style.padding = "0";
      ta.style.border = "none";
      ta.style.outline = "none";
      ta.style.boxShadow = "none";
      ta.style.background = "transparent";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      ta.setSelectionRange(0, ta.value.length);
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      if (ok) return true;
    } catch (_) { /* fall through */ }
    // 3) Last resort: select the terminal itself
    try {
      const range = document.createRange();
      range.selectNodeContents(logEl);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      const ok = document.execCommand("copy");
      sel.removeAllRanges();
      if (ok) return true;
    } catch (_) {}
    return false;
  }

  function showCopyToast(msg, isError = false) {
    const el = $("#copy-toast");
    if (!el) return;
    el.classList.remove("hidden");
    el.textContent = msg;
    el.style.color = isError ? "var(--danger, #f66)" : "var(--good, #3c3)";
    setTimeout(() => el.classList.add("hidden"), 3500);
  }

  $("#btn-copy-log")?.addEventListener("click", async (ev) => {
    ev.preventDefault();
    const btn = $("#btn-copy-log");
    if (btn) btn.disabled = true;
    try {
      // Prefer live terminal text (what user sees); fallback to API full logs
      let text = (logEl && logEl.textContent) ? logEl.textContent : "";
      let lines = text ? text.split("\n").length : 0;
      if (!text.trim()) {
        try {
          const d = await api("/api/logs/full");
          text = d.text || "";
          lines = d.lines || (text ? text.split("\n").length : 0);
        } catch (e) {
          showCopyToast("Nelze načíst logy: " + (e.message || e), true);
          return;
        }
      }
      if (!text.trim()) {
        showCopyToast("Terminál je prázdný", true);
        return;
      }
      const ok = await copyTextToClipboard(text);
      if (ok) {
        showCopyToast(`Zkopírováno ${lines} řádků`);
      } else {
        // Open a selectable dialog so user can Ctrl+C manually
        const w = window.open("", "_blank", "width=800,height=600");
        if (w) {
          w.document.write("<pre style='white-space:pre-wrap;font:12px monospace'></pre>");
          w.document.querySelector("pre").textContent = text;
          w.document.title = "Logy — Ctrl+A, Ctrl+C";
          showCopyToast("Otevřeno v novém okně — Ctrl+A, Ctrl+C", true);
        } else {
          showCopyToast("Schránka blokována prohlížečem (zkus HTTPS nebo Ctrl+A v terminálu)", true);
        }
      }
    } catch (e) {
      showCopyToast("Copy chyba: " + (e.message || e), true);
    } finally {
      if (btn) btn.disabled = false;
    }
  });
  $("#btn-download-log")?.addEventListener("click", async () => {
    try {
      let text = (logEl && logEl.textContent) ? logEl.textContent : "";
      if (!text.trim()) {
        const res = await fetch("/api/logs/download", { headers: headers(false) });
        if (!res.ok) throw new Error("download " + res.status);
        const blob = await res.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "llm-training-logs.txt";
        a.click();
        URL.revokeObjectURL(a.href);
        showCopyToast("Staženo llm-training-logs.txt");
        return;
      }
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "llm-training-logs.txt";
      a.click();
      URL.revokeObjectURL(a.href);
      showCopyToast("Staženo llm-training-logs.txt");
    } catch (e) {
      showCopyToast("Download chyba: " + (e.message || e), true);
    }
  });
  $("#btn-clear-log")?.addEventListener("click", () => {
    logEl.textContent = "";
    lineCount = 0;
    if ($("#log-count")) $("#log-count").textContent = "0 řádků";
  });

  async function boot() {
    applyModeDefaults();
    // Use the full name (line + version), not just the line field — otherwise
    // "KucLab Hertz" + "0.1" would slug to "kuclab-hertz" and drop the version.
    if (!ollamaTouched) syncOllamaFromIdentity();
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

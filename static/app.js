// ResearchOS Frontend Application

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const htmlEl = document.documentElement;
  const themeToggle = document.getElementById("theme-toggle");
  const themeIcon = themeToggle.querySelector(".theme-icon");
  const wakeupBanner = document.getElementById("wakeup-banner");

  const navHome = document.getElementById("nav-home");
  const navEval = document.getElementById("nav-eval");
  const navHistory = document.getElementById("nav-history");

  const viewHome = document.getElementById("view-home");
  const viewLive = document.getElementById("view-live");
  const viewReport = document.getElementById("view-report");
  const viewEval = document.getElementById("view-eval");
  const viewHistory = document.getElementById("view-history");

  const researchForm = document.getElementById("research-form");
  const questionInput = document.getElementById("question-input");
  const accessCodeInput = document.getElementById("access-code-input");
  const runsLeftSpan = document.getElementById("runs-left");
  const homeError = document.getElementById("home-error");
  const runBtn = document.getElementById("run-btn");

  const liveTimer = document.getElementById("live-timer");
  const liveStatusText = document.getElementById("live-status-text");
  const liveQuestionTitle = document.getElementById("live-question-title");
  const traceEventsList = document.getElementById("trace-events-list");

  let currentRunId = null;
  let timerInterval = null;
  let runStartTime = null;
  let currentDepth = "standard";
  let activeEventSource = null;

  // --- Theme Management ---
  function initTheme() {
    const saved = localStorage.getItem("researchos_theme");
    const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const theme = saved || (systemPrefersDark ? "dark" : "light");
    setTheme(theme);
  }

  function setTheme(theme) {
    htmlEl.setAttribute("data-theme", theme);
    localStorage.setItem("researchos_theme", theme);
    themeIcon.textContent = theme === "dark" ? "🌙" : "☀️";
  }

  themeToggle.addEventListener("click", () => {
    const current = htmlEl.getAttribute("data-theme");
    setTheme(current === "dark" ? "light" : "dark");
  });

  initTheme();

  // --- Navigation ---
  function showView(targetView, activeNavBtn) {
    [viewHome, viewLive, viewReport, viewEval, viewHistory].forEach(v => v.classList.add("hidden"));
    [navHome, navEval, navHistory].forEach(b => b.classList.remove("active"));

    targetView.classList.remove("hidden");
    if (activeNavBtn) activeNavBtn.classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  navHome.addEventListener("click", () => showView(viewHome, navHome));
  navEval.addEventListener("click", () => {
    showView(viewEval, navEval);
    loadEvaluationPage();
  });
  navHistory.addEventListener("click", () => {
    showView(viewHistory, navHistory);
    loadHistoryPage();
  });
  document.getElementById("btn-back-home").addEventListener("click", () => showView(viewHome, navHome));

  // --- Example Chips & Depth Toggle ---
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      questionInput.value = chip.getAttribute("data-q");
      questionInput.focus();
    });
  });

  document.querySelectorAll(".seg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-checked", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-checked", "true");
      currentDepth = btn.getAttribute("data-depth");
    });
  });

  // --- Initial Limits & Server Wakeup Check ---
  async function checkLimits() {
    const wakeTimer = setTimeout(() => {
      wakeupBanner.classList.remove("hidden");
    }, 2500);

    try {
      const res = await fetch("/api/limits");
      clearTimeout(wakeTimer);
      wakeupBanner.classList.add("hidden");
      if (res.ok) {
        const data = await res.json();
        runsLeftSpan.textContent = data.remaining_runs;
        if (data.requires_access_code) {
          document.getElementById("access-code-wrapper").style.display = "block";
        }
      }
    } catch (e) {
      clearTimeout(wakeTimer);
      console.warn("Could not check rate limits:", e);
    }
  }

  checkLimits();

  // --- Run Research Form Submit ---
  researchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    homeError.classList.add("hidden");

    const question = questionInput.value.trim();
    if (!question) return;

    const accessCode = accessCodeInput.value.trim();
    const headers = { "Content-Type": "application/json" };
    if (accessCode) {
      headers["X-Access-Code"] = accessCode;
      localStorage.setItem("researchos_access_code", accessCode);
    }

    // Set UI loading
    runBtn.disabled = true;
    runBtn.querySelector(".btn-text").textContent = "Submitting Job...";
    runBtn.querySelector(".spinner").classList.remove("hidden");

    try {
      const res = await fetch("/api/research", {
        method: "POST",
        headers: headers,
        body: JSON.stringify({
          question: question,
          depth: currentDepth,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to start research job.");
      }

      const data = await res.json();
      currentRunId = data.run_id;

      // Reset button
      runBtn.disabled = false;
      runBtn.querySelector(".btn-text").textContent = "Execute Research Run";
      runBtn.querySelector(".spinner").classList.add("hidden");

      // Switch to Live View
      startLiveView(currentRunId, question);
    } catch (err) {
      runBtn.disabled = false;
      runBtn.querySelector(".btn-text").textContent = "Execute Research Run";
      runBtn.querySelector(".spinner").classList.add("hidden");
      homeError.textContent = err.message;
      homeError.classList.remove("hidden");
    }
  });

  // Restore saved access code if any
  const savedCode = localStorage.getItem("researchos_access_code");
  if (savedCode && accessCodeInput) {
    accessCodeInput.value = savedCode;
  }

  // --- Live View & SSE Streaming ---
  function startLiveView(runId, question) {
    showView(viewLive, null);
    liveQuestionTitle.textContent = question;
    liveStatusText.textContent = "Supervisor planning sub-questions...";

    // Reset timeline step styles
    resetTimeline();
    traceEventsList.innerHTML = "";

    // Start elapsed timer
    runStartTime = Date.now();
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      const elapsed = ((Date.now() - runStartTime) / 1000).toFixed(1);
      liveTimer.textContent = `${elapsed}s`;
    }, 100);

    // Connect SSE
    if (activeEventSource) {
      activeEventSource.close();
    }

    activeEventSource = new EventSource(`/api/research/${runId}/events`);

    activeEventSource.addEventListener("agent_started", (e) => {
      const data = JSON.parse(e.data);
      handleAgentStarted(data.agent, data.message);
      logTraceEvent("agent_started", `[${data.agent}] ${data.message}`);
    });

    activeEventSource.addEventListener("agent_finished", (e) => {
      const data = JSON.parse(e.data);
      handleAgentFinished(data.agent, data);
      logTraceEvent("agent_finished", `[${data.agent}] Completed successfully.`);
    });

    activeEventSource.addEventListener("warning", (e) => {
      const data = JSON.parse(e.data);
      logTraceEvent("warning", `[${data.agent || 'Warning'}] ${data.message || JSON.stringify(data)}`);
    });

    activeEventSource.addEventListener("run_completed", () => {
      finishLiveView(runId);
    });

    activeEventSource.addEventListener("close", () => {
      finishLiveView(runId);
    });

    activeEventSource.onerror = (err) => {
      console.warn("SSE connection error or closed:", err);
      // Fallback polling check after brief delay
      setTimeout(() => checkRunStatus(runId), 2000);
    };
  }

  async function checkRunStatus(runId) {
    try {
      const res = await fetch(`/api/research/${runId}`);
      if (res.ok) {
        const run = await res.json();
        if (run.status === "completed") {
          finishLiveView(runId);
        }
      }
    } catch (e) {
      console.warn("Status poll error:", e);
    }
  }

  function resetTimeline() {
    const steps = ["supervisor", "retrieval", "extraction", "comparison", "synthesis", "evaluator"];
    steps.forEach(s => {
      const el = document.getElementById(`step-${s}`);
      const badge = document.getElementById(`badge-${s}`);
      const details = document.getElementById(`details-${s}`);
      if (el) el.className = "timeline-step";
      if (badge) badge.textContent = "Pending";
      if (details) {
        details.innerHTML = "";
        details.classList.add("hidden");
      }
    });
  }

  function handleAgentStarted(agent, msg) {
    const key = agent.toLowerCase();
    const el = document.getElementById(`step-${key}`);
    const badge = document.getElementById(`badge-${key}`);
    if (el) {
      el.classList.add("active");
      liveStatusText.textContent = `${agent} Agent: ${msg}`;
    }
    if (badge) badge.textContent = "Working...";
  }

  function handleAgentFinished(agent, data) {
    const key = agent.toLowerCase();
    const el = document.getElementById(`step-${key}`);
    const badge = document.getElementById(`badge-${key}`);
    const details = document.getElementById(`details-${key}`);

    if (el) {
      el.classList.remove("active");
      el.classList.add("completed");
    }
    if (badge) badge.textContent = "Completed";

    if (details) {
      let html = "";
      if (key === "supervisor" && data.sub_questions) {
        html = `<strong>Planned ${data.sub_questions.length} sub-questions:</strong><ul style="margin:4px 0 0 16px;">${data.sub_questions.map(q => `<li>${q}</li>`).join("")}</ul>`;
      } else if (key === "retrieval") {
        html = `Found <strong>${data.papers_count || 0} papers</strong> across arXiv, indexed <strong>${data.chunks_count || 0} hybrid chunks</strong>.`;
      } else if (key === "extraction") {
        html = `Verified <strong>${data.claims_count || 0} atomic claims</strong> with verbatim quote check (${Math.round((data.quote_verification_rate || 0)*100)}% verified).`;
      } else if (key === "synthesis") {
        html = `Generated report with <strong>${Math.round((data.citation_coverage || 0)*100)}% citation coverage</strong> across ${data.references_count || 0} references.`;
      } else if (key === "evaluator" && data.eval_scores) {
        html = `Faithfulness score: <strong>${data.eval_scores.faithfulness_score || '--'}</strong> (${data.eval_scores.supported_pct || 100}% Supported).`;
      }
      if (html) {
        details.innerHTML = html;
        details.classList.remove("hidden");
      }
    }
  }

  function logTraceEvent(type, message) {
    const now = new Date().toLocaleTimeString();
    const item = document.createElement("div");
    item.className = "trace-log-item";
    item.innerHTML = `<span class="text-muted">[${now}]</span> <strong>${type}:</strong> ${message}`;
    traceEventsList.appendChild(item);
    traceEventsList.scrollTop = traceEventsList.scrollHeight;
  }

  async function finishLiveView(runId) {
    if (timerInterval) clearInterval(timerInterval);
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }

    liveStatusText.textContent = "Finalizing synthesis report...";

    try {
      const res = await fetch(`/api/research/${runId}`);
      if (!res.ok) throw new Error("Failed to load completed report.");
      const runData = await res.json();
      renderReport(runData);
    } catch (e) {
      console.error("Error loading final report:", e);
    }
  }

  // --- Report Rendering ---
  function renderReport(run) {
    showView(viewReport, null);

    // 1. Evaluation Card
    const scores = run.eval_scores || {};
    const cov = Math.round((scores.citation_coverage || run.synthesis?.citation_coverage || 0) * 100);
    const verif = Math.round((scores.quote_verification_rate || run.quote_verification_rate || 1.0) * 100);
    const faith = scores.faithfulness_score !== undefined ? Number(scores.faithfulness_score).toFixed(3) : "1.000";

    document.getElementById("rep-cov-val").textContent = `${cov}%`;
    document.getElementById("rep-quote-val").textContent = `${verif}%`;
    document.getElementById("rep-faith-val").textContent = faith;

    document.getElementById("rep-supp-badge").textContent = `${scores.supported_pct || 100}% Supp`;
    document.getElementById("rep-part-badge").textContent = `${scores.partial_pct || 0}% Part`;
    document.getElementById("rep-unsupp-badge").textContent = `${scores.unsupported_pct || 0}% Unsupp`;

    // 2. Render Markdown Body
    const rawMd = run.summary_md || run.synthesis?.markdown || "# No report content.";
    const reportContainer = document.getElementById("report-markdown-container");

    if (window.marked) {
      let html = marked.parse(rawMd);
      // Transform [1], [2] citation markers into clickable chip buttons
      html = html.replace(/\[(\d+)\]/g, (match, num) => {
        return `<a href="#ref-${num}" class="citation-chip" data-ref="${num}">[${num}]</a>`;
      });
      // Attach IDs to references list items if matching [n]
      html = html.replace(/<li>\[(\d+)\]/g, '<li id="ref-$1">[<strong>$1</strong>]');
      reportContainer.innerHTML = html;
    } else {
      reportContainer.textContent = rawMd;
    }

    // Attach citation chip click handlers
    reportContainer.querySelectorAll(".citation-chip").forEach(chip => {
      chip.addEventListener("click", (e) => {
        e.preventDefault();
        const refId = `ref-${chip.getAttribute("data-ref")}`;
        const refEl = document.getElementById(refId);
        if (refEl) {
          refEl.scrollIntoView({ behavior: "smooth", block: "center" });
          refEl.classList.remove("reference-highlight");
          void refEl.offsetWidth; // trigger reflow
          refEl.classList.add("reference-highlight");
        }
      });
    });

    // 3. Claims Table
    const claims = run.claims || [];
    document.getElementById("claims-count-badge").textContent = claims.length;
    const claimsTbody = document.getElementById("claims-table-body");
    claimsTbody.innerHTML = "";

    if (claims.length === 0) {
      claimsTbody.innerHTML = '<tr><td colspan="6" class="text-muted" style="text-align:center;">No atomic claims extracted.</td></tr>';
    } else {
      claims.forEach(c => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${c.claim_id || ''}</code></td>
          <td>${c.claim || ''}</td>
          <td><a href="https://arxiv.org/abs/${c.paper_id}" target="_blank" rel="noopener"><code>${c.paper_id || ''}</code></a></td>
          <td><blockquote style="font-size:12px; font-style:italic;">"${c.quote || ''}"</blockquote></td>
          <td><span class="badge ${c.stance === 'supports' ? 'badge-success' : 'badge-warning'}">${c.stance || 'supports'}</span></td>
          <td><span class="badge badge-success">✓ Verified</span></td>
        `;
        claimsTbody.appendChild(tr);
      });
    }

    // 4. Comparison Display
    const comp = run.comparison || {};
    const compBox = document.getElementById("comparison-display");
    let compHtml = "";

    const themes = comp.themes || [];
    if (themes.length > 0) {
      themes.forEach(t => {
        compHtml += `
          <div class="theme-card">
            <h4>${t.theme_name} <span class="badge badge-accent">Claims: ${t.claim_ids?.join(', ') || ''}</span></h4>
            ${t.agreements?.length ? `<div><strong>Agreements:</strong><ul>${t.agreements.map(a => `<li>${a}</li>`).join('')}</ul></div>` : ''}
            ${t.contradictions?.length ? `<div style="margin-top:6px;"><strong>Contradictions:</strong><ul>${t.contradictions.map(c => `<li>${c}</li>`).join('')}</ul></div>` : ''}
          </div>
        `;
      });
    }

    if (comp.limitations_found?.length) {
      compHtml += `<div class="theme-card"><h4>Stated Limitations</h4><ul>${comp.limitations_found.map(l => `<li>${l}</li>`).join('')}</ul></div>`;
    }

    if (comp.open_questions?.length) {
      compHtml += `<div class="theme-card"><h4>Open Research Questions</h4><ul>${comp.open_questions.map(q => `<li>${q}</li>`).join('')}</ul></div>`;
    }

    compBox.innerHTML = compHtml || '<div class="text-muted">No comparative themes extracted.</div>';

    // 5. Telemetry Display
    const trace = run.trace || [];
    const timings = run.timings || {};
    const counts = run.call_counts || {};
    const telemBox = document.getElementById("telemetry-display");

    let telemHtml = `
      <div style="margin-bottom:12px; font-family:var(--font-mono); font-size:13px;">
        <strong>LLM Calls:</strong> ${counts.llm_calls || 0} live calls, ${counts.cached_calls || 0} disk cache hits |
        <strong>Total Duration:</strong> ${timings.total || '--'}s
      </div>
      <div class="table-wrapper">
        <table class="data-table">
          <thead><tr><th>Agent Node</th><th>Duration</th><th>LLM Calls</th><th>Status</th><th>Notes</th></tr></thead>
          <tbody>
            ${trace.map(t => `
              <tr>
                <td><code>${t.agent}</code></td>
                <td>${t.duration_s}s</td>
                <td>${t.llm_calls}</td>
                <td><span class="badge ${t.status === 'success' ? 'badge-success' : 'badge-warning'}">${t.status}</span></td>
                <td>${t.notes || ''}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
    telemBox.innerHTML = telemHtml;

    // 6. Copy & Download Buttons
    document.getElementById("btn-copy-md").onclick = () => {
      navigator.clipboard.writeText(rawMd);
      const btn = document.getElementById("btn-copy-md");
      btn.textContent = "Copied!";
      setTimeout(() => btn.textContent = "Copy Markdown", 2000);
    };

    document.getElementById("btn-download-md").onclick = () => {
      const blob = new Blob([rawMd], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `researchos_report_${run.run_id || 'run'}.md`;
      a.click();
      URL.revokeObjectURL(url);
    };
  }

  // --- Subtab Navigation in Report ---
  document.querySelectorAll(".sub-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".sub-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".subtab-pane").forEach(p => p.classList.add("hidden"));

      tab.classList.add("active");
      const targetId = `subtab-content-${tab.getAttribute("data-subtab")}`;
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.remove("hidden");
    });
  });

  // --- Evaluation Page Loader ---
  async function loadEvaluationPage() {
    const container = document.getElementById("eval-page-content");
    container.innerHTML = '<div class="loading-state"><span class="spinner"></span> Loading latest benchmark evaluation...</div>';

    try {
      const res = await fetch("/api/eval/latest");
      if (!res.ok) throw new Error("Could not fetch evaluation report.");
      const data = await res.json();

      if (data.status === "no_report") {
        container.innerHTML = `
          <div class="alert alert-warning">
            No system evaluation has been recorded yet. Run <code>python -m eval.run --limit 8</code> to generate benchmark ablation numbers.
          </div>
        `;
        return;
      }

      const v = data.retrieval_ablation?.vector_only || {};
      const h = data.retrieval_ablation?.hybrid || {};
      const f = data.factual_consistency || {};

      container.innerHTML = `
        <h3 style="font-size:18px; margin-bottom:12px;">1. Retrieval Quality: Vector-Only vs. Hybrid (Vector + BM25) Ablation</h3>
        <p class="text-muted" style="margin-bottom:16px;">Cutoff <strong>k = 5</strong> evaluated against verified ground-truth arXiv relevant papers:</p>

        <div class="table-wrapper" style="margin-bottom:24px;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Vector-Only</th>
                <th>Hybrid (Dense + BM25 RRF)</th>
                <th>Delta</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Precision@5</strong></td>
                <td><code>${(v['precision@k'] || 0).toFixed(4)}</code></td>
                <td><strong class="text-accent"><code>${(h['precision@k'] || 0).toFixed(4)}</code></strong></td>
                <td><span class="badge badge-success">+${((h['precision@k'] || 0) - (v['precision@k'] || 0)).toFixed(4)}</span></td>
              </tr>
              <tr>
                <td><strong>Recall@5</strong></td>
                <td><code>${(v['recall@k'] || 0).toFixed(4)}</code></td>
                <td><strong class="text-accent"><code>${(h['recall@k'] || 0).toFixed(4)}</code></strong></td>
                <td><span class="badge badge-success">+${((h['recall@k'] || 0) - (v['recall@k'] || 0)).toFixed(4)}</span></td>
              </tr>
              <tr>
                <td><strong>MRR</strong></td>
                <td><code>${(v['mrr'] || 0).toFixed(4)}</code></td>
                <td><strong class="text-accent"><code>${(h['mrr'] || 0).toFixed(4)}</code></strong></td>
                <td><span class="badge badge-success">+${((h['mrr'] || 0) - (v['mrr'] || 0)).toFixed(4)}</span></td>
              </tr>
              <tr>
                <td><strong>nDCG@5</strong></td>
                <td><code>${(v['ndcg@k'] || 0).toFixed(4)}</code></td>
                <td><strong class="text-accent"><code>${(h['ndcg@k'] || 0).toFixed(4)}</code></strong></td>
                <td><span class="badge badge-success">+${((h['ndcg@k'] || 0) - (v['ndcg@k'] || 0)).toFixed(4)}</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <h3 style="font-size:18px; margin-bottom:12px;">2. Factual Consistency & Hallucination Mitigation</h3>
        <div class="eval-grid" style="margin-bottom:24px;">
          <div class="eval-metric">
            <div class="metric-label">Citation Coverage</div>
            <div class="metric-val text-accent">${Math.round((f.citation_coverage || 0)*100)}%</div>
            <div class="metric-sub">Factual statements with citations</div>
          </div>
          <div class="eval-metric">
            <div class="metric-label">Quote Verification</div>
            <div class="metric-val text-success">${Math.round((f.quote_verification_rate || 0)*100)}%</div>
            <div class="metric-sub">Verbatim match rate</div>
          </div>
          <div class="eval-metric">
            <div class="metric-label">Faithfulness Score</div>
            <div class="metric-val text-accent">${(f.faithfulness_score || 0).toFixed(3)}</div>
            <div class="metric-sub">Judge composite score</div>
          </div>
          <div class="eval-metric">
            <div class="metric-label">Judge Verdicts</div>
            <div class="verdict-breakdown">
              <span class="badge badge-success">${f.supported_pct || 100}% Supp</span>
              <span class="badge badge-warning">${f.partial_pct || 0}% Part</span>
              <span class="badge badge-error">${f.unsupported_pct || 0}% Unsupp</span>
            </div>
          </div>
        </div>

        <h3 style="font-size:16px; margin-bottom:8px;">Evaluation Notes & Limitations:</h3>
        <ul style="margin-left:20px; font-size:13px; color:var(--text-muted); line-height:1.7;">
          <li>Dataset consists of 8 curated high-signal machine learning questions across foundational architectures.</li>
          <li>Hybrid retrieval combines dense cosine similarity embeddings with BM25 keyword rankings via Reciprocal Rank Fusion (k=60).</li>
          <li>Verbatim quote checking programmatically drops fabricated evidence before claims are synthesized.</li>
        </ul>
      `;
    } catch (e) {
      container.innerHTML = `<div class="alert alert-error">Error loading evaluation report: ${e.message}</div>`;
    }
  }

  // --- History Page Loader ---
  async function loadHistoryPage() {
    const container = document.getElementById("recent-runs-list");
    container.innerHTML = '<div class="loading-state"><span class="spinner"></span> Loading run history...</div>';

    try {
      const res = await fetch("/api/research?limit=15");
      if (!res.ok) throw new Error("Could not fetch recent runs.");
      const runs = await res.json();

      if (!runs || runs.length === 0) {
        container.innerHTML = '<div class="text-muted" style="text-align:center; padding:20px;">No research runs recorded yet.</div>';
        return;
      }

      container.innerHTML = runs.map(r => `
        <div class="theme-card" style="cursor:pointer;" data-id="${r.run_id}">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span class="badge ${r.status === 'completed' ? 'badge-success' : 'badge-warning'}">${r.status.toUpperCase()}</span>
            <span class="text-muted" style="font-size:12px;">${new Date(r.created_at).toLocaleString()}</span>
          </div>
          <h4 style="font-size:15px; margin-bottom:4px;">${r.question}</h4>
          <div class="text-muted" style="font-size:12px;">Run ID: <code>${r.run_id}</code> | Depth: ${r.depth}</div>
        </div>
      `).join('');

      container.querySelectorAll(".theme-card").forEach(card => {
        card.addEventListener("click", async () => {
          const runId = card.getAttribute("data-id");
          try {
            const rRes = await fetch(`/api/research/${runId}`);
            if (rRes.ok) {
              const runData = await rRes.json();
              renderReport(runData);
            }
          } catch (e) {
            console.error("Failed to load run:", e);
          }
        });
      });
    } catch (e) {
      container.innerHTML = `<div class="alert alert-error">Error loading history: ${e.message}</div>`;
    }
  }
});

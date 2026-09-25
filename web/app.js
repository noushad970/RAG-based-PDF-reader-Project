/**
 * Local PDF RAG Studio - Frontend Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const uploadProgress = document.getElementById("uploadProgress");
  const progressBarFill = document.getElementById("progressBarFill");
  const uploadStatusText = document.getElementById("uploadStatusText");
  const docList = document.getElementById("docList");
  const docCount = document.getElementById("docCount");
  const reindexBtn = document.getElementById("reindexBtn");

  const ollamaStatus = document.getElementById("ollamaStatus");
  const embedModelTag = document.getElementById("embedModelTag");
  const llmModelTag = document.getElementById("llmModelTag");
  const vectorCount = document.getElementById("vectorCount");

  const topKSlider = document.getElementById("topKSlider");
  const topKValue = document.getElementById("topKValue");

  const chatMessages = document.getElementById("chatMessages");
  const welcomeCard = document.getElementById("welcomeCard");
  const chatForm = document.getElementById("chatForm");
  const questionInput = document.getElementById("questionInput");
  const sendBtn = document.getElementById("sendBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");

  let isGenerating = false;

  // Initialize
  init();

  function init() {
    loadStatus();
    loadDocuments();
    setupEventListeners();
  }

  function setupEventListeners() {
    // Top-K slider
    topKSlider.addEventListener("input", (e) => {
      topKValue.textContent = e.target.value;
    });

    // File Upload Drag & Drop
    browseBtn.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });

    // Re-index button
    reindexBtn.addEventListener("click", triggerReindex);

    // Clear chat
    clearChatBtn.addEventListener("click", () => {
      chatMessages.innerHTML = "";
      if (welcomeCard) {
        chatMessages.appendChild(welcomeCard);
      }
    });

    // Suggestion chips
    document.addEventListener("click", (e) => {
      if (e.target.classList.contains("suggestion-chip")) {
        const query = e.target.getAttribute("data-query");
        if (query) {
          questionInput.value = query;
          handleSendQuestion();
        }
      }
    });

    // Auto-resize textarea
    questionInput.addEventListener("input", () => {
      questionInput.style.height = "auto";
      questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + "px";
    });

    // Chat submit
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleSendQuestion();
    });

    questionInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendQuestion();
      }
    });
  }

  // ============================================================================
  // Status & Documents API
  // ============================================================================
  async function loadStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();

      if (data.ollama_running) {
        ollamaStatus.textContent = "Online";
        ollamaStatus.className = "status-badge";
      } else {
        ollamaStatus.textContent = "Offline";
        ollamaStatus.className = "status-badge error";
      }

      embedModelTag.textContent = data.embedding_model;
      llmModelTag.textContent = data.llm_model;
      vectorCount.textContent = `${data.total_vectors} vectors`;
    } catch (err) {
      ollamaStatus.textContent = "Disconnected";
      ollamaStatus.className = "status-badge error";
    }
  }

  async function loadDocuments() {
    try {
      const res = await fetch("/api/documents");
      const data = await res.json();
      const docs = data.documents || [];

      docCount.textContent = docs.length;

      if (docs.length === 0) {
        docList.innerHTML = '<div class="empty-docs">No documents uploaded yet.</div>';
        return;
      }

      docList.innerHTML = docs.map(doc => {
        const ext = doc.extension.replace(".", "");
        const badgeClass = `badge-${ext}` in ["badge-pdf", "badge-txt", "badge-md"] ? `badge-${ext}` : `badge-pdf`;
        return `
          <div class="doc-item">
            <div class="doc-info">
              <span class="doc-badge badge-${ext}">${ext}</span>
              <div>
                <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
                <span class="doc-size">${doc.size_kb} KB</span>
              </div>
            </div>
            <button class="btn-icon btn-icon-danger delete-doc-btn" data-file="${doc.filename}" title="Delete document">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
        `;
      }).join("");

      // Attach delete handlers
      document.querySelectorAll(".delete-doc-btn").forEach(btn => {
        btn.addEventListener("click", async (e) => {
          e.stopPropagation();
          const filename = btn.getAttribute("data-file");
          if (confirm(`Delete '${filename}' and re-index?`)) {
            await deleteDocument(filename);
          }
        });
      });
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  }

  async function handleFileUpload(file) {
    uploadProgress.classList.remove("hidden");
    progressBarFill.style.width = "40%";
    uploadStatusText.textContent = `Uploading '${file.name}' & creating chunks...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
      progressBarFill.style.width = "75%";
      uploadStatusText.textContent = `Embedding text with nomic-embed-text & updating FAISS...`;

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      progressBarFill.style.width = "100%";
      uploadStatusText.textContent = `Done! Indexed ${data.stats.chunks_count} chunks.`;

      setTimeout(() => {
        uploadProgress.classList.add("hidden");
        progressBarFill.style.width = "0%";
      }, 2000);

      loadDocuments();
      loadStatus();
    } catch (err) {
      alert(`Upload error: ${err.message}`);
      uploadProgress.classList.add("hidden");
      progressBarFill.style.width = "0%";
    }
  }

  async function deleteDocument(filename) {
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
        method: "DELETE"
      });
      if (res.ok) {
        loadDocuments();
        loadStatus();
      }
    } catch (err) {
      alert(`Delete error: ${err.message}`);
    }
  }

  async function triggerReindex() {
    reindexBtn.style.transform = "rotate(180deg)";
    try {
      const res = await fetch("/api/reindex", { method: "POST" });
      const data = await res.json();
      alert(`Re-indexed ${data.stats.documents_count} documents (${data.stats.chunks_count} chunks).`);
      loadStatus();
    } catch (err) {
      alert(`Re-indexing failed: ${err.message}`);
    } finally {
      setTimeout(() => { reindexBtn.style.transform = "none"; }, 400);
    }
  }

  // ============================================================================
  // Chat & Query Execution
  // ============================================================================
  async function handleSendQuestion() {
    const question = questionInput.value.trim();
    if (!question || isGenerating) return;

    if (welcomeCard && welcomeCard.parentNode) {
      welcomeCard.remove();
    }

    // Append user message
    appendMessage("user", question);

    questionInput.value = "";
    questionInput.style.height = "auto";
    isGenerating = true;
    sendBtn.disabled = true;

    // Append loading typing bubble
    const typingId = appendTypingIndicator();
    scrollToBottom();

    const topK = parseInt(topKSlider.value) || 3;

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK })
      });

      removeTypingIndicator(typingId);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }

      const data = await res.json();
      appendAssistantResponse(data);
    } catch (err) {
      removeTypingIndicator(typingId);
      appendMessage("assistant", `⚠️ Error: ${err.message}`);
    } finally {
      isGenerating = false;
      sendBtn.disabled = false;
      scrollToBottom();
      questionInput.focus();
    }
  }

  function appendMessage(role, text) {
    const msg = document.createElement("div");
    msg.className = `message ${role}`;

    const avatarSvg = role === "user" 
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>`;

    msg.innerHTML = `
      <div class="msg-avatar">${avatarSvg}</div>
      <div class="msg-content">
        <div class="msg-bubble">${formatMarkdown(text)}</div>
      </div>
    `;

    chatMessages.appendChild(msg);
    scrollToBottom();
  }

  function appendAssistantResponse(data) {
    const msg = document.createElement("div");
    msg.className = "message assistant";

    // Format Sources HTML
    let sourcesHtml = "";
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
      const items = data.retrieved_chunks.map((chunk, idx) => {
        const pageText = chunk.page ? `Page ${chunk.page}` : `Chunk ${chunk.chunk_id}`;
        const simScore = chunk.similarity !== undefined ? (chunk.similarity * 100).toFixed(1) + "%" : "";
        return `
          <div class="source-item">
            <div class="source-meta">
              <strong>${idx + 1}.</strong>
              <span>${chunk.source} (${pageText})</span>
            </div>
            ${simScore ? `<span class="similarity-badge">Match: ${simScore}</span>` : ""}
          </div>
        `;
      }).join("");

      sourcesHtml = `
        <div class="sources-card">
          <div class="sources-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
            Sources & Grounded Citations
          </div>
          <div class="sources-list">${items}</div>
        </div>
      `;
    }

    // Format Debug Telemetry HTML
    const debugJson = JSON.stringify({
      debug_info: data.debug,
      retrieved_context_preview: data.context,
      prompt_sent_to_ollama: data.prompt
    }, null, 2);

    const debugHtml = `
      <details class="debug-details">
        <summary>🔍 Inspect RAG Pipeline & Prompt Telemetry (${data.duration_ms}ms)</summary>
        <pre class="debug-box">${escapeHtml(debugJson)}</pre>
      </details>
    `;

    msg.innerHTML = `
      <div class="msg-avatar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg>
      </div>
      <div class="msg-content">
        <div class="msg-bubble">${formatMarkdown(data.answer)}</div>
        ${sourcesHtml}
        ${debugHtml}
      </div>
    `;

    chatMessages.appendChild(msg);
    scrollToBottom();
  }

  function appendTypingIndicator() {
    const id = "typing-" + Date.now();
    const typing = document.createElement("div");
    typing.id = id;
    typing.className = "message assistant";
    typing.innerHTML = `
      <div class="msg-avatar">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path></svg>
      </div>
      <div class="msg-content">
        <div class="typing-bubble">
          <div class="dot"></div>
          <div class="dot"></div>
          <div class="dot"></div>
        </div>
      </div>
    `;
    chatMessages.appendChild(typing);
    return id;
  }

  function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let formatted = escapeHtml(text);

    // Code blocks ```code```
    formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre class="debug-box"><code>$1</code></pre>');
    // Inline code `code`
    formatted = formatted.replace(/`([^`]+)`/g, '<code class="mono-tag">$1</code>');
    // Bold **text**
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Italic *text*
    formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // Line breaks
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});

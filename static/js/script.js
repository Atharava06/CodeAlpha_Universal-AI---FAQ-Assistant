/**
 * Universal AI & FAQ Assistant - Frontend Application Logic
 * Integrates Natural Language Processing (TF-IDF Cosine Similarity)
 * with Google Gemini AI for questions across all domains.
 */

(function () {
  'use strict';

  // --- DOM Elements ---
  const chatMessages = document.getElementById('chat-messages');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const clearChatBtn = document.getElementById('clear-chat-btn');
  const typingIndicator = document.getElementById('typing-indicator');
  const charCounter = document.getElementById('char-counter');
  const welcomeCard = document.getElementById('welcome-card');
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const connectionStatus = document.getElementById('connection-status');
  const toast = document.getElementById('toast');
  const toastMessage = document.getElementById('toast-message');

  // Google API Key & Badge Elements
  const googleKeyBadge = document.getElementById('google-key-badge');
  const googleStatusText = document.getElementById('google-status-text');
  const googleApiKeyInput = document.getElementById('google-api-key-input');
  const toggleKeyVisibility = document.getElementById('toggle-key-visibility');
  const verifyKeyBtn = document.getElementById('verify-key-btn');
  const clearKeyBtn = document.getElementById('clear-key-btn');
  const keyValidationStatus = document.getElementById('key-validation-status');

  // FAQ Modal Elements
  const browseFaqsBtn = document.getElementById('browse-faqs-btn');
  const faqModal = document.getElementById('faq-modal');
  const closeFaqModal = document.getElementById('close-faq-modal');
  const faqSearchInput = document.getElementById('faq-search-input');
  const faqCategoryFilter = document.getElementById('faq-category-filter');
  const faqListContainer = document.getElementById('faq-list-container');

  // Settings Modal Elements
  const settingsBtn = document.getElementById('settings-btn');
  const settingsModal = document.getElementById('settings-modal');
  const closeSettingsModal = document.getElementById('close-settings-modal');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdValDisplay = document.getElementById('threshold-val-display');
  const saveSettingsBtn = document.getElementById('save-settings-btn');

  // --- Storage Keys ---
  const STORAGE_KEY = 'universal_faq_chat_history_v2';
  const THEME_STORAGE_KEY = 'faq_app_theme';
  const GOOGLE_KEY_STORAGE = 'google_gemini_api_key';
  const AI_MODE_STORAGE = 'ai_operation_mode';

  // --- State ---
  let isSubmitting = false;
  let customThreshold = parseFloat(localStorage.getItem('faq_threshold') || '0.25');
  let googleApiKey = localStorage.getItem(GOOGLE_KEY_STORAGE) || '';
  let aiMode = localStorage.getItem(AI_MODE_STORAGE) || 'hybrid';
  let serverGeminiActive = false;
  let loadedFaqs = [];

  // --- Formatting Helpers ---
  function formatTimestamp(date = new Date()) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function escapeHTML(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Basic Markdown Formatter for Gemini AI code and lists
  function formatMarkdown(text) {
    if (!text) return '';
    let html = escapeHTML(text);

    // Code blocks with syntax highlighting container
    html = html.replace(/```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code `code`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold **text** or __text__
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Unordered lists (- or *)
    html = html.replace(/(?:^|\n)[-*]\s+([^\n]+)/g, '<br>&bull; $1');

    // Newlines to break (outside pre tags)
    html = html.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');

    return html;
  }

  function showToast(msg) {
    if (!toast) return;
    toastMessage.textContent = msg;
    toast.classList.remove('hidden');
    setTimeout(() => {
      toast.classList.add('hidden');
    }, 2800);
  }

  // --- Theme Controller ---
  function initTheme() {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY) || 'dark';
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
  }

  function updateThemeIcon(theme) {
    if (!themeToggleBtn) return;
    const icon = themeToggleBtn.querySelector('i');
    if (icon) {
      icon.className = theme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
    }
  }

  themeToggleBtn.addEventListener('click', () => {
    const currentTheme = document.body.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.body.setAttribute('data-theme', nextTheme);
    localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    updateThemeIcon(nextTheme);
    showToast(`Switched to ${nextTheme} theme`);
  });

  // --- Google API Key & Status Badge ---
  function updateGoogleKeyStatusBadge() {
    const hasKey = Boolean((googleApiKey && googleApiKey.trim().length > 10) || serverGeminiActive);
    if (hasKey) {
      googleKeyBadge.classList.add('active');
      googleStatusText.textContent = 'Google AI Active';
      googleKeyBadge.style.borderColor = '#10b981';
      keyValidationStatus.className = 'key-status-badge active';
      keyValidationStatus.textContent = 'Active (Ready for all domains)';
    } else {
      googleKeyBadge.classList.remove('active');
      googleStatusText.textContent = 'Add Google Key';
      googleKeyBadge.style.borderColor = 'rgba(96, 165, 250, 0.3)';
      keyValidationStatus.className = 'key-status-badge';
      keyValidationStatus.textContent = 'Not configured';
    }
  }

  if (googleKeyBadge) {
    googleKeyBadge.addEventListener('click', () => {
      settingsModal.classList.remove('hidden');
      googleApiKeyInput.focus();
    });
  }

  // --- Auto-resize Textarea & Char Counter ---
  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
    const length = userInput.value.length;
    charCounter.textContent = `${length}/1000`;
  });

  // Enter to send (Shift+Enter for newline)
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isSubmitting) {
        handleSendMessage();
      }
    }
  });

  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!isSubmitting) {
      handleSendMessage();
    }
  });

  // --- Message UI Rendering ---
  function appendUserMessage(text, time = formatTimestamp()) {
    const row = document.createElement('div');
    row.className = 'message-row user-row';
    row.innerHTML = `
      <div class="message-bubble">
        <div class="message-text">${escapeHTML(text)}</div>
        <div class="message-meta">${time}</div>
      </div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function appendBotMessage(data, time = formatTimestamp()) {
    const row = document.createElement('div');
    const isMatched = data.matched === true;
    const source = data.source || (isMatched ? 'Verified FAQ' : 'Assistant');
    row.className = `message-row bot-row ${isMatched ? '' : 'fallback-msg'}`;

    // Source Tag
    let sourceBadge = '';
    if (source.includes('Gemini')) {
      sourceBadge = `<span class="category-tag" style="background: rgba(66, 133, 244, 0.15); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3)"><i class="fa-brands fa-google"></i> Google Gemini AI</span>`;
    } else if (source.includes('FAQ')) {
      sourceBadge = `<span class="category-tag"><i class="fa-solid fa-building-columns"></i> Institutional FAQ</span>`;
    }

    const categoryBadge = data.category && data.category !== 'Universal AI'
      ? `<span class="category-tag"><i class="fa-solid fa-folder"></i> ${escapeHTML(data.category)}</span>`
      : '';

    const confidenceBadge = (isMatched && data.confidence !== undefined)
      ? `<span class="confidence-tag high" title="Confidence: ${data.confidence}"><i class="fa-solid fa-circle-check"></i> Answered</span>`
      : `<span class="confidence-tag low"><i class="fa-solid fa-circle-question"></i> No Direct Match</span>`;

    const matchedQuestionBanner = data.matched_question
      ? `<div class="matched-question-banner"><i class="fa-solid fa-quote-left"></i> <span>Matched FAQ: <strong>${escapeHTML(data.matched_question)}</strong></span></div>`
      : '';

    // Prompt to add key if query is unhandled and no key is set
    const keyPrompt = data.needs_google_key
      ? `<div style="margin-top: 10px;">
           <button class="btn primary-btn open-key-settings-btn" style="font-size: 0.8rem; padding: 6px 12px;">
             <i class="fa-solid fa-key"></i> Enter Google API Key in Settings
           </button>
         </div>`
      : '';

    row.innerHTML = `
      <div class="bot-avatar-sm" aria-hidden="true">
        <i class="${source.includes('Gemini') ? 'fa-solid fa-wand-magic-sparkles' : 'fa-solid fa-robot'}"></i>
      </div>
      <div class="message-bubble">
        <div class="bot-header-meta">
          <div class="meta-tags">
            ${sourceBadge}
            ${categoryBadge}
            ${confidenceBadge}
          </div>
        </div>
        ${matchedQuestionBanner}
        <div class="bot-answer-text">${formatMarkdown(data.answer)}</div>
        ${keyPrompt}
        <div class="bot-footer-meta">
          <span>${time}</span>
          <div class="bubble-actions">
            <button class="bubble-action-btn copy-btn" title="Copy answer" aria-label="Copy answer">
              <i class="fa-regular fa-copy"></i>
            </button>
            <button class="bubble-action-btn like-btn" title="Helpful" aria-label="Helpful response">
              <i class="fa-regular fa-thumbs-up"></i>
            </button>
          </div>
        </div>
      </div>
    `;

    // Connect open settings button if present
    const keyBtn = row.querySelector('.open-key-settings-btn');
    if (keyBtn) {
      keyBtn.addEventListener('click', () => {
        settingsModal.classList.remove('hidden');
        googleApiKeyInput.focus();
      });
    }

    // Copy answer event
    const copyBtn = row.querySelector('.copy-btn');
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(data.answer).then(() => {
        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
        showToast('Answer copied to clipboard!');
        setTimeout(() => {
          copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>';
        }, 2000);
      });
    });

    // Thumbs up feedback event
    const likeBtn = row.querySelector('.like-btn');
    likeBtn.addEventListener('click', () => {
      likeBtn.classList.toggle('active');
      likeBtn.innerHTML = likeBtn.classList.contains('active')
        ? '<i class="fa-solid fa-thumbs-up"></i>'
        : '<i class="fa-regular fa-thumbs-up"></i>';
      showToast('Thank you for your feedback!');
    });

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  // --- Session Storage Management ---
  function saveSessionHistory(type, data, time) {
    try {
      const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      history.push({ type, data, time });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
      console.warn('Unable to persist session history:', e);
    }
  }

  function loadSessionHistory() {
    try {
      const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      if (history.length > 0) {
        history.forEach(item => {
          if (item.type === 'user') {
            appendUserMessage(item.data, item.time);
          } else if (item.type === 'bot') {
            appendBotMessage(item.data, item.time);
          }
        });
      }
    } catch (e) {
      console.error('Failed to load session history:', e);
    }
  }

  function clearSession() {
    localStorage.removeItem(STORAGE_KEY);
    const rows = chatMessages.querySelectorAll('.message-row');
    rows.forEach(r => r.remove());
    if (welcomeCard) {
      welcomeCard.style.display = 'block';
    }
    showToast('Conversation cleared');
    userInput.focus();
  }

  clearChatBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear this conversation?')) {
      clearSession();
    }
  });

  // --- Send Message Flow ---
  async function handleSendMessage(prefilledQuery = null) {
    const message = (prefilledQuery || userInput.value).trim();
    if (!message || isSubmitting) return;

    // Reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    charCounter.textContent = '0/1000';

    // Show message immediately
    const userTimestamp = formatTimestamp();
    appendUserMessage(message, userTimestamp);
    saveSessionHistory('user', message, userTimestamp);

    // Disable input while processing
    isSubmitting = true;
    sendBtn.disabled = true;
    typingIndicator.classList.remove('hidden');
    scrollToBottom();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: message,
          threshold: customThreshold,
          api_key: googleApiKey,
          mode: aiMode
        })
      });

      const data = await response.json();
      typingIndicator.classList.add('hidden');

      if (!response.ok) {
        const errorMsg = data.error || 'Server error processing question.';
        const botData = {
          answer: `⚠️ Error: ${errorMsg}`,
          matched_question: null,
          category: 'Error',
          confidence: 0,
          matched: false,
          source: 'System'
        };
        const botTimestamp = formatTimestamp();
        appendBotMessage(botData, botTimestamp);
        saveSessionHistory('bot', botData, botTimestamp);
      } else {
        const botTimestamp = formatTimestamp();
        appendBotMessage(data, botTimestamp);
        saveSessionHistory('bot', data, botTimestamp);
      }
    } catch (err) {
      console.error('Chat error:', err);
      typingIndicator.classList.add('hidden');
      const networkErrorData = {
        answer: "I'm having trouble connecting to the server. Please verify that the Flask backend is running on http://127.0.0.1:5000.",
        matched_question: null,
        category: 'Network Error',
        confidence: 0,
        matched: false,
        source: 'System'
      };
      const botTimestamp = formatTimestamp();
      appendBotMessage(networkErrorData, botTimestamp);
      saveSessionHistory('bot', networkErrorData, botTimestamp);
    } finally {
      isSubmitting = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  }

  // --- Suggested Question Chips ---
  function initSuggestedChips() {
    const chips = document.querySelectorAll('.chip');
    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        const question = chip.getAttribute('data-question') || chip.innerText.trim();
        handleSendMessage(question);
      });
    });
  }

  // --- Backend Health Check & FAQ Loading ---
  async function checkBackendHealth() {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        serverGeminiActive = Boolean(data.gemini_configured);
        updateGoogleKeyStatusBadge();

        const statusMsg = serverGeminiActive || googleApiKey
          ? `<i class="fa-solid fa-circle-check"></i> Universal AI + ${data.faqs_loaded} FAQs Ready`
          : `<i class="fa-solid fa-circle-check"></i> ${data.faqs_loaded} FAQs Ready (Add Key for All Domains)`;

        connectionStatus.innerHTML = statusMsg;
        connectionStatus.style.color = 'var(--success)';
      } else {
        connectionStatus.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Backend Degraded';
        connectionStatus.style.color = 'var(--warning)';
      }
    } catch (e) {
      connectionStatus.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Server Offline';
      connectionStatus.style.color = 'var(--danger)';
    }
  }

  // --- FAQ Knowledge Base Explorer Drawer ---
  async function loadFaqExplorer() {
    try {
      const res = await fetch('/api/faqs');
      if (!res.ok) throw new Error('Failed to load FAQs');
      const data = await res.json();
      loadedFaqs = data.faqs || [];

      faqCategoryFilter.innerHTML = '<option value="all">All Categories</option>';
      if (data.categories) {
        data.categories.forEach(cat => {
          const opt = document.createElement('option');
          opt.value = cat;
          opt.textContent = cat;
          faqCategoryFilter.appendChild(opt);
        });
      }

      renderFaqList(loadedFaqs);
    } catch (err) {
      faqListContainer.innerHTML = `
        <div class="loading-spinner" style="color: var(--danger)">
          <i class="fa-solid fa-circle-exclamation"></i> Error loading FAQs.
        </div>
      `;
    }
  }

  function renderFaqList(faqs) {
    if (!faqs || faqs.length === 0) {
      faqListContainer.innerHTML = '<p style="text-align:center; padding: 20px; color: var(--text-dim)">No questions match your filter.</p>';
      return;
    }

    faqListContainer.innerHTML = '';
    faqs.forEach(faq => {
      const item = document.createElement('div');
      item.className = 'faq-card-item';
      item.innerHTML = `
        <div class="faq-card-top">
          <span class="category-tag"><i class="fa-solid fa-folder"></i> ${escapeHTML(faq.category)}</span>
          <button class="ask-faq-btn" data-question="${escapeHTML(faq.question)}">
            <i class="fa-regular fa-paper-plane"></i> Ask This
          </button>
        </div>
        <div class="faq-card-question">${escapeHTML(faq.question)}</div>
      `;

      item.querySelector('.ask-faq-btn').addEventListener('click', (e) => {
        const q = e.currentTarget.getAttribute('data-question');
        faqModal.classList.add('hidden');
        handleSendMessage(q);
      });

      faqListContainer.appendChild(item);
    });
  }

  function filterFaqs() {
    const query = (faqSearchInput.value || '').toLowerCase().trim();
    const category = faqCategoryFilter.value;

    const filtered = loadedFaqs.filter(item => {
      const matchesCategory = category === 'all' || item.category === category;
      const matchesSearch = !query || item.question.toLowerCase().includes(query) || (item.category && item.category.toLowerCase().includes(query));
      return matchesCategory && matchesSearch;
    });

    renderFaqList(filtered);
  }

  faqSearchInput.addEventListener('input', filterFaqs);
  faqCategoryFilter.addEventListener('change', filterFaqs);

  browseFaqsBtn.addEventListener('click', () => {
    faqModal.classList.remove('hidden');
    if (loadedFaqs.length === 0) {
      loadFaqExplorer();
    }
    faqSearchInput.focus();
  });

  closeFaqModal.addEventListener('click', () => {
    faqModal.classList.add('hidden');
  });

  faqModal.addEventListener('click', (e) => {
    if (e.target === faqModal) faqModal.classList.add('hidden');
  });

  // --- Settings & Google API Key Logic ---
  function initSettings() {
    thresholdSlider.value = customThreshold;
    thresholdValDisplay.textContent = customThreshold.toFixed(2);
    googleApiKeyInput.value = googleApiKey;

    // Set active radio button for AI mode
    const modeRadio = document.querySelector(`input[name="ai-mode"][value="${aiMode}"]`);
    if (modeRadio) modeRadio.checked = true;

    // Toggle key visibility
    toggleKeyVisibility.addEventListener('click', () => {
      const isPass = googleApiKeyInput.type === 'password';
      googleApiKeyInput.type = isPass ? 'text' : 'password';
      toggleKeyVisibility.innerHTML = isPass ? '<i class="fa-regular fa-eye-slash"></i>' : '<i class="fa-regular fa-eye"></i>';
    });

    // Verify and Save Google API Key
    verifyKeyBtn.addEventListener('click', async () => {
      const keyVal = googleApiKeyInput.value.trim();
      if (!keyVal) {
        showToast('Please enter a Google API Key first');
        return;
      }

      verifyKeyBtn.disabled = true;
      verifyKeyBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Testing...';

      try {
        const res = await fetch('/api/verify-key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ api_key: keyVal })
        });
        const data = await res.json();

        if (res.ok && data.valid) {
          googleApiKey = keyVal;
          localStorage.setItem(GOOGLE_KEY_STORAGE, googleApiKey);
          updateGoogleKeyStatusBadge();
          showToast('✓ Key verified! Questions across every domain are now enabled.');
        } else {
          keyValidationStatus.className = 'key-status-badge error';
          keyValidationStatus.textContent = 'Invalid Key';
          showToast(`Validation Failed: ${data.message || 'Key invalid'}`);
        }
      } catch (err) {
        showToast('Connection error while validating key.');
      } finally {
        verifyKeyBtn.disabled = false;
        verifyKeyBtn.innerHTML = '<i class="fa-solid fa-circle-check"></i> Validate & Save';
      }
    });

    // Clear Google API Key
    clearKeyBtn.addEventListener('click', () => {
      googleApiKey = '';
      googleApiKeyInput.value = '';
      localStorage.removeItem(GOOGLE_KEY_STORAGE);
      updateGoogleKeyStatusBadge();
      showToast('Google API Key removed.');
    });

    // Threshold Slider
    thresholdSlider.addEventListener('input', () => {
      thresholdValDisplay.textContent = parseFloat(thresholdSlider.value).toFixed(2);
    });

    // Save All Settings
    saveSettingsBtn.addEventListener('click', () => {
      customThreshold = parseFloat(thresholdSlider.value);
      localStorage.setItem('faq_threshold', customThreshold);

      // Save mode
      const selectedMode = document.querySelector('input[name="ai-mode"]:checked');
      if (selectedMode) {
        aiMode = selectedMode.value;
        localStorage.setItem(AI_MODE_STORAGE, aiMode);
      }

      // If user typed a key but didn't click verify, save it too
      const currentInput = googleApiKeyInput.value.trim();
      if (currentInput !== googleApiKey) {
        googleApiKey = currentInput;
        localStorage.setItem(GOOGLE_KEY_STORAGE, googleApiKey);
        updateGoogleKeyStatusBadge();
      }

      settingsModal.classList.add('hidden');
      showToast('Settings saved successfully');
    });

    settingsBtn.addEventListener('click', () => {
      settingsModal.classList.remove('hidden');
    });

    closeSettingsModal.addEventListener('click', () => {
      settingsModal.classList.add('hidden');
    });

    settingsModal.addEventListener('click', (e) => {
      if (e.target === settingsModal) settingsModal.classList.add('hidden');
    });
  }

  // Close modals on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      faqModal.classList.add('hidden');
      settingsModal.classList.add('hidden');
    }
  });

  // --- Application Bootstrapping ---
  function init() {
    initTheme();
    initSuggestedChips();
    initSettings();
    updateGoogleKeyStatusBadge();
    loadSessionHistory();
    checkBackendHealth();
    userInput.focus();
  }

  document.addEventListener('DOMContentLoaded', init);
})();

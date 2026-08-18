/** Isolated AI provider + key storage. Gemini keys never go to OpenRouter. */

export function looksLikeAiStudioKey(key) {
  return String(key || '').trim().startsWith('AIza');
}

export function looksLikeOpenRouterKey(key) {
  const raw = String(key || '').trim();
  return raw.startsWith('sk-or-') || raw.startsWith('sk-orv-');
}

export function loadAiSettings() {
  const storedProvider = (typeof localStorage !== 'undefined' && localStorage.getItem('ai_provider')) || '';
  const legacy = (typeof localStorage !== 'undefined' && localStorage.getItem('gemini_key')) || '';
  let openrouterKey = (typeof localStorage !== 'undefined' && localStorage.getItem('openrouter_key')) || '';
  let geminiKey = (typeof localStorage !== 'undefined' && localStorage.getItem('gemini_key_native')) || '';

  if (legacy) {
    if (looksLikeOpenRouterKey(legacy) && !openrouterKey) {
      openrouterKey = legacy;
    } else if (looksLikeAiStudioKey(legacy) && !geminiKey) {
      geminiKey = legacy;
    } else if (!looksLikeOpenRouterKey(legacy) && !looksLikeAiStudioKey(legacy)) {
      if (storedProvider === 'gemini' && !geminiKey) geminiKey = legacy;
      else if (storedProvider === 'openrouter' && !openrouterKey) openrouterKey = legacy;
    }
  }

  let provider = storedProvider;
  if (provider !== 'openrouter' && provider !== 'gemini') {
    if (openrouterKey) provider = 'openrouter';
    else if (geminiKey) provider = 'gemini';
    else provider = 'openrouter';
  }
  return { provider, openrouterKey, geminiKey };
}

export function selectedAiKey({ provider, openrouterKey, geminiKey }) {
  const key = provider === 'gemini' ? geminiKey : openrouterKey;
  if (provider === 'openrouter' && looksLikeAiStudioKey(key)) return '';
  if (provider === 'gemini' && looksLikeOpenRouterKey(key)) return '';
  return key || '';
}

export function aiHeaders(settings) {
  const key = selectedAiKey(settings);
  const headers = { 'X-AI-Provider': settings.provider || 'openrouter' };
  if (key) headers['X-Gemini-Key'] = key;
  return headers;
}

export function persistAiSettings({ provider, openrouterKey, geminiKey }) {
  if (typeof localStorage === 'undefined') return;
  if (provider === 'openrouter' || provider === 'gemini') {
    localStorage.setItem('ai_provider', provider);
  }
  if (openrouterKey) localStorage.setItem('openrouter_key', openrouterKey);
  if (geminiKey) {
    localStorage.setItem('gemini_key_native', geminiKey);
    localStorage.setItem('gemini_key', geminiKey);
  }
  const leftover = localStorage.getItem('gemini_key') || '';
  if (looksLikeOpenRouterKey(leftover)) {
    localStorage.removeItem('gemini_key');
  }
}

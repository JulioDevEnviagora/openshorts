import React, { useState, useEffect } from 'react';
import { Key, Eye, EyeOff, Check } from 'lucide-react';
import { looksLikeAiStudioKey, looksLikeOpenRouterKey } from '../lib/aiKeys';

export default function KeyInput({
    provider = 'openrouter',
    onProviderChange,
    openrouterKey = '',
    geminiKey = '',
    onOpenrouterKeySet,
    onGeminiKeySet,
}) {
    const savedKey = provider === 'gemini' ? geminiKey : openrouterKey;
    const [key, setKey] = useState(savedKey || '');
    const [isVisible, setIsVisible] = useState(false);
    const [isSaved, setIsSaved] = useState(!!savedKey);
    const [mismatch, setMismatch] = useState('');

    useEffect(() => {
        setKey(savedKey || '');
        setIsSaved(!!savedKey);
        setMismatch('');
    }, [savedKey, provider]);

    const handleSave = () => {
        const trimmed = key.trim();
        if (!trimmed) return;
        if (provider === 'openrouter' && looksLikeAiStudioKey(trimmed)) {
            setMismatch('That looks like a Google AI Studio key (AIza…). Switch the provider to Gemini, or paste an OpenRouter key.');
            return;
        }
        if (provider === 'gemini' && looksLikeOpenRouterKey(trimmed)) {
            setMismatch('That looks like an OpenRouter key. Switch the provider to OpenRouter, or paste a Google AI Studio key.');
            return;
        }
        setMismatch('');
        if (provider === 'gemini') onGeminiKeySet(trimmed);
        else onOpenrouterKeySet(trimmed);
        setIsSaved(true);
    };

    const isGemini = provider === 'gemini';

    return (
        <div className="card p-4 sm:p-6 mb-8 animate-fade">
            <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-paper3 rounded-input text-brass">
                    <Key size={18} />
                </div>
                <h2 className="font-display lowercase text-lg text-ink">AI provider</h2>
            </div>

            <div className="flex gap-2 mb-4">
                <button
                    type="button"
                    onClick={() => onProviderChange && onProviderChange('openrouter')}
                    className={`px-3 py-1.5 text-xs rounded-input border ${
                        !isGemini ? 'border-brass bg-paper3 text-ink' : 'border-rule text-muted'
                    }`}
                    aria-pressed={!isGemini}
                >
                    OpenRouter
                </button>
                <button
                    type="button"
                    onClick={() => onProviderChange && onProviderChange('gemini')}
                    className={`px-3 py-1.5 text-xs rounded-input border ${
                        isGemini ? 'border-brass bg-paper3 text-ink' : 'border-rule text-muted'
                    }`}
                    aria-pressed={isGemini}
                >
                    Gemini
                </button>
            </div>

            <div className="flex flex-col sm:flex-row gap-3">
                <div className="relative sm:flex-1">
                    <input
                        type={isVisible ? "text" : "password"}
                        value={key}
                        onChange={(e) => {
                            setKey(e.target.value);
                            setIsSaved(false);
                            setMismatch('');
                        }}
                        placeholder={isGemini ? "AIza…" : "sk-or-v1-..."}
                        className="input-field pr-12 font-mono"
                    />
                    <button
                        onClick={() => setIsVisible(!isVisible)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
                    >
                        {isVisible ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                </div>
                <button
                    onClick={handleSave}
                    disabled={!key || isSaved}
                    className={isSaved ? 'badge-ok px-4 cursor-default' : 'btn-primary'}
                >
                    {isSaved ? <><Check size={14} /> Ready</> : 'Set Key'}
                </button>
            </div>
            {mismatch && (
                <p className="mt-3 text-xs text-warn">{mismatch}</p>
            )}
            <p className="mt-3 text-xs text-muted">
                {isGemini
                    ? 'Google AI Studio key for native Gemini (File API, image generation). Never sent to openrouter.ai.'
                    : 'OpenRouter key for clip scoring, layout and vision. Never sent to Google. Stored locally in your browser.'}
                <br />
                <a
                    href={isGemini ? "https://aistudio.google.com/apikey" : "https://openrouter.ai/keys"}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-brass hover:underline mt-1 inline-block"
                >
                    {isGemini ? 'Get a Gemini API key →' : 'Get an OpenRouter API key →'}
                </a>
            </p>
        </div>
    );
}

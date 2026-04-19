'use client';

import { useState, useRef, useEffect } from 'react';
import { supabase, isSupabaseConfigured } from '@/lib/supabase';

interface Message { id: string; role: 'user' | 'assistant'; content: string; }

const DEMO: Record<string, string> = {
  'default': `def process_data(data):
    """Process and validate input data."""
    if not isinstance(data, list):
        raise TypeError("Expected a list")
    results = []
    for item in data:
        if item is not None:
            results.append(str(item).strip())
    return sorted(results)`,
  'hello': `# I'm scratch — a 49M parameter model trained on Python code.
# I generate code-like patterns but don't understand logic.

def greet(name: str) -> str:
    return f"Hello, {name}!"`,
  'sort': `def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr`,
  'class': `class DataProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self._cache = {}

    def initialize(self):
        self._load_config()

    def _load_config(self):
        for key, value in self.config.items():
            setattr(self, key, value)

    def process(self, data):
        self.initialize()
        return self._transform(data)`,
};

function getDemo(prompt: string): string {
  const l = prompt.toLowerCase();
  if (l.includes('hello') || l.includes('hi')) return DEMO['hello'];
  if (l.includes('sort')) return DEMO['sort'];
  if (l.includes('class')) return DEMO['class'];
  return DEMO['default'];
}

export default function PlaygroundPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(200);
  const [showSettings, setShowSettings] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  useEffect(() => {
    fetch(`${apiUrl}/health`).then(() => setApiConnected(true)).catch(() => setApiConnected(false));
  }, [apiUrl]);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const save = async (u: string, a: string) => {
    if (!isSupabaseConfigured() || !supabase) return;
    try {
      const { data: c } = await supabase.from('conversations').insert({ title: u.slice(0, 50) }).select().single();
      if (c) await supabase.from('messages').insert([
        { conversation_id: c.id, role: 'user', content: u },
        { conversation_id: c.id, role: 'assistant', content: a },
      ]);
    } catch { /* skip */ }
  };

  const submit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;
    const um: Message = { id: Date.now().toString(), role: 'user', content: input.trim() };
    setMessages((p) => [...p, um]);
    setInput('');
    setLoading(true);
    let res = '';
    try {
      if (apiConnected) {
        const r = await fetch(`${apiUrl}/generate`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: um.content, max_tokens: maxTokens, temperature }),
        });
        const d = await r.json();
        res = d.generated_text || 'Error';
      } else {
        await new Promise((r) => setTimeout(r, 600 + Math.random() * 800));
        res = getDemo(um.content);
      }
    } catch { res = getDemo(um.content); }
    const am: Message = { id: (Date.now() + 1).toString(), role: 'assistant', content: res };
    setMessages((p) => [...p, am]);
    setLoading(false);
    save(um.content, res);
  };

  const onKey = (e: React.KeyboardEvent) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } };

  const card: React.CSSProperties = { backgroundColor: '#141414', border: '1px solid #1e1e1e', borderRadius: '10px' };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: 'calc(100vh - 56px)' }}>

      {/* Header */}
      <div style={{ borderBottom: '1px solid #1e1e1e', padding: '12px 24px' }}>
        <div style={{ maxWidth: '720px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '14px', fontWeight: 600, color: '#fff' }}>playground</span>
            <span style={{
              fontSize: '11px', padding: '2px 8px', borderRadius: '4px',
              border: `1px solid ${apiConnected ? 'rgba(34,197,94,0.3)' : '#333'}`,
              color: apiConnected ? '#22c55e' : '#737373',
            }}>
              {apiConnected ? 'live' : 'demo'}
            </span>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button onClick={() => setShowSettings(!showSettings)}
              style={{ background: 'none', border: 'none', color: '#525252', cursor: 'pointer', fontSize: '13px' }}>
              ⚙
            </button>
            <button onClick={() => setMessages([])}
              style={{ background: 'none', border: 'none', color: '#525252', cursor: 'pointer', fontSize: '13px' }}>
              🗑
            </button>
          </div>
        </div>
      </div>

      {/* Settings */}
      {showSettings && (
        <div style={{ borderBottom: '1px solid #1e1e1e', padding: '16px 24px' }}>
          <div style={{ maxWidth: '720px', margin: '0 auto', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
            <div>
              <label style={{ fontSize: '12px', color: '#525252', display: 'block', marginBottom: '6px' }}>
                temperature: {temperature}
              </label>
              <input type="range" min="0.1" max="1.5" step="0.1" value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#22c55e' }} />
            </div>
            <div>
              <label style={{ fontSize: '12px', color: '#525252', display: 'block', marginBottom: '6px' }}>
                max tokens: {maxTokens}
              </label>
              <input type="range" min="50" max="500" step="50" value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                style={{ width: '100%', accentColor: '#22c55e' }} />
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <div style={{ maxWidth: '720px', margin: '0 auto', padding: '24px' }}>
          {messages.length === 0 ? (
            <div style={{ paddingTop: '80px' }}>
              <p style={{ fontSize: '14px', color: '#525252', marginBottom: '20px' }}>
                Type a Python prompt. The model generates code-like completions.
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {['def fibonacci(n):', 'class HTTPServer:', '# sort a list', 'hello'].map((p) => (
                  <button key={p} onClick={() => { setInput(p); inputRef.current?.focus(); }}
                    style={{
                      padding: '8px 14px', border: '1px solid #1e1e1e', borderRadius: '8px',
                      fontSize: '13px', color: '#737373', backgroundColor: 'transparent', cursor: 'pointer',
                      fontFamily: "'JetBrains Mono', monospace",
                    }}>
                    {p}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {messages.map((msg) => (
                <div key={msg.id} style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                  <div style={{
                    ...card, padding: '14px 18px', maxWidth: '85%',
                    backgroundColor: msg.role === 'user' ? '#1a1a1a' : '#141414',
                  }}>
                    {msg.role === 'assistant' && (
                      <div style={{ fontSize: '10px', color: '#525252', marginBottom: '8px', fontFamily: "'JetBrains Mono', monospace" }}>
                        scratch
                      </div>
                    )}
                    <pre style={{
                      fontSize: '13px', fontFamily: "'JetBrains Mono', monospace",
                      whiteSpace: 'pre-wrap', lineHeight: 1.7, color: '#ccc', margin: 0,
                    }}>
                      {msg.content}
                    </pre>
                  </div>
                </div>
              ))}
              {loading && (
                <div style={{ ...card, padding: '14px 18px', width: 'fit-content' }}>
                  <div style={{ display: 'flex', gap: '6px' }}>
                    <span className="typing-dot" style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#525252' }} />
                    <span className="typing-dot" style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#525252' }} />
                    <span className="typing-dot" style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#525252' }} />
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <div style={{ borderTop: '1px solid #1e1e1e', padding: '16px 24px' }}>
        <form onSubmit={submit} style={{ maxWidth: '720px', margin: '0 auto' }}>
          <div style={{ ...card, display: 'flex', alignItems: 'flex-end', gap: '8px', padding: '8px' }}>
            <textarea ref={inputRef} value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey} placeholder="def binary_search(arr, target):"
              rows={1}
              style={{
                flex: 1, backgroundColor: 'transparent', padding: '10px 14px',
                fontSize: '14px', color: '#ccc', resize: 'none', outline: 'none',
                border: 'none', fontFamily: "'JetBrains Mono', monospace",
                maxHeight: '100px', lineHeight: 1.6,
              }} />
            <button type="submit" disabled={loading || !input.trim()}
              style={{
                padding: '10px', backgroundColor: '#22c55e', color: '#000',
                borderRadius: '8px', border: 'none', cursor: 'pointer',
                opacity: loading || !input.trim() ? 0.2 : 1,
                flexShrink: 0, fontSize: '14px',
              }}>
              {loading ? '...' : '↑'}
            </button>
          </div>
          <p style={{ fontSize: '11px', color: '#525252', marginTop: '8px', textAlign: 'center' }}>
            {apiConnected ? 'connected to local inference server' : 'demo mode — run api/serve.py for live inference'}
          </p>
        </form>
      </div>
    </div>
  );
}

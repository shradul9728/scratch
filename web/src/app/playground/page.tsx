'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Loader2, Sparkles, Settings2, Trash2, Code2, User } from 'lucide-react';
import { supabase, isSupabaseConfigured } from '@/lib/supabase';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

const DEMO_RESPONSES: Record<string, string> = {
  'default': `def process_data(data):
    """Process and validate input data."""
    if not isinstance(data, list):
        raise TypeError("Expected a list")
    
    results = []
    for item in data:
        if item is not None:
            results.append(str(item).strip())
    
    return sorted(results)`,
  'hello': `# Hello! I'm Scratch, a 49M parameter GPT model
# trained from scratch on Python code.
# I can generate code-like patterns, but I don't
# truly understand logic — I'm a demonstration
# of Transformer architecture, not a production model.

def greet(name: str) -> str:
    return f"Hello, {name}! Welcome to Scratch."`,
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
        self._initialized = False
    
    def initialize(self):
        """Initialize the processor."""
        if self._initialized:
            return
        self._load_config()
        self._initialized = True
    
    def _load_config(self):
        for key, value in self.config.items():
            setattr(self, key, value)
    
    def process(self, data):
        if not self._initialized:
            self.initialize()
        return self._transform(data)`,
};

function getDemoResponse(prompt: string): string {
  const lower = prompt.toLowerCase();
  if (lower.includes('hello') || lower.includes('hi')) return DEMO_RESPONSES['hello'];
  if (lower.includes('sort')) return DEMO_RESPONSES['sort'];
  if (lower.includes('class')) return DEMO_RESPONSES['class'];
  return DEMO_RESPONSES['default'];
}

export default function PlaygroundPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(200);
  const [showSettings, setShowSettings] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  useEffect(() => {
    // Check if inference API is available
    fetch(`${apiUrl}/health`).then(() => setApiConnected(true)).catch(() => setApiConnected(false));
  }, [apiUrl]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const saveToSupabase = async (userMsg: string, assistantMsg: string) => {
    if (!isSupabaseConfigured() || !supabase) return;
    try {
      const { data: conv } = await supabase
        .from('conversations')
        .insert({ title: userMsg.slice(0, 50) })
        .select()
        .single();
      
      if (conv) {
        await supabase.from('messages').insert([
          { conversation_id: conv.id, role: 'user', content: userMsg },
          { conversation_id: conv.id, role: 'assistant', content: assistantMsg },
        ]);
      }
    } catch (e) {
      console.log('Supabase not configured, skipping save');
    }
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    let response = '';

    try {
      if (apiConnected) {
        const res = await fetch(`${apiUrl}/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: userMessage.content,
            max_tokens: maxTokens,
            temperature,
          }),
        });
        const data = await res.json();
        response = data.generated_text || data.text || 'Error generating response';
      } else {
        // Demo mode - simulate typing delay
        await new Promise((r) => setTimeout(r, 800 + Math.random() * 1200));
        response = getDemoResponse(userMessage.content);
      }
    } catch {
      response = getDemoResponse(userMessage.content);
    }

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: response,
    };

    setMessages((prev) => [...prev, assistantMessage]);
    setLoading(false);

    saveToSupabase(userMessage.content, response);
  };

  const clearChat = () => {
    setMessages([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <div className="border-b border-white/5 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold">Playground</h1>
            <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
              apiConnected 
                ? 'bg-green-500/10 text-green-400 border border-green-500/20' 
                : 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${apiConnected ? 'bg-green-400' : 'bg-yellow-400'}`} />
              {apiConnected ? 'Live Model' : 'Demo Mode'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all"
            >
              <Settings2 size={18} />
            </button>
            <button
              onClick={clearChat}
              className="p-2 rounded-lg text-gray-400 hover:text-red-400 hover:bg-red-400/5 transition-all"
            >
              <Trash2 size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Settings panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-b border-white/5 overflow-hidden"
          >
            <div className="max-w-4xl mx-auto px-6 py-4 grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="text-sm text-gray-400 mb-2 block">Temperature: {temperature}</label>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-cyan-400"
                />
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>Focused</span>
                  <span>Creative</span>
                </div>
              </div>
              <div>
                <label className="text-sm text-gray-400 mb-2 block">Max Tokens: {maxTokens}</label>
                <input
                  type="range"
                  min="50"
                  max="500"
                  step="50"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full accent-cyan-400"
                />
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>Short</span>
                  <span>Long</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-6">
          {messages.length === 0 ? (
            <div className="text-center py-20">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center mx-auto mb-6">
                <Sparkles size={28} className="text-cyan-400" />
              </div>
              <h2 className="text-2xl font-bold mb-3">Start a conversation</h2>
              <p className="text-gray-500 mb-8 max-w-md mx-auto">
                Type a Python code prompt and see what Scratch generates. Try function signatures, class definitions, or comments.
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                {[
                  'def fibonacci(n):',
                  'class HTTPServer:',
                  '# Sort a list of numbers',
                  'hello',
                ].map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => { setInput(prompt); inputRef.current?.focus(); }}
                    className="px-4 py-2 rounded-xl glass text-sm text-gray-400 hover:text-cyan-400 hover:border-cyan-400/20 transition-all"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}
                >
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center flex-shrink-0 mt-1">
                      <Code2 size={16} className="text-white" />
                    </div>
                  )}
                  <div className={`max-w-[80%] rounded-2xl px-5 py-4 ${
                    msg.role === 'user'
                      ? 'bg-cyan-500/10 border border-cyan-500/20 text-gray-200'
                      : 'glass'
                  }`}>
                    <pre className="text-sm font-mono whitespace-pre-wrap leading-relaxed text-gray-300">
                      {msg.content}
                    </pre>
                  </div>
                  {msg.role === 'user' && (
                    <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center flex-shrink-0 mt-1">
                      <User size={16} className="text-gray-400" />
                    </div>
                  )}
                </motion.div>
              ))}

              {loading && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-3">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                    <Code2 size={16} className="text-white" />
                  </div>
                  <div className="glass rounded-2xl px-5 py-4">
                    <div className="flex gap-1.5">
                      <span className="typing-dot w-2 h-2 rounded-full bg-cyan-400" />
                      <span className="typing-dot w-2 h-2 rounded-full bg-cyan-400" />
                      <span className="typing-dot w-2 h-2 rounded-full bg-cyan-400" />
                    </div>
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input bar */}
      <div className="border-t border-white/5 p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="glass rounded-2xl flex items-end gap-2 p-2">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a Python prompt... (e.g. def binary_search(arr, target):)"
              rows={1}
              className="flex-1 bg-transparent px-4 py-3 text-sm text-gray-200 placeholder-gray-600 resize-none outline-none font-mono"
              style={{ maxHeight: '120px' }}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="p-3 rounded-xl bg-gradient-to-r from-cyan-500 to-purple-600 text-white disabled:opacity-30 disabled:cursor-not-allowed hover:scale-105 transition-transform flex-shrink-0"
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
            </button>
          </div>
          <p className="text-xs text-gray-600 mt-2 text-center">
            {apiConnected 
              ? 'Connected to inference server — generating live responses' 
              : 'Demo mode — showing pre-generated code samples. Run api/serve.py for live inference.'}
          </p>
        </form>
      </div>
    </div>
  );
}

'use client';

import { motion } from 'framer-motion';
import { ArrowDown, Layers, Cpu, Zap, Brain, GitBranch, RotateCcw } from 'lucide-react';

const blocks = [
  {
    id: 'input',
    title: 'Input Tokens',
    subtitle: 'Token IDs (integers)',
    color: 'from-gray-500 to-gray-600',
    icon: <Cpu size={20} />,
    description: 'Raw text is encoded into integer token IDs using a BPE tokenizer (tiktoken cl100k_base with ~100k vocab).',
  },
  {
    id: 'embeddings',
    title: 'Token + Positional Embeddings',
    subtitle: 'models/embeddings.py',
    color: 'from-blue-500 to-blue-600',
    icon: <Layers size={20} />,
    description: 'Each token ID is mapped to a 384-dimensional vector via nn.Embedding. Learnable positional embeddings encode sequence position. Both are summed and passed through dropout.',
  },
  {
    id: 'attention',
    title: 'Multi-Head Self-Attention',
    subtitle: 'models/attention.py',
    color: 'from-cyan-500 to-cyan-600',
    icon: <GitBranch size={20} />,
    description: 'Q, K, V projections split across 6 heads. Scaled dot-product attention with causal mask prevents attending to future tokens. Supports GQA (fewer KV heads) for memory efficiency.',
    code: `Attention(Q, K, V) = softmax(QK^T / √d_k) · V`
  },
  {
    id: 'ffn',
    title: 'Feed-Forward Network',
    subtitle: 'models/ffn.py',
    color: 'from-purple-500 to-purple-600',
    icon: <Zap size={20} />,
    description: 'Two-layer MLP with 4× expansion: Linear(384→1536) → GELU → Linear(1536→384) → Dropout. Can be replaced with MoE (8 experts, top-2 routing) for sparse scaling.',
  },
  {
    id: 'block',
    title: 'Transformer Block (×6)',
    subtitle: 'models/block.py',
    color: 'from-violet-500 to-violet-600',
    icon: <RotateCcw size={20} />,
    description: 'Pre-Norm architecture: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual. 6 blocks stacked sequentially. Each block preserves input dimensions (batch, seq_len, 384).',
  },
  {
    id: 'output',
    title: 'LM Head → Logits',
    subtitle: 'models/gpt.py',
    color: 'from-pink-500 to-pink-600',
    icon: <Brain size={20} />,
    description: 'Final LayerNorm → Linear projection to vocab size (100,277). Weight-tied with token embeddings. Output logits are used for next-token prediction via cross-entropy loss.',
  },
];

const advancedFeatures = [
  {
    title: 'Rotary Positional Embeddings (RoPE)',
    file: 'models/rope.py',
    description: 'Encodes relative position via rotation matrices applied to Q and K. Precomputes sin/cos frequency tables for efficient position encoding without learned embeddings.',
  },
  {
    title: 'Grouped Query Attention (GQA)',
    file: 'models/attention.py',
    description: 'Uses fewer K/V heads than Q heads (e.g., 1 KV head per 4 Q heads). Reduces KV-cache memory during inference by broadcasting KV across query groups.',
  },
  {
    title: 'Mixture of Experts (MoE)',
    file: 'models/moe.py',
    description: 'Replaces dense FFN with N expert networks. A gating router selects top-k experts per token. Auxiliary load-balancing loss prevents routing collapse.',
  },
  {
    title: 'Direct Preference Optimization (DPO)',
    file: 'finetune_dpo.py',
    description: 'Aligns model with preferences using log-probability ratios between policy and frozen reference model on chosen vs. rejected response pairs.',
  },
];

export default function ArchitecturePage() {
  return (
    <div className="min-h-screen">
      {/* Header */}
      <section className="py-16 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <h1 className="text-4xl md:text-6xl font-bold mb-4">
              <span className="gradient-text">Architecture</span>
            </h1>
            <p className="text-gray-400 text-lg max-w-2xl mx-auto">
              A visual walkthrough of the decoder-only Transformer architecture, 
              from input tokens to output logits.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Architecture Flow */}
      <section className="py-8 px-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {blocks.map((block, i) => (
            <motion.div
              key={block.id}
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              viewport={{ once: true }}
            >
              <div className="glass rounded-2xl p-6 hover:border-cyan-400/20 transition-all group">
                <div className="flex items-start gap-4">
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${block.color} flex items-center justify-center flex-shrink-0 text-white group-hover:scale-110 transition-transform`}>
                    {block.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-1">
                      <h3 className="text-lg font-semibold">{block.title}</h3>
                      <span className="text-xs font-mono text-gray-600 bg-white/5 px-2 py-0.5 rounded">{block.subtitle}</span>
                    </div>
                    <p className="text-gray-400 text-sm leading-relaxed">{block.description}</p>
                    {block.code && (
                      <div className="mt-3 bg-black/30 rounded-lg px-4 py-2 font-mono text-sm text-cyan-400">
                        {block.code}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {i < blocks.length - 1 && (
                <div className="flex justify-center py-2">
                  <ArrowDown size={20} className="text-gray-600" />
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </section>

      {/* Model Summary */}
      <section className="py-16 px-6">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="glass rounded-2xl p-8"
          >
            <h2 className="text-2xl font-bold mb-6">Model Configuration</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {[
                { label: 'Layers', value: '6' },
                { label: 'Heads', value: '6' },
                { label: 'd_model', value: '384' },
                { label: 'FFN dim', value: '1,536' },
                { label: 'Vocab size', value: '100,277' },
                { label: 'Block size', value: '256' },
                { label: 'Parameters', value: '49.2M' },
                { label: 'Precision', value: 'FP16' },
              ].map((item) => (
                <div key={item.label} className="text-center">
                  <div className="text-2xl font-bold gradient-text">{item.value}</div>
                  <div className="text-sm text-gray-500 mt-1">{item.label}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Advanced Features */}
      <section className="py-16 px-6">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">
            Advanced <span className="gradient-text">Features</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {advancedFeatures.map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="glass rounded-2xl p-6"
              >
                <h3 className="text-lg font-semibold mb-1">{feature.title}</h3>
                <span className="text-xs font-mono text-cyan-400 bg-cyan-400/10 px-2 py-0.5 rounded">{feature.file}</span>
                <p className="text-gray-400 text-sm mt-3 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

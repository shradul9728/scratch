'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowRight, Brain, Code2, Cpu, Layers, Zap, Box } from 'lucide-react';

const features = [
  {
    icon: <Layers size={24} />,
    title: 'Transformer Architecture',
    description: 'Multi-Head Attention, Feed-Forward Networks, and Pre-Norm residual connections — all from scratch.',
  },
  {
    icon: <Brain size={24} />,
    title: 'Mixture of Experts',
    description: 'Sparse MoE layers with top-k routing and load-balancing loss for efficient scaling.',
  },
  {
    icon: <Code2 size={24} />,
    title: 'Trained on Python Code',
    description: '49M parameters trained on 122MB of Python source code across 50,000 GPU steps.',
  },
  {
    icon: <Cpu size={24} />,
    title: 'GPU Accelerated',
    description: 'Mixed precision (FP16) training with gradient accumulation on NVIDIA RTX GPUs.',
  },
  {
    icon: <Zap size={24} />,
    title: 'BPE Tokenization',
    description: 'Sub-word tokenization via tiktoken (cl100k_base) for efficient code representation.',
  },
  {
    icon: <Box size={24} />,
    title: 'DPO Alignment',
    description: 'Direct Preference Optimization pipeline for aligning model outputs with human preferences.',
  },
];

const stats = [
  { value: '49.2M', label: 'Parameters' },
  { value: '50K', label: 'Training Steps' },
  { value: '122MB', label: 'Code Dataset' },
  { value: '76', label: 'Tasks Completed' },
];

const codeSnippet = `class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=384,
                 n_heads=6, n_layers=6):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)`;

export default function HomePage() {
  return (
    <div className="animated-gradient">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
        {/* Floating particles */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 rounded-full bg-cyan-400/30"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animation: `float ${3 + Math.random() * 4}s ease-in-out infinite`,
                animationDelay: `${Math.random() * 3}s`,
              }}
            />
          ))}
        </div>

        <div className="max-w-7xl mx-auto px-6 text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass text-sm text-cyan-400 mb-8">
              <Zap size={14} />
              <span>Built from scratch with PyTorch</span>
            </div>

            <h1 className="text-5xl md:text-7xl lg:text-8xl font-black mb-6 leading-tight">
              Meet{' '}
              <span className="gradient-text">Scratch</span>
            </h1>

            <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
              A complete GPT Transformer model built entirely from scratch. 
              No HuggingFace, no shortcuts — pure neural architecture with 
              49.2 million parameters trained on Python code.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/playground"
                className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-purple-600 text-white font-semibold text-lg hover:scale-105 transition-transform shadow-lg shadow-cyan-500/25"
              >
                Try Playground
                <ArrowRight size={20} />
              </Link>
              <Link
                href="/architecture"
                className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-xl border border-white/10 text-gray-300 font-semibold text-lg hover:border-cyan-400/50 hover:text-white transition-all"
              >
                View Architecture
              </Link>
            </div>
          </motion.div>

          {/* Code preview */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="mt-16 max-w-2xl mx-auto"
          >
            <div className="glass rounded-2xl overflow-hidden glow-border">
              <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
                <span className="text-xs text-gray-500 ml-2 font-mono">models/gpt.py</span>
              </div>
              <pre className="p-6 text-left text-sm font-mono text-gray-300 overflow-x-auto leading-relaxed">
                <code>{codeSnippet}</code>
              </pre>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="border-y border-white/5 py-8">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-black gradient-text mb-1">{stat.value}</div>
                <div className="text-sm text-gray-500">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-5xl font-bold mb-4">
              Built with <span className="gradient-text">precision</span>
            </h2>
            <p className="text-gray-500 text-lg max-w-xl mx-auto">
              Every component of the Transformer — from attention to alignment — implemented and tested from first principles.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="glass rounded-2xl p-6 hover:border-cyan-400/20 transition-all duration-300 group cursor-default"
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center text-cyan-400 mb-4 group-hover:scale-110 transition-transform">
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-500 text-sm leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="glass rounded-3xl p-12 glow-border"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Ready to explore?
            </h2>
            <p className="text-gray-400 mb-8 text-lg">
              Jump into the playground and see what a 49M parameter model trained on Python code can generate.
            </p>
            <Link
              href="/playground"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-purple-600 text-white font-semibold text-lg hover:scale-105 transition-transform shadow-lg shadow-cyan-500/25"
            >
              Open Playground
              <ArrowRight size={20} />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
}

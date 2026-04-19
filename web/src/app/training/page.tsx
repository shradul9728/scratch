'use client';

import { motion } from 'framer-motion';
import { TrendingDown, Clock, Cpu, Database, Gauge, BarChart3 } from 'lucide-react';

const trainingData = [
  { step: 0, train: 11.59, val: 11.59 },
  { step: 5000, train: 2.64, val: 2.78 },
  { step: 10000, train: 2.38, val: 2.75 },
  { step: 15000, train: 2.27, val: 2.38 },
  { step: 20000, train: 2.19, val: 2.31 },
  { step: 25000, train: 2.08, val: 2.45 },
  { step: 30000, train: 1.99, val: 2.31 },
  { step: 35000, train: 1.93, val: 2.08 },
  { step: 40000, train: 2.00, val: 2.04 },
  { step: 45000, train: 1.97, val: 2.06 },
  { step: 50000, train: 1.94, val: 2.07 },
];

const trainingConfig = [
  { icon: <Cpu size={18} />, label: 'GPU', value: 'NVIDIA RTX 4060 (8GB)' },
  { icon: <Clock size={18} />, label: 'Training Time', value: '~4.3 hours' },
  { icon: <Database size={18} />, label: 'Dataset', value: '122MB Python code' },
  { icon: <BarChart3 size={18} />, label: 'Total Steps', value: '50,000' },
  { icon: <Gauge size={18} />, label: 'Batch Size', value: '8 × 4 accum = 32' },
  { icon: <TrendingDown size={18} />, label: 'Final Val Loss', value: '2.07' },
];

const hyperparams = [
  { param: 'Learning Rate', value: '6e-4 → 6e-5 (cosine)' },
  { param: 'Warmup Steps', value: '500' },
  { param: 'Weight Decay', value: '0.1' },
  { param: 'Betas', value: '(0.9, 0.95)' },
  { param: 'Grad Clip', value: '1.0' },
  { param: 'Dropout', value: '0.1' },
  { param: 'Precision', value: 'FP16 (AMP)' },
  { param: 'Tokenizer', value: 'BPE (tiktoken cl100k)' },
  { param: 'Context Length', value: '256 tokens' },
  { param: 'Optimizer', value: 'AdamW (decoupled)' },
];

function LossChart() {
  const maxLoss = 12;
  const chartH = 300;
  const chartW = 700;
  const padL = 60;
  const padB = 40;
  const padT = 20;
  const padR = 20;
  const w = chartW - padL - padR;
  const h = chartH - padT - padB;

  const toX = (step: number) => padL + (step / 50000) * w;
  const toY = (loss: number) => padT + (1 - loss / maxLoss) * h;

  const trainPath = trainingData
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.step)} ${toY(d.train)}`)
    .join(' ');
  const valPath = trainingData
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.step)} ${toY(d.val)}`)
    .join(' ');

  return (
    <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full" style={{ maxHeight: '360px' }}>
      {/* Grid lines */}
      {[0, 2, 4, 6, 8, 10, 12].map((v) => (
        <g key={v}>
          <line x1={padL} y1={toY(v)} x2={chartW - padR} y2={toY(v)} stroke="rgba(255,255,255,0.05)" />
          <text x={padL - 10} y={toY(v) + 4} textAnchor="end" fill="#555" fontSize="11">{v}</text>
        </g>
      ))}
      {[0, 10000, 20000, 30000, 40000, 50000].map((s) => (
        <text key={s} x={toX(s)} y={chartH - 10} textAnchor="middle" fill="#555" fontSize="11">{s / 1000}k</text>
      ))}

      {/* Lines */}
      <path d={trainPath} fill="none" stroke="url(#trainGrad)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d={valPath} fill="none" stroke="url(#valGrad)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="6 3" />

      {/* Dots */}
      {trainingData.map((d) => (
        <g key={d.step}>
          <circle cx={toX(d.step)} cy={toY(d.train)} r="3" fill="#00d4ff" />
          <circle cx={toX(d.step)} cy={toY(d.val)} r="3" fill="#a855f7" />
        </g>
      ))}

      {/* Gradients */}
      <defs>
        <linearGradient id="trainGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#00d4ff" />
          <stop offset="100%" stopColor="#06b6d4" />
        </linearGradient>
        <linearGradient id="valGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#a855f7" />
          <stop offset="100%" stopColor="#ec4899" />
        </linearGradient>
      </defs>

      {/* Legend */}
      <circle cx={chartW - 150} cy={padT + 10} r="4" fill="#00d4ff" />
      <text x={chartW - 140} y={padT + 14} fill="#aaa" fontSize="12">Train Loss</text>
      <circle cx={chartW - 150} cy={padT + 30} r="4" fill="#a855f7" />
      <text x={chartW - 140} y={padT + 34} fill="#aaa" fontSize="12">Val Loss</text>
    </svg>
  );
}

export default function TrainingPage() {
  return (
    <div className="min-h-screen">
      {/* Header */}
      <section className="py-16 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <h1 className="text-4xl md:text-6xl font-bold mb-4">
              <span className="gradient-text">Training</span>
            </h1>
            <p className="text-gray-400 text-lg max-w-2xl mx-auto">
              50,000 steps on 122MB of Python code using an NVIDIA RTX 4060 with mixed precision training.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Training Config Cards */}
      <section className="px-6 pb-12">
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-3 gap-4">
          {trainingConfig.map((item, i) => (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              viewport={{ once: true }}
              className="glass rounded-xl p-4"
            >
              <div className="flex items-center gap-2 text-cyan-400 mb-2">
                {item.icon}
                <span className="text-xs text-gray-500 uppercase tracking-wider">{item.label}</span>
              </div>
              <div className="text-lg font-semibold">{item.value}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Loss Chart */}
      <section className="px-6 pb-16">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="glass rounded-2xl p-8"
          >
            <h2 className="text-2xl font-bold mb-6">Loss Curve</h2>
            <LossChart />
          </motion.div>
        </div>
      </section>

      {/* Hyperparameters Table */}
      <section className="px-6 pb-16">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="glass rounded-2xl p-8"
          >
            <h2 className="text-2xl font-bold mb-6">Hyperparameters</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-3">
              {hyperparams.map((hp) => (
                <div key={hp.param} className="flex items-center justify-between py-2 border-b border-white/5">
                  <span className="text-gray-400 text-sm">{hp.param}</span>
                  <span className="text-sm font-mono text-cyan-400">{hp.value}</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Training Steps Table */}
      <section className="px-6 pb-16">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="glass rounded-2xl p-8"
          >
            <h2 className="text-2xl font-bold mb-6">Training Progress</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 text-gray-400 font-medium">Step</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-medium">Train Loss</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-medium">Val Loss</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-medium">Improvement</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingData.map((d, i) => (
                    <tr key={d.step} className="border-b border-white/5 hover:bg-white/2">
                      <td className="py-3 px-4 font-mono">{d.step.toLocaleString()}</td>
                      <td className="py-3 px-4 font-mono text-cyan-400">{d.train.toFixed(2)}</td>
                      <td className="py-3 px-4 font-mono text-purple-400">{d.val.toFixed(2)}</td>
                      <td className="py-3 px-4">
                        {i > 0 && (
                          <span className={`text-xs px-2 py-1 rounded ${
                            d.val < trainingData[i-1].val 
                              ? 'bg-green-500/10 text-green-400'
                              : 'bg-red-500/10 text-red-400'
                          }`}>
                            {d.val < trainingData[i-1].val ? '↓' : '↑'} {Math.abs(d.val - trainingData[i-1].val).toFixed(2)}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}

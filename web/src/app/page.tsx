'use client';

import Link from 'next/link';

const codeSnippet = `class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)`;

export default function HomePage() {
  return (
    <div style={{ maxWidth: '720px', margin: '0 auto', padding: '0 24px' }}>

      {/* Hero */}
      <section style={{ paddingTop: '80px', paddingBottom: '64px' }}>
        <h1 style={{ fontSize: '40px', fontWeight: 700, marginBottom: '16px', color: '#fff' }}>
          scratch
        </h1>
        <p style={{ fontSize: '17px', lineHeight: 1.8, color: '#a3a3a3', maxWidth: '560px', marginBottom: '32px' }}>
          A GPT model written from scratch in PyTorch. 49 million parameters,
          trained on 122MB of Python source code. No HuggingFace, no pretrained
          weights. Every layer written by hand.
        </p>
        <div style={{ display: 'flex', gap: '12px' }}>
          <Link
            href="/playground"
            style={{
              padding: '10px 20px', backgroundColor: '#22c55e', color: '#000',
              fontSize: '14px', fontWeight: 600, borderRadius: '8px', textDecoration: 'none',
            }}
          >
            Try it
          </Link>
          <a
            href="https://github.com/shradul9728/scratch"
            target="_blank" rel="noreferrer"
            style={{
              padding: '10px 20px', border: '1px solid #333', color: '#a3a3a3',
              fontSize: '14px', fontWeight: 500, borderRadius: '8px', textDecoration: 'none',
            }}
          >
            Source code
          </a>
        </div>
      </section>

      {/* What it is */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '20px', color: '#fff' }}>
          What this is
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', fontSize: '15px', lineHeight: 1.8, color: '#a3a3a3' }}>
          <p>
            This is a decoder-only Transformer — the same architecture behind GPT-2, GPT-3,
            and most LLMs. The difference is that every component is implemented from first
            principles: attention, embeddings, feed-forward layers, the training loop, the
            tokenizer integration.
          </p>
          <p>
            It&apos;s not a production model. It&apos;s 49M parameters (GPT-2 small is 124M, GPT-3
            is 175B). It was trained for 4 hours on a single RTX 4060 laptop GPU. It can
            generate code-like patterns but it doesn&apos;t understand logic.
          </p>
          <p>
            The point is the code itself — seeing how attention works, how loss drops over
            50K steps, how BPE tokenization handles Python syntax.
          </p>
        </div>
      </section>

      {/* Numbers */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '24px', color: '#fff' }}>
          Numbers
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px' }}>
          {[
            { value: '49.2M', label: 'parameters' },
            { value: '50,000', label: 'training steps' },
            { value: '122MB', label: 'code dataset' },
            { value: '~4.3h', label: 'training time' },
          ].map((stat) => (
            <div key={stat.label}>
              <div style={{ fontSize: '28px', fontWeight: 700, color: '#fff' }}>{stat.value}</div>
              <div style={{ fontSize: '13px', color: '#525252', marginTop: '4px' }}>{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Code */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '8px', color: '#fff' }}>
          What it looks like
        </h2>
        <p style={{ fontSize: '14px', color: '#525252', marginBottom: '16px' }}>
          The full model definition. This is the actual code.
        </p>
        <div style={{ backgroundColor: '#141414', border: '1px solid #1e1e1e', borderRadius: '12px', overflow: 'hidden' }}>
          <div style={{ padding: '10px 16px', borderBottom: '1px solid #1e1e1e', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#333' }} />
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#333' }} />
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#333' }} />
            <span style={{ fontSize: '12px', color: '#525252', fontFamily: "'JetBrains Mono', monospace", marginLeft: '8px' }}>
              models/gpt.py
            </span>
          </div>
          <pre style={{
            padding: '20px', fontSize: '13px', fontFamily: "'JetBrains Mono', monospace",
            color: '#a3a3a3', overflowX: 'auto', lineHeight: 1.7, margin: 0,
          }}>
            <code>{codeSnippet}</code>
          </pre>
        </div>
      </section>

      {/* Repo contents */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '24px', color: '#fff' }}>
          What&apos;s in the repo
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {[
            { name: 'models/', desc: 'Attention, FFN, MoE, RoPE, Transformer block, full GPT — 7 files' },
            { name: 'utils/', desc: 'Config loader, data pipeline, BPE tokenizer, dataset, training — 5 files' },
            { name: 'tests/', desc: '56 unit tests covering every module' },
            { name: 'train.py', desc: 'Training loop with FP16 mixed precision and gradient accumulation' },
            { name: 'api/serve.py', desc: 'FastAPI server for model inference' },
            { name: 'web/', desc: 'This website — Next.js, deployed on Vercel' },
            { name: 'finetune_sft.py', desc: 'Supervised fine-tuning script' },
            { name: 'finetune_dpo.py', desc: 'Direct Preference Optimization alignment' },
          ].map((item) => (
            <div key={item.name} style={{ display: 'flex', gap: '16px', fontSize: '14px', lineHeight: 1.6 }}>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#22c55e', width: '140px', flexShrink: 0, fontSize: '13px' }}>
                {item.name}
              </span>
              <span style={{ color: '#737373' }}>{item.desc}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Training result */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '12px', color: '#fff' }}>
          Training result
        </h2>
        <p style={{ fontSize: '15px', color: '#a3a3a3', marginBottom: '24px', lineHeight: 1.8 }}>
          Loss dropped from 11.59 to 1.94 over 50K steps. Validation loss landed at 2.07.
          The model learned Python syntax — indentation, function signatures, class structures —
          but doesn&apos;t produce correct logic. That&apos;s expected at this scale.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
          {[
            { label: 'train loss', value: '1.94' },
            { label: 'val loss', value: '2.07' },
            { label: 'gpu', value: '4060' },
          ].map((item) => (
            <div key={item.label} style={{ backgroundColor: '#141414', border: '1px solid #1e1e1e', borderRadius: '10px', padding: '16px' }}>
              <div style={{ fontSize: '12px', color: '#525252', marginBottom: '4px' }}>{item.label}</div>
              <div style={{ fontSize: '24px', fontWeight: 700, color: '#fff' }}>{item.value}</div>
            </div>
          ))}
        </div>
        <Link href="/training" style={{ display: 'inline-block', marginTop: '16px', fontSize: '14px', color: '#22c55e' }}>
          Full training metrics →
        </Link>
      </section>
    </div>
  );
}

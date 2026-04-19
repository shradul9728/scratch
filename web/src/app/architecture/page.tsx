'use client';

const layers = [
  { name: 'Input', detail: 'Raw text → integer token IDs via tiktoken BPE (cl100k_base, 100K vocab)' },
  { name: 'Embedding', detail: 'Token embedding (100K × 384) + learned positional embedding (256 × 384). Summed, then dropout.' },
  { name: 'Attention', detail: 'Multi-head self-attention. 6 heads, 64 dims each. QKV projections → scaled dot-product → causal mask → concat → output projection.' },
  { name: 'FFN', detail: 'Two linear layers with GELU activation: 384 → 1536 → 384. Can swap for MoE (8 experts, top-2 routing).' },
  { name: 'Block × 6', detail: 'Pre-norm: LayerNorm → Attention → residual → LayerNorm → FFN → residual. Stacked 6 times.' },
  { name: 'Output', detail: 'Final LayerNorm → linear projection to vocab size (100,277). Weights tied with token embedding.' },
];

const extras = [
  { name: 'RoPE', file: 'models/rope.py', desc: 'Rotary position encoding. Applies rotation matrices to Q and K based on position.' },
  { name: 'GQA', file: 'models/attention.py', desc: 'Grouped query attention. Uses fewer KV heads than Q heads to reduce memory.' },
  { name: 'MoE', file: 'models/moe.py', desc: 'Mixture of experts. Routes each token to top-k experts. Auxiliary loss prevents collapse.' },
  { name: 'DPO', file: 'finetune_dpo.py', desc: 'Direct preference optimization. Aligns model using chosen/rejected pairs via log-prob ratios.' },
];

const card = {
  backgroundColor: '#141414',
  border: '1px solid #1e1e1e',
  borderRadius: '10px',
  padding: '16px',
};

export default function ArchitecturePage() {
  return (
    <div style={{ maxWidth: '720px', margin: '0 auto', padding: '0 24px' }}>

      <section style={{ paddingTop: '80px', paddingBottom: '48px' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '8px', color: '#fff' }}>Architecture</h1>
        <p style={{ fontSize: '15px', color: '#737373' }}>
          Decoder-only Transformer. 6 layers, 6 heads, 384 dimensions, 49.2M parameters.
        </p>
      </section>

      {/* Data flow */}
      <section style={{ paddingBottom: '48px' }}>
        {layers.map((layer, i) => (
          <div key={layer.name} style={{
            display: 'flex', gap: '20px', padding: '16px 0',
            borderTop: i > 0 ? '1px solid #141414' : 'none',
          }}>
            <span style={{
              fontFamily: "'JetBrains Mono', monospace", color: '#22c55e',
              fontSize: '13px', width: '100px', flexShrink: 0, paddingTop: '2px',
            }}>
              {layer.name}
            </span>
            <p style={{ fontSize: '14px', color: '#a3a3a3', lineHeight: 1.7 }}>{layer.detail}</p>
          </div>
        ))}
      </section>

      {/* Config grid */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '20px', color: '#fff' }}>Config</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          {[
            ['layers', '6'], ['heads', '6'], ['d_model', '384'], ['ffn_dim', '1,536'],
            ['vocab', '100,277'], ['context', '256'], ['params', '49.2M'], ['precision', 'FP16'],
          ].map(([k, v]) => (
            <div key={k} style={card}>
              <div style={{ fontSize: '11px', color: '#525252', fontFamily: "'JetBrains Mono', monospace" }}>{k}</div>
              <div style={{ fontSize: '18px', fontWeight: 600, marginTop: '4px', color: '#fff' }}>{v}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Also implemented */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '20px', color: '#fff' }}>Also implemented</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {extras.map((e) => (
            <div key={e.name} style={card}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '6px' }}>
                <span style={{ fontSize: '15px', fontWeight: 600, color: '#fff' }}>{e.name}</span>
                <span style={{ fontSize: '11px', fontFamily: "'JetBrains Mono', monospace", color: '#525252' }}>{e.file}</span>
              </div>
              <p style={{ fontSize: '14px', color: '#737373', lineHeight: 1.7 }}>{e.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Math */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '16px', color: '#fff' }}>The math</h2>
        <div style={{ ...card, fontFamily: "'JetBrains Mono', monospace", fontSize: '14px', color: '#a3a3a3' }}>
          <p>Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) · V</p>
          <p style={{ marginTop: '8px', color: '#525252', fontSize: '12px' }}>where d_k = 64 (384 / 6 heads)</p>
        </div>
      </section>
    </div>
  );
}

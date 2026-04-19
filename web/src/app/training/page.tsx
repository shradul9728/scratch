'use client';

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

const card: React.CSSProperties = {
  backgroundColor: '#141414', border: '1px solid #1e1e1e', borderRadius: '10px', padding: '16px',
};

function LossChart() {
  const maxLoss = 12, chartH = 260, chartW = 640;
  const padL = 40, padB = 30, padT = 10, padR = 10;
  const w = chartW - padL - padR, h = chartH - padT - padB;
  const toX = (step: number) => padL + (step / 50000) * w;
  const toY = (loss: number) => padT + (1 - loss / maxLoss) * h;
  const trainPath = trainingData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.step)} ${toY(d.train)}`).join(' ');
  const valPath = trainingData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.step)} ${toY(d.val)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${chartW} ${chartH}`} style={{ width: '100%' }}>
      {[0, 2, 4, 6, 8, 10, 12].map((v) => (
        <g key={v}>
          <line x1={padL} y1={toY(v)} x2={chartW - padR} y2={toY(v)} stroke="#1e1e1e" />
          <text x={padL - 8} y={toY(v) + 4} textAnchor="end" fill="#525252" fontSize="10">{v}</text>
        </g>
      ))}
      {[0, 10000, 20000, 30000, 40000, 50000].map((s) => (
        <text key={s} x={toX(s)} y={chartH - 6} textAnchor="middle" fill="#525252" fontSize="10">{s / 1000}k</text>
      ))}
      <path d={trainPath} fill="none" stroke="#22c55e" strokeWidth="2" />
      <path d={valPath} fill="none" stroke="#525252" strokeWidth="2" strokeDasharray="4 2" />
      {trainingData.map((d) => (
        <g key={d.step}>
          <circle cx={toX(d.step)} cy={toY(d.train)} r="2.5" fill="#22c55e" />
          <circle cx={toX(d.step)} cy={toY(d.val)} r="2.5" fill="#525252" />
        </g>
      ))}
      <circle cx={chartW - 90} cy={padT + 8} r="3" fill="#22c55e" />
      <text x={chartW - 82} y={padT + 12} fill="#999" fontSize="10">train</text>
      <circle cx={chartW - 90} cy={padT + 24} r="3" fill="#525252" />
      <text x={chartW - 82} y={padT + 28} fill="#999" fontSize="10">val</text>
    </svg>
  );
}

export default function TrainingPage() {
  return (
    <div style={{ maxWidth: '720px', margin: '0 auto', padding: '0 24px' }}>

      <section style={{ paddingTop: '80px', paddingBottom: '48px' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '8px', color: '#fff' }}>Training</h1>
        <p style={{ fontSize: '15px', color: '#737373' }}>
          50,000 steps on 122MB of Python source code. Single RTX 4060, FP16, ~4.3 hours.
        </p>
      </section>

      {/* Setup cards */}
      <section style={{ paddingBottom: '48px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
          {[
            ['gpu', 'RTX 4060 (8GB)'],
            ['time', '~4.3 hours'],
            ['dataset', '122MB Python'],
            ['steps', '50,000'],
            ['batch', '8 × 4 accum'],
            ['val loss', '2.07'],
          ].map(([k, v]) => (
            <div key={k} style={card}>
              <div style={{ fontSize: '10px', color: '#525252', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{k}</div>
              <div style={{ fontSize: '15px', fontWeight: 600, marginTop: '4px', color: '#fff' }}>{v}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Chart */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '16px', color: '#fff' }}>Loss curve</h2>
        <div style={{ ...card, padding: '20px' }}>
          <LossChart />
        </div>
      </section>

      {/* Hyperparameters */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '16px', color: '#fff' }}>Hyperparameters</h2>
        <div style={{ backgroundColor: '#141414', border: '1px solid #1e1e1e', borderRadius: '10px', overflow: 'hidden' }}>
          {[
            ['learning rate', '6e-4 → 6e-5 (cosine)'],
            ['warmup', '500 steps'],
            ['weight decay', '0.1'],
            ['optimizer', 'AdamW (β1=0.9, β2=0.95)'],
            ['grad clip', '1.0'],
            ['dropout', '0.1'],
            ['precision', 'FP16 (torch.amp)'],
            ['tokenizer', 'tiktoken cl100k_base'],
          ].map(([k, v], i) => (
            <div key={k} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '12px 16px', fontSize: '14px',
              borderTop: i > 0 ? '1px solid #1a1a1a' : 'none',
            }}>
              <span style={{ color: '#737373' }}>{k}</span>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#22c55e', fontSize: '12px' }}>{v}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Step table */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '16px', color: '#fff' }}>Step by step</h2>
        <div style={{ backgroundColor: '#141414', border: '1px solid #1e1e1e', borderRadius: '10px', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #1e1e1e' }}>
                {['step', 'train', 'val', 'delta'].map((h) => (
                  <th key={h} style={{ textAlign: 'left', padding: '10px 16px', color: '#525252', fontWeight: 500, fontSize: '12px' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {trainingData.map((d, i) => (
                <tr key={d.step} style={{ borderTop: i > 0 ? '1px solid #1a1a1a' : 'none' }}>
                  <td style={{ padding: '8px 16px', fontFamily: "'JetBrains Mono', monospace", fontSize: '12px' }}>
                    {d.step.toLocaleString()}
                  </td>
                  <td style={{ padding: '8px 16px', fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', color: '#22c55e' }}>
                    {d.train.toFixed(2)}
                  </td>
                  <td style={{ padding: '8px 16px', fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', color: '#a3a3a3' }}>
                    {d.val.toFixed(2)}
                  </td>
                  <td style={{ padding: '8px 16px', fontSize: '12px' }}>
                    {i > 0 && (
                      <span style={{ color: d.val < trainingData[i-1].val ? '#22c55e' : '#525252' }}>
                        {d.val < trainingData[i-1].val ? '↓' : '↑'}{Math.abs(d.val - trainingData[i-1].val).toFixed(2)}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Honest note */}
      <section style={{ paddingTop: '48px', paddingBottom: '48px', borderTop: '1px solid #1e1e1e' }}>
        <div style={{ ...card, fontSize: '14px', color: '#737373', lineHeight: 1.8 }}>
          <p style={{ fontWeight: 600, color: '#a3a3a3', marginBottom: '8px' }}>What this means</p>
          <p>
            The model learned Python syntax — indentation, colons, parentheses, function/class
            patterns. But at 49M parameters with 256-token context, it doesn&apos;t produce logically
            correct code. It repeats patterns and generates plausible-looking but incorrect logic.
            That&apos;s normal for a model this size.
          </p>
        </div>
      </section>
    </div>
  );
}

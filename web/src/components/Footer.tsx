import Link from 'next/link';

export default function Footer() {
  return (
    <footer style={{ borderTop: '1px solid #1e1e1e', marginTop: '80px' }}>
      <div style={{
        maxWidth: '720px', margin: '0 auto', padding: '32px 24px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        flexWrap: 'wrap', gap: '16px',
      }}>
        <p style={{ fontSize: '13px', color: '#525252' }}>
          scratch — built with PyTorch
        </p>
        <div style={{ display: 'flex', gap: '20px', fontSize: '13px' }}>
          <Link href="/playground" style={{ color: '#525252', textDecoration: 'none' }}>Playground</Link>
          <Link href="/architecture" style={{ color: '#525252', textDecoration: 'none' }}>Architecture</Link>
          <a href="https://github.com/shradul9728/scratch" target="_blank" rel="noreferrer"
            style={{ color: '#525252', textDecoration: 'none' }}>GitHub</a>
        </div>
      </div>
    </footer>
  );
}

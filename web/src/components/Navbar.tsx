'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

const navLinks = [
  { href: '/', label: 'Home' },
  { href: '/playground', label: 'Playground' },
  { href: '/architecture', label: 'Architecture' },
  { href: '/training', label: 'Training' },
];

export default function Navbar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 50,
      backgroundColor: '#0a0a0a', borderBottom: '1px solid #1e1e1e',
    }}>
      <div style={{
        maxWidth: '720px', margin: '0 auto', padding: '0 24px',
        height: '56px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <Link href="/" style={{ fontSize: '16px', fontWeight: 600, color: '#fff', textDecoration: 'none' }}>
          scratch
        </Link>

        {/* Desktop */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
          className="hidden md:flex">
          {navLinks.map((link) => (
            <Link key={link.href} href={link.href} style={{
              padding: '6px 14px', borderRadius: '6px', fontSize: '14px',
              color: pathname === link.href ? '#22c55e' : '#737373',
              textDecoration: 'none',
            }}>
              {link.label}
            </Link>
          ))}
        </div>

        {/* Mobile toggle */}
        <button onClick={() => setOpen(!open)} className="md:hidden"
          style={{ color: '#737373', background: 'none', border: 'none', cursor: 'pointer', fontSize: '20px' }}>
          {open ? '✕' : '☰'}
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div style={{ borderTop: '1px solid #1e1e1e', padding: '12px 24px', backgroundColor: '#0a0a0a' }}
          className="md:hidden">
          {navLinks.map((link) => (
            <Link key={link.href} href={link.href} onClick={() => setOpen(false)} style={{
              display: 'block', padding: '10px 0', fontSize: '14px',
              color: pathname === link.href ? '#22c55e' : '#737373',
              textDecoration: 'none',
            }}>
              {link.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}

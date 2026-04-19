import { Sparkles, ExternalLink } from 'lucide-react';
import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="border-t border-white/5 mt-20">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-purple-500 flex items-center justify-center">
                <Sparkles size={18} className="text-white" />
              </div>
              <span className="text-xl font-bold gradient-text">Scratch</span>
            </div>
            <p className="text-gray-500 text-sm max-w-md leading-relaxed">
              A GPT model built entirely from scratch using PyTorch. No HuggingFace, no shortcuts — 
              pure Transformer architecture with attention, MoE, RoPE, and DPO alignment.
            </p>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-4 uppercase tracking-wider">Explore</h4>
            <ul className="space-y-2">
              {[
                { href: '/playground', label: 'Playground' },
                { href: '/architecture', label: 'Architecture' },
                { href: '/training', label: 'Training' },
              ].map((link) => (
                <li key={link.href}>
                  <Link href={link.href} className="text-gray-500 hover:text-cyan-400 text-sm transition-colors">
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-gray-300 mb-4 uppercase tracking-wider">Tech Stack</h4>
            <ul className="space-y-2 text-gray-500 text-sm">
              <li>PyTorch 2.x</li>
              <li>Next.js 15</li>
              <li>Supabase</li>
              <li>Vercel</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-white/5 mt-8 pt-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-gray-600 text-sm">
            © 2026 Scratch — GPT From Scratch. Built with ❤️ and PyTorch.
          </p>
          <div className="flex items-center gap-4">
            <a href="https://github.com" target="_blank" rel="noreferrer" className="text-gray-600 hover:text-cyan-400 transition-colors flex items-center gap-1 text-sm">
              <ExternalLink size={14} /> GitHub
            </a>
            <a href="https://twitter.com" target="_blank" rel="noreferrer" className="text-gray-600 hover:text-cyan-400 transition-colors flex items-center gap-1 text-sm">
              <ExternalLink size={14} /> Twitter
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}

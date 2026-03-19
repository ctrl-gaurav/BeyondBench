import { Hexagon, Github, FileText, Package, ExternalLink, Mail, ArrowUpRight } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

export default function Footer() {
  const { isDark } = useTheme()

  const linkClass = `inline-flex items-center gap-1.5 transition-all duration-300 group/link ${
    isDark ? 'text-gray-500 hover:text-bb-accent' : 'text-gray-500 hover:text-bb-accent-dark'
  }`

  return (
    <footer className={`border-t mt-20 relative overflow-hidden ${
      isDark
        ? 'border-bb-dark-50/20 bg-bb-dark-600/50'
        : 'border-bb-light-300/50 bg-white/40'
    }`}>
      {/* Animated gradient line at top */}
      <div className="absolute top-0 left-0 right-0 h-[2px] overflow-hidden">
        <div className="h-full w-full bg-gradient-to-r from-transparent via-bb-accent/50 to-transparent animate-gradient" style={{ backgroundSize: '200% 100%' }} />
      </div>

      {/* Subtle animated mesh background */}
      <div className={`absolute inset-0 pointer-events-none ${isDark ? 'opacity-[0.02]' : 'opacity-[0.015]'}`}>
        <div className="absolute top-10 left-10 w-32 h-32 border border-bb-accent/20 rounded-full animate-float-slow" />
        <div className="absolute bottom-10 right-20 w-24 h-24 border border-bb-teal/20 rounded-full animate-float" />
        <div className="absolute top-20 right-1/3 w-16 h-16 border border-bb-mint/20 rotate-45 animate-float-delayed" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 relative">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2.5 mb-4 group">
              <div className="relative">
                <Hexagon className={`w-6 h-6 transition-all duration-500 group-hover:rotate-[30deg] ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
                <div className={`absolute inset-0 w-6 h-6 rounded-full blur-lg transition-all duration-500 group-hover:blur-xl ${
                  isDark ? 'bg-bb-accent/20 group-hover:bg-bb-accent/40' : 'bg-bb-accent-dark/15 group-hover:bg-bb-accent-dark/30'
                }`} />
              </div>
              <span className={`text-lg font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Beyond<span className={isDark ? 'text-bb-accent neon-text' : 'text-bb-accent-dark'}>Bench</span>
              </span>
            </div>
            <p className={`text-sm leading-relaxed mb-3 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
              Contamination-Resistant Evaluation of Reasoning in Language Models.
            </p>
            <a
              href="https://openreview.net/forum?id=mIKqVWGjwI"
              target="_blank"
              rel="noopener noreferrer"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-full transition-all hover:scale-105 ${
                isDark
                  ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/20 hover:bg-bb-accent/20 hover:shadow-[0_0_15px_rgba(0,230,118,0.15)]'
                  : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/20 hover:bg-bb-accent-dark/20'
              }`}
            >
              Accepted at ICLR 2026
              <ArrowUpRight className="w-3 h-3" />
            </a>
          </div>

          {/* Links */}
          <div>
            <h4 className={`text-sm font-semibold uppercase tracking-widest mb-4 ${
              isDark ? 'text-gray-300' : 'text-gray-700'
            }`}>Links</h4>
            <ul className="space-y-3 text-sm">
              <li>
                <a href="https://arxiv.org/abs/2509.24210" target="_blank" rel="noopener noreferrer" className={linkClass}>
                  <FileText className="w-3.5 h-3.5 transition-transform duration-300 group-hover/link:scale-110" />
                  <span className="hover-line">arXiv Paper</span>
                </a>
              </li>
              <li>
                <a href="https://openreview.net/forum?id=mIKqVWGjwI" target="_blank" rel="noopener noreferrer" className={linkClass}>
                  <ExternalLink className="w-3.5 h-3.5 transition-transform duration-300 group-hover/link:scale-110" />
                  <span className="hover-line">OpenReview</span>
                </a>
              </li>
              <li>
                <a href="https://github.com/ctrl-gaurav/BeyondBench" target="_blank" rel="noopener noreferrer" className={linkClass}>
                  <Github className="w-3.5 h-3.5 transition-transform duration-300 group-hover/link:scale-110" />
                  <span className="hover-line">GitHub Repository</span>
                </a>
              </li>
              <li>
                <a href="https://pypi.org/project/beyondbench/" target="_blank" rel="noopener noreferrer" className={linkClass}>
                  <Package className="w-3.5 h-3.5 transition-transform duration-300 group-hover/link:scale-110" />
                  <span className="hover-line">PyPI Package</span>
                </a>
              </li>
              <li>
                <a href="https://ctrl-gaurav.github.io/BeyondBench/" target="_blank" rel="noopener noreferrer" className={linkClass}>
                  <ArrowUpRight className="w-3.5 h-3.5 transition-transform duration-300 group-hover/link:scale-110" />
                  <span className="hover-line">Leaderboard</span>
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className={`text-sm font-semibold uppercase tracking-widest mb-4 ${
              isDark ? 'text-gray-300' : 'text-gray-700'
            }`}>Contact</h4>
            <ul className="space-y-3 text-sm">
              <li>
                <a href="mailto:gks@vt.edu" className={linkClass}>
                  <Mail className="w-3.5 h-3.5 transition-transform duration-300 group-hover/link:scale-110" />
                  <span className="hover-line">gks@vt.edu</span>
                </a>
              </li>
              <li className={`${isDark ? 'text-gray-600' : 'text-gray-400'}`}>
                Virginia Tech, CS Department
              </li>
            </ul>
          </div>
        </div>

        <div className={`mt-10 pt-6 border-t text-center text-xs ${
          isDark ? 'border-bb-dark-50/10 text-gray-600' : 'border-bb-light-300/50 text-gray-400'
        }`}>
          <span className="inline-flex items-center gap-2">
            Built by the BeyondBench Team
            <span className={isDark ? 'text-bb-accent/40' : 'text-bb-accent-dark/40'}>&bull;</span>
            MIT License
          </span>
        </div>
      </div>
    </footer>
  )
}

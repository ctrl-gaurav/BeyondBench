import { Link, useLocation } from 'react-router-dom'
import { Trophy, BookOpen, Github, FileText, Hexagon, Sun, Moon, ExternalLink, Menu, X, Package, Sparkles } from 'lucide-react'
import { useState, useEffect } from 'react'
import { useTheme } from '../context/ThemeContext'

export default function Navbar() {
  const location = useLocation()
  const isActive = (path) => location.pathname === path
  const { isDark, toggleTheme } = useTheme()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10)
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // Close mobile menu on route change
  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  const navLinkClass = (path) =>
    `relative flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
      isActive(path)
        ? isDark
          ? 'bg-bb-accent/10 text-bb-accent shadow-[0_0_12px_rgba(0,230,118,0.15)] nav-active-indicator'
          : 'bg-bb-accent-dark/10 text-bb-accent-dark nav-active-indicator'
        : isDark
          ? 'text-gray-400 hover:text-white hover:bg-bb-dark-300/50'
          : 'text-gray-500 hover:text-gray-900 hover:bg-bb-light-200'
    }`

  const extLinkClass = `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 group ${
    isDark
      ? 'text-gray-400 hover:text-white hover:bg-bb-dark-300/50'
      : 'text-gray-500 hover:text-gray-900 hover:bg-bb-light-200'
  }`

  return (
    <>
      <nav className={`sticky top-0 z-50 backdrop-blur-2xl border-b transition-all duration-500 ${
        isDark
          ? `bg-bb-dark-500/70 ${scrolled ? 'border-bb-accent/10 shadow-[0_4px_30px_rgba(0,230,118,0.05)]' : 'border-bb-dark-50/20'}`
          : `bg-white/70 ${scrolled ? 'border-bb-accent-dark/10 shadow-md' : 'border-bb-light-300/50 shadow-sm'}`
      }`}>
        {/* Green accent line at bottom of nav */}
        <div className={`absolute bottom-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent ${
          isDark ? 'via-bb-accent/20' : 'via-bb-accent-dark/15'
        } to-transparent transition-opacity duration-500 ${scrolled ? 'opacity-100' : 'opacity-0'}`} />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3 group">
              <div className="relative">
                <Hexagon className={`w-8 h-8 transition-all duration-500 group-hover:scale-110 group-hover:rotate-[30deg] ${
                  isDark ? 'text-bb-accent' : 'text-bb-accent-dark'
                }`} />
                <div className={`absolute inset-0 w-8 h-8 rounded-full blur-xl transition-all duration-500 group-hover:blur-2xl ${
                  isDark ? 'bg-bb-accent/20 group-hover:bg-bb-accent/40' : 'bg-bb-accent-dark/15 group-hover:bg-bb-accent-dark/25'
                }`} />
                {/* Sparkle on hover */}
                <Sparkles className={`absolute -top-1 -right-1 w-3 h-3 transition-all duration-300 ${
                  isDark ? 'text-bb-accent' : 'text-bb-accent-dark'
                } opacity-0 group-hover:opacity-100 group-hover:scale-100 scale-0`} />
              </div>
              <span className="text-xl font-bold tracking-tight">
                <span className={isDark ? 'text-white' : 'text-gray-900'}>Beyond</span>
                <span className={`${isDark ? 'text-bb-accent neon-text' : 'text-bb-accent-dark'}`}>Bench</span>
              </span>
              <a
                href="https://openreview.net/forum?id=mIKqVWGjwI"
                target="_blank"
                rel="noopener noreferrer"
                className={`hidden sm:inline-flex items-center gap-1 px-2.5 py-1 text-[10px] font-mono font-bold rounded-full transition-all hover:scale-105 ${
                  isDark
                    ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/20 hover:bg-bb-accent/20 hover:shadow-[0_0_10px_rgba(0,230,118,0.15)]'
                    : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/20 hover:bg-bb-accent-dark/20'
                }`}
                onClick={(e) => e.stopPropagation()}
              >
                ICLR 2026
                <ExternalLink className="w-2.5 h-2.5" />
              </a>
            </Link>

            {/* Desktop Nav */}
            <div className="hidden md:flex items-center gap-1">
              <Link to="/" className={navLinkClass('/')}>
                <Trophy className="w-4 h-4" />
                <span>Leaderboard</span>
              </Link>
              <Link to="/docs" className={navLinkClass('/docs')}>
                <BookOpen className="w-4 h-4" />
                <span>Documentation</span>
              </Link>

              <div className={`w-px h-6 mx-2 ${isDark ? 'bg-bb-dark-50/20' : 'bg-bb-light-300'}`} />

              <a href="https://arxiv.org/abs/2509.24210" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
                <FileText className="w-4 h-4" />
                <span>arXiv</span>
                <ExternalLink className="w-3 h-3 opacity-0 -translate-x-1 group-hover:opacity-50 group-hover:translate-x-0 transition-all duration-200" />
              </a>
              <a href="https://github.com/ctrl-gaurav/BeyondBench" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
                <Github className="w-4 h-4" />
                <span>GitHub</span>
                <ExternalLink className="w-3 h-3 opacity-0 -translate-x-1 group-hover:opacity-50 group-hover:translate-x-0 transition-all duration-200" />
              </a>
              <a href="https://pypi.org/project/beyondbench/" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
                <Package className="w-4 h-4" />
                <span>PyPI</span>
                <ExternalLink className="w-3 h-3 opacity-0 -translate-x-1 group-hover:opacity-50 group-hover:translate-x-0 transition-all duration-200" />
              </a>

              <div className={`w-px h-6 mx-2 ${isDark ? 'bg-bb-dark-50/20' : 'bg-bb-light-300'}`} />

              {/* Theme Toggle - enhanced */}
              <button
                onClick={toggleTheme}
                className={`relative flex items-center justify-center w-9 h-9 rounded-lg transition-all duration-500 overflow-hidden ${
                  isDark
                    ? 'text-gray-400 hover:text-yellow-300 hover:bg-bb-dark-300/50 hover:shadow-[0_0_12px_rgba(253,224,71,0.15)]'
                    : 'text-gray-500 hover:text-amber-500 hover:bg-bb-light-200 hover:shadow-[0_0_12px_rgba(245,158,11,0.15)]'
                }`}
                title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                <div className={`transition-all duration-500 ${isDark ? 'rotate-0 scale-100' : 'rotate-180 scale-0 absolute'}`}>
                  <Sun className="w-4 h-4" />
                </div>
                <div className={`transition-all duration-500 ${!isDark ? 'rotate-0 scale-100' : '-rotate-180 scale-0 absolute'}`}>
                  <Moon className="w-4 h-4" />
                </div>
              </button>
            </div>

            {/* Mobile Controls */}
            <div className="flex items-center gap-2 md:hidden">
              <button
                onClick={toggleTheme}
                className={`flex items-center justify-center w-9 h-9 rounded-lg transition-all ${
                  isDark
                    ? 'text-gray-400 hover:text-yellow-300 hover:bg-bb-dark-300/50'
                    : 'text-gray-500 hover:text-amber-500 hover:bg-bb-light-200'
                }`}
              >
                {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </button>
              <button
                onClick={() => setMobileOpen(!mobileOpen)}
                className={`flex items-center justify-center w-9 h-9 rounded-lg transition-all ${
                  isDark
                    ? 'text-gray-400 hover:text-white hover:bg-bb-dark-300/50'
                    : 'text-gray-500 hover:text-gray-900 hover:bg-bb-light-200'
                }`}
              >
                <div className="relative w-5 h-5">
                  <X className={`w-5 h-5 absolute inset-0 transition-all duration-300 ${mobileOpen ? 'rotate-0 opacity-100' : 'rotate-90 opacity-0'}`} />
                  <Menu className={`w-5 h-5 absolute inset-0 transition-all duration-300 ${mobileOpen ? '-rotate-90 opacity-0' : 'rotate-0 opacity-100'}`} />
                </div>
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile Menu - slide in overlay */}
      <div className={`fixed inset-0 z-40 md:hidden transition-all duration-300 ${
        mobileOpen ? 'pointer-events-auto' : 'pointer-events-none'
      }`}>
        {/* Backdrop */}
        <div
          className={`absolute inset-0 backdrop-blur-sm transition-opacity duration-300 ${
            mobileOpen ? 'opacity-100' : 'opacity-0'
          } ${isDark ? 'bg-bb-dark-500/60' : 'bg-black/20'}`}
          onClick={() => setMobileOpen(false)}
        />
        {/* Panel */}
        <div className={`absolute top-16 right-0 bottom-0 w-72 transition-transform duration-300 ease-out border-l ${
          mobileOpen ? 'translate-x-0' : 'translate-x-full'
        } ${isDark ? 'bg-bb-dark-400/95 backdrop-blur-2xl border-bb-dark-50/20' : 'bg-white/95 backdrop-blur-2xl border-bb-light-300/50'}`}>
          <div className="p-4 flex flex-col gap-1">
            <Link to="/" className={navLinkClass('/')} onClick={() => setMobileOpen(false)}>
              <Trophy className="w-4 h-4" />
              <span>Leaderboard</span>
            </Link>
            <Link to="/docs" className={navLinkClass('/docs')} onClick={() => setMobileOpen(false)}>
              <BookOpen className="w-4 h-4" />
              <span>Documentation</span>
            </Link>
            <div className={`h-px my-2 ${isDark ? 'bg-bb-dark-50/20' : 'bg-bb-light-300/50'}`} />
            <a href="https://arxiv.org/abs/2509.24210" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
              <FileText className="w-4 h-4" /> arXiv Paper
            </a>
            <a href="https://openreview.net/forum?id=mIKqVWGjwI" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
              <ExternalLink className="w-4 h-4" /> OpenReview
            </a>
            <a href="https://github.com/ctrl-gaurav/BeyondBench" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
              <Github className="w-4 h-4" /> GitHub
            </a>
            <a href="https://pypi.org/project/beyondbench/" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
              <Package className="w-4 h-4" /> PyPI
            </a>
          </div>
        </div>
      </div>
    </>
  )
}

import { Link, useLocation } from 'react-router-dom'
import { Trophy, BookOpen, Github, FileText, Hexagon, Sun, Moon, ExternalLink, Menu, X, Package, BarChart3, Layers } from 'lucide-react'
import { useState, useEffect } from 'react'
import { useTheme } from '../context/ThemeContext'
import { usePyPIVersion } from '../hooks/usePyPIVersion'

export default function Navbar() {
  const location = useLocation()
  const isActive = (path) => location.pathname === path
  const { isDark, toggleTheme } = useTheme()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [isScrolled, setIsScrolled] = useState(false)
  const pypiVersion = usePyPIVersion()

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  const navLinkClass = (path) =>
    `relative flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
      isActive(path)
        ? isDark
          ? 'bg-bb-accent/10 text-bb-accent shadow-[0_0_12px_rgba(0,230,118,0.15)]'
          : 'bg-bb-accent-dark/10 text-bb-accent-dark'
        : isDark
          ? 'text-gray-400 hover:text-white hover:bg-bb-dark-300/50'
          : 'text-gray-500 hover:text-gray-900 hover:bg-bb-light-200'
    }`

  const extLinkClass = `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
    isDark
      ? 'text-gray-400 hover:text-white hover:bg-bb-dark-300/50'
      : 'text-gray-500 hover:text-gray-900 hover:bg-bb-light-200'
  }`

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
      isScrolled
        ? isDark
          ? 'bg-bb-dark-500/80 backdrop-blur-2xl border-b border-bb-dark-50/20 shadow-lg shadow-black/30'
          : 'bg-white/80 backdrop-blur-2xl border-b border-bb-light-300/50 shadow-lg shadow-gray-200/30'
        : 'bg-transparent'
    }`}>
      {/* Gradient top accent line */}
      {isScrolled && (
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-bb-accent/30 to-transparent" />
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="relative">
              <Hexagon className={`w-8 h-8 transition-all duration-300 group-hover:scale-110 ${
                isDark ? 'text-bb-accent' : 'text-bb-accent-dark'
              }`} />
              <div className={`absolute inset-0 w-8 h-8 rounded-full blur-xl transition-all group-hover:blur-2xl ${
                isDark ? 'bg-bb-accent/20 group-hover:bg-bb-accent/40' : 'bg-bb-accent-dark/15 group-hover:bg-bb-accent-dark/25'
              }`} />
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
                  ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/20 hover:bg-bb-accent/20'
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
              {isActive('/') && (
                <span className={`absolute -bottom-px left-2 right-2 h-px ${
                  isDark
                    ? 'bg-gradient-to-r from-transparent via-bb-accent to-transparent'
                    : 'bg-gradient-to-r from-transparent via-bb-accent-dark to-transparent'
                }`} />
              )}
            </Link>
            <Link to="/docs" className={navLinkClass('/docs')}>
              <BookOpen className="w-4 h-4" />
              <span>Documentation</span>
              {isActive('/docs') && (
                <span className={`absolute -bottom-px left-2 right-2 h-px ${
                  isDark
                    ? 'bg-gradient-to-r from-transparent via-bb-accent to-transparent'
                    : 'bg-gradient-to-r from-transparent via-bb-accent-dark to-transparent'
                }`} />
              )}
            </Link>
            <Link to="/results" className={navLinkClass('/results')}>
              <BarChart3 className="w-4 h-4" />
              <span>Results</span>
              {isActive('/results') && (
                <span className={`absolute -bottom-px left-2 right-2 h-px ${
                  isDark
                    ? 'bg-gradient-to-r from-transparent via-bb-accent to-transparent'
                    : 'bg-gradient-to-r from-transparent via-bb-accent-dark to-transparent'
                }`} />
              )}
            </Link>
            <Link to="/tasks" className={navLinkClass('/tasks')}>
              <Layers className="w-4 h-4" />
              <span>Tasks</span>
              {isActive('/tasks') && (
                <span className={`absolute -bottom-px left-2 right-2 h-px ${
                  isDark
                    ? 'bg-gradient-to-r from-transparent via-bb-accent to-transparent'
                    : 'bg-gradient-to-r from-transparent via-bb-accent-dark to-transparent'
                }`} />
              )}
            </Link>

            <div className={`w-px h-6 mx-2 ${isDark ? 'bg-bb-dark-50/20' : 'bg-bb-light-300'}`} />

            <a href="https://arxiv.org/abs/2509.24210" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
              <FileText className="w-4 h-4" />
              <span>arXiv</span>
            </a>
            <a href="https://github.com/ctrl-gaurav/BeyondBench" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
              <Github className="w-4 h-4" />
              <span>GitHub</span>
            </a>
            <a href="https://pypi.org/project/beyondbench/" target="_blank" rel="noopener noreferrer" className={extLinkClass}>
              <Package className="w-4 h-4" />
              <span>PyPI{pypiVersion ? ` v${pypiVersion}` : ''}</span>
            </a>

            <div className={`w-px h-6 mx-2 ${isDark ? 'bg-bb-dark-50/20' : 'bg-bb-light-300'}`} />

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className={`relative flex items-center justify-center w-9 h-9 rounded-lg transition-all duration-300 ${
                isDark
                  ? 'text-gray-400 hover:text-yellow-300 hover:bg-bb-dark-300/50'
                  : 'text-gray-500 hover:text-amber-500 hover:bg-bb-light-200'
              }`}
              title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
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
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileOpen && (
          <div className={`md:hidden py-4 border-t animate-slide-down ${
            isDark ? 'border-bb-dark-50/20' : 'border-bb-light-300/50'
          }`}>
            <div className="flex flex-col gap-1">
              <Link to="/" className={navLinkClass('/')} onClick={() => setMobileOpen(false)}>
                <Trophy className="w-4 h-4" />
                <span>Leaderboard</span>
              </Link>
              <Link to="/docs" className={navLinkClass('/docs')} onClick={() => setMobileOpen(false)}>
                <BookOpen className="w-4 h-4" />
                <span>Documentation</span>
              </Link>
              <Link to="/results" className={navLinkClass('/results')} onClick={() => setMobileOpen(false)}>
                <BarChart3 className="w-4 h-4" />
                <span>Results</span>
              </Link>
              <Link to="/tasks" className={navLinkClass('/tasks')} onClick={() => setMobileOpen(false)}>
                <Layers className="w-4 h-4" />
                <span>Tasks</span>
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
                <Package className="w-4 h-4" /> PyPI{pypiVersion ? ` v${pypiVersion}` : ''}
              </a>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}

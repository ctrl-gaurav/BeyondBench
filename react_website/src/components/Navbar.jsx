import { Link, useLocation } from 'react-router-dom'
import { Trophy, BookOpen, Github, FileText, Hexagon } from 'lucide-react'

export default function Navbar() {
  const location = useLocation()
  const isActive = (path) => location.pathname === path

  return (
    <nav className="sticky top-0 z-50 bg-bb-dark-500/80 backdrop-blur-xl border-b border-bb-dark-50/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="relative">
              <Hexagon className="w-8 h-8 text-bb-accent group-hover:text-bb-mint transition-colors" />
              <div className="absolute inset-0 w-8 h-8 bg-bb-accent/20 rounded-full blur-lg group-hover:bg-bb-accent/30 transition-all" />
            </div>
            <span className="text-xl font-bold">
              <span className="text-white">Beyond</span>
              <span className="text-bb-accent">Bench</span>
            </span>
            <span className="hidden sm:inline-block px-2 py-0.5 text-[10px] font-mono font-bold bg-bb-accent/10 text-bb-accent border border-bb-accent/20 rounded-full">
              ICLR 2026
            </span>
          </Link>

          <div className="flex items-center gap-1">
            <Link
              to="/"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                isActive('/') ? 'bg-bb-accent/10 text-bb-accent' : 'text-gray-400 hover:text-white hover:bg-bb-dark-300/50'
              }`}
            >
              <Trophy className="w-4 h-4" />
              <span className="hidden sm:inline">Leaderboard</span>
            </Link>
            <Link
              to="/docs"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                isActive('/docs') ? 'bg-bb-accent/10 text-bb-accent' : 'text-gray-400 hover:text-white hover:bg-bb-dark-300/50'
              }`}
            >
              <BookOpen className="w-4 h-4" />
              <span className="hidden sm:inline">Documentation</span>
            </Link>
            <a
              href="https://arxiv.org/abs/2509.24210"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-bb-dark-300/50 transition-all"
            >
              <FileText className="w-4 h-4" />
              <span className="hidden sm:inline">Paper</span>
            </a>
            <a
              href="https://github.com/ctrl-gaurav/BeyondBench"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-bb-dark-300/50 transition-all"
            >
              <Github className="w-4 h-4" />
              <span className="hidden sm:inline">GitHub</span>
            </a>
          </div>
        </div>
      </div>
    </nav>
  )
}

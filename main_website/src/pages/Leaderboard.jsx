import { useState, useMemo, useEffect, useRef, useCallback } from 'react'
import { Trophy, Search, ChevronDown, ChevronUp, Filter, BarChart3, Zap, Brain, Target, Hexagon, TrendingUp, Award, Crown, Medal, Star, ArrowUpRight, ExternalLink } from 'lucide-react'
import { modelData, paperInfo } from '../data/modelData'
import { useTheme } from '../context/ThemeContext'

// ---- IntersectionObserver-based scroll reveal ----
function useInView(options = {}) {
  const ref = useRef(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setInView(true)
        observer.unobserve(el)
      }
    }, { threshold: 0.1, ...options })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return [ref, inView]
}

function ScrollReveal({ children, delay = 0, className = '', direction = 'up' }) {
  const [ref, inView] = useInView()
  const transforms = {
    up: 'translate-y-8',
    down: 'translate-y-[-2rem]',
    left: 'translate-x-8',
    right: 'translate-x-[-2rem]',
    scale: 'scale-95',
  }

  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${
        inView ? 'opacity-100 translate-y-0 translate-x-0 scale-100' : `opacity-0 ${transforms[direction]}`
      } ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  )
}

// ---- Animated counter ----
function AnimatedCounter({ value, duration = 1500, suffix = '', prefix = '' }) {
  const [count, setCount] = useState(0)
  const [ref, inView] = useInView()
  const hasAnimated = useRef(false)

  useEffect(() => {
    if (!inView || hasAnimated.current) return
    hasAnimated.current = true

    const numValue = typeof value === 'string' ? parseFloat(value.replace(/[^0-9.]/g, '')) : value
    if (isNaN(numValue)) {
      setCount(value)
      return
    }

    const startTime = performance.now()
    const animate = (currentTime) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3) // easeOutCubic
      setCount(Math.round(eased * numValue))
      if (progress < 1) requestAnimationFrame(animate)
    }
    requestAnimationFrame(animate)
  }, [inView, value, duration])

  return <span ref={ref}>{prefix}{typeof value === 'string' && value.startsWith('>') ? `>${count}` : count}{suffix}</span>
}

// ---- Floating particles in hero ----
function HeroParticles({ isDark }) {
  const particles = useMemo(() =>
    Array.from({ length: 20 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 4 + 2,
      delay: Math.random() * 8,
      duration: Math.random() * 8 + 8,
      opacity: Math.random() * 0.3 + 0.1,
    })), [])

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map(p => (
        <div
          key={p.id}
          className={`absolute rounded-full ${isDark ? 'bg-bb-accent' : 'bg-bb-accent-dark'}`}
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: `${p.size}px`,
            height: `${p.size}px`,
            opacity: p.opacity,
            animation: `particle-float-${(p.id % 3) + 1} ${p.duration}s ease-in-out ${p.delay}s infinite`,
            filter: `blur(${p.size > 4 ? 1 : 0}px)`,
          }}
        />
      ))}
    </div>
  )
}

// ---- Floating hexagons ----
function FloatingHexagons({ isDark }) {
  const hexagons = useMemo(() =>
    Array.from({ length: 6 }, (_, i) => ({
      id: i,
      x: 10 + Math.random() * 80,
      y: 10 + Math.random() * 80,
      size: 20 + Math.random() * 30,
      delay: i * 2,
      rotation: Math.random() * 360,
    })), [])

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {hexagons.map(h => (
        <Hexagon
          key={h.id}
          className={`absolute ${isDark ? 'text-bb-accent/[0.04]' : 'text-bb-accent-dark/[0.05]'}`}
          style={{
            left: `${h.x}%`,
            top: `${h.y}%`,
            width: `${h.size}px`,
            height: `${h.size}px`,
            transform: `rotate(${h.rotation}deg)`,
            animation: `particle-float-2 ${12 + h.id * 2}s ease-in-out ${h.delay}s infinite`,
          }}
        />
      ))}
    </div>
  )
}

// ---- Hero title with glitch effect on mount ----
function GlitchTitle({ isDark }) {
  const [glitch, setGlitch] = useState(true)

  useEffect(() => {
    const timer = setTimeout(() => setGlitch(false), 800)
    return () => clearTimeout(timer)
  }, [])

  return (
    <h1 className="text-6xl sm:text-7xl lg:text-8xl font-black mb-5 tracking-tight leading-none relative">
      <span className={`inline-block ${isDark ? 'text-white' : 'text-gray-900'} ${glitch ? 'animate-glitch-1' : ''}`}>
        Beyond
      </span>
      <span className={`inline-block ${isDark ? 'text-bb-accent neon-text' : 'text-bb-accent-dark'} ${glitch ? 'animate-glitch-2' : ''}`}>
        Bench
      </span>
    </h1>
  )
}

const FAMILY_COLORS = {
  'Qwen': '#7c3aed',
  'Google': '#0ea5e9',
  'Microsoft': '#f59e0b',
  'Meta': '#3b82f6',
  'Mistral': '#ef4444',
  'OpenAI': '#10b981',
  'HuggingFace': '#f97316',
}

function getFamily(model) {
  return model.split('/')[0]
}

// ---- Stat Card with glow and tilt ----
function StatCard({ icon: Icon, label, value, sub, delay = 0, rawValue, suffix = '', prefix = '' }) {
  const { isDark } = useTheme()
  const isSpecial = value === '>10^15'
  return (
    <ScrollReveal delay={delay} direction="up">
      <div
        className={`group relative overflow-hidden rounded-xl p-5 flex flex-col items-center text-center tilt-card shine-effect ${
          isDark
            ? 'bg-bb-dark-300/60 backdrop-blur-xl border border-bb-dark-50/30 hover:border-bb-accent/40 glow-border-pulse'
            : 'bg-white/70 backdrop-blur-xl border border-bb-light-300/60 hover:border-bb-accent-dark/40 shadow-sm hover:shadow-lg'
        }`}
      >
        {/* Corner accent */}
        <div className={`absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl rounded-bl-full transition-opacity duration-300 ${
          isDark ? 'from-bb-accent/5 to-transparent opacity-0 group-hover:opacity-100' : 'from-bb-accent-dark/5 to-transparent opacity-0 group-hover:opacity-100'
        }`} />

        <div className={`relative w-12 h-12 rounded-xl flex items-center justify-center mb-3 transition-all duration-500 group-hover:scale-110 group-hover:rotate-3 ${
          isDark
            ? 'bg-bb-accent/10 group-hover:bg-bb-accent/20 group-hover:shadow-[0_0_20px_rgba(0,230,118,0.15)]'
            : 'bg-bb-accent-dark/10 group-hover:bg-bb-accent-dark/15'
        }`}>
          <Icon className={`w-5 h-5 transition-transform duration-500 group-hover:scale-110 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
        </div>
        <div className={`text-2xl font-bold font-mono relative ${isDark ? 'text-white' : 'text-gray-900'}`}>
          {isSpecial ? (
            <span>&gt;10<sup className="text-sm">15</sup></span>
          ) : (
            <AnimatedCounter value={rawValue || parseInt(value)} suffix={suffix} prefix={prefix} />
          )}
        </div>
        <div className={`text-xs mt-1.5 font-medium tracking-wide ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
        {sub && <div className={`text-[10px] mt-0.5 ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{sub}</div>}
      </div>
    </ScrollReveal>
  )
}

// ---- Podium Card with particles, spotlight, shimmer ----
function PodiumCard({ model, index }) {
  const { isDark } = useTheme()
  const configs = [
    {
      medal: <Crown className="w-6 h-6 animate-crown-bounce" />,
      color: 'text-yellow-400',
      bg: isDark ? 'from-yellow-400/10 via-yellow-400/5 to-transparent' : 'from-yellow-400/15 via-yellow-400/5 to-transparent',
      border: isDark ? 'border-yellow-400/30 hover:border-yellow-400/50' : 'border-yellow-400/40 hover:border-yellow-400/60',
      glow: isDark ? 'shadow-[0_0_40px_rgba(250,204,21,0.12)]' : 'shadow-[0_0_30px_rgba(250,204,21,0.1)]',
      label: '1st Place',
      spotlight: true,
    },
    {
      medal: <Medal className="w-5 h-5" />,
      color: 'text-gray-300',
      bg: isDark ? 'from-gray-300/10 via-gray-300/5 to-transparent' : 'from-gray-400/10 via-gray-400/5 to-transparent',
      border: isDark ? 'border-gray-400/20 hover:border-gray-400/40' : 'border-gray-400/30 hover:border-gray-400/50',
      glow: '',
      label: '2nd Place',
      spotlight: false,
    },
    {
      medal: <Star className="w-5 h-5" />,
      color: 'text-amber-600',
      bg: isDark ? 'from-amber-600/10 via-amber-600/5 to-transparent' : 'from-amber-600/10 via-amber-600/5 to-transparent',
      border: isDark ? 'border-amber-600/20 hover:border-amber-600/40' : 'border-amber-600/30 hover:border-amber-600/50',
      glow: '',
      label: '3rd Place',
      spotlight: false,
    },
  ]
  const c = configs[index]
  const m = model

  return (
    <ScrollReveal delay={index * 150} direction="up">
      <div className={`group relative overflow-hidden rounded-xl border p-5 transition-all duration-300 hover:scale-[1.03] hover:-translate-y-2 shine-effect ${c.border} ${c.glow} ${
        isDark ? 'bg-bb-dark-300/60 backdrop-blur-xl' : 'bg-white/80 backdrop-blur-xl shadow-sm hover:shadow-xl'
      } ${c.spotlight ? 'podium-spotlight' : ''}`}>
        {/* Background gradient */}
        <div className={`absolute inset-0 bg-gradient-to-br ${c.bg} opacity-60`} />
        {/* Corner decoration */}
        <div className={`absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl ${c.bg} rounded-bl-full opacity-40`} />

        {/* Spotlight glow for #1 */}
        {index === 0 && (
          <div className={`absolute -top-20 left-1/2 -translate-x-1/2 w-40 h-40 rounded-full blur-3xl pointer-events-none ${
            isDark ? 'bg-yellow-400/10' : 'bg-yellow-400/5'
          } animate-spotlight`} />
        )}

        <div className="relative">
          <div className="flex items-start gap-3">
            <div className={`flex items-center justify-center w-10 h-10 rounded-lg transition-all duration-300 group-hover:scale-110 ${
              isDark ? 'bg-bb-dark-400/60' : 'bg-white/60'
            }`}>
              <span className={c.color}>{c.medal}</span>
            </div>
            <div className="flex-1 min-w-0">
              <div className={`text-sm font-semibold truncate ${isDark ? 'text-white' : 'text-gray-900'}`}>
                {m.model.split('/')[1] || m.model}
              </div>
              <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                {getFamily(m.model)} {m.params && `\u00b7 ${m.params}`}
              </div>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-3 gap-3">
            {[
              { val: m.accuracy, label: 'Overall', accent: true },
              { val: m.easy_acc, label: 'Easy', accent: false },
              { val: m.hard_acc, label: 'Hard', accent: false },
            ].map(({ val, label, accent }) => (
              <div key={label} className={`text-center p-2 rounded-lg transition-all duration-300 group-hover:scale-[1.02] ${
                isDark ? 'bg-bb-dark-400/40' : 'bg-bb-light-200/60'
              }`}>
                <div className={`text-lg font-bold font-mono ${
                  accent
                    ? isDark ? 'text-bb-accent' : 'text-bb-accent-dark'
                    : isDark ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  {val.toFixed(1)}%
                </div>
                <div className={`text-[10px] font-medium ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{label}</div>
                {/* Mini progress bar */}
                <div className={`mini-progress mt-1 ${isDark ? 'bg-bb-dark-500/60' : 'bg-bb-light-300/40'}`}>
                  <div
                    className={`mini-progress-fill ${accent
                      ? isDark ? 'bg-bb-accent/60' : 'bg-bb-accent-dark/60'
                      : isDark ? 'bg-gray-500/40' : 'bg-gray-400/40'
                    }`}
                    style={{ width: `${val}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </ScrollReveal>
  )
}

// ---- Wave divider SVG ----
function WaveDivider({ isDark }) {
  return (
    <div className="relative h-12 my-8 overflow-hidden">
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1200 48" preserveAspectRatio="none">
        <path
          d="M0,24 C200,48 400,0 600,24 C800,48 1000,0 1200,24"
          fill="none"
          stroke={isDark ? 'rgba(0,230,118,0.15)' : 'rgba(0,200,83,0.15)'}
          strokeWidth="1"
        />
        <path
          d="M0,24 C200,0 400,48 600,24 C800,0 1000,48 1200,24"
          fill="none"
          stroke={isDark ? 'rgba(0,191,165,0.1)' : 'rgba(0,191,165,0.1)'}
          strokeWidth="1"
        />
      </svg>
      <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 rounded-full ${
        isDark ? 'bg-bb-accent shadow-[0_0_10px_rgba(0,230,118,0.5)]' : 'bg-bb-accent-dark shadow-[0_0_10px_rgba(0,200,83,0.5)]'
      }`} />
    </div>
  )
}

export default function Leaderboard() {
  const { isDark } = useTheme()
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState('accuracy')
  const [sortDir, setSortDir] = useState('desc')
  const [familyFilter, setFamilyFilter] = useState('all')
  const [showFilters, setShowFilters] = useState(false)
  const [expandedRow, setExpandedRow] = useState(null)

  const families = useMemo(() => {
    const fams = new Set(modelData.map(m => getFamily(m.model)))
    return ['all', ...Array.from(fams).sort()]
  }, [])

  const sortedData = useMemo(() => {
    let data = [...modelData]
    if (search) {
      const q = search.toLowerCase()
      data = data.filter(m => m.model.toLowerCase().includes(q))
    }
    if (familyFilter !== 'all') {
      data = data.filter(m => getFamily(m.model) === familyFilter)
    }
    data.sort((a, b) => {
      const mult = sortDir === 'desc' ? -1 : 1
      if (typeof a[sortKey] === 'string') {
        return a[sortKey].localeCompare(b[sortKey]) * mult
      }
      return (a[sortKey] - b[sortKey]) * mult
    })
    return data.map((m, i) => ({ ...m, displayRank: i + 1 }))
  }, [search, sortKey, sortDir, familyFilter])

  const globalRanked = useMemo(() => {
    const ranked = [...modelData].sort((a, b) => b.accuracy - a.accuracy)
    const map = {}
    ranked.forEach((m, i) => { map[m.model] = i + 1 })
    return map
  }, [])

  function toggleSort(key) {
    if (sortKey === key) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  function SortHeader({ k, children, className = '' }) {
    const active = sortKey === k
    return (
      <th
        className={`px-3 py-3.5 text-[11px] font-semibold uppercase tracking-wider cursor-pointer select-none transition-all duration-200 th-glow ${
          active
            ? isDark ? 'text-bb-accent' : 'text-bb-accent-dark'
            : isDark ? 'text-gray-500 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600'
        } ${className}`}
        onClick={() => toggleSort(k)}
      >
        <div className="flex items-center gap-1 group/sort">
          {children}
          <span className={`transition-all duration-300 ${active ? 'opacity-100 scale-100' : 'opacity-0 scale-75 group-hover/sort:opacity-50 group-hover/sort:scale-100'}`}>
            {active && sortDir === 'desc' ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
          </span>
        </div>
      </th>
    )
  }

  const top3 = useMemo(() => {
    return [...modelData].sort((a, b) => b.accuracy - a.accuracy).slice(0, 3)
  }, [])

  // Maximum accuracy for progress bar scaling
  const maxAccuracy = useMemo(() => {
    return Math.max(...modelData.map(m => m.accuracy))
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* ============ HERO SECTION ============ */}
      <div className="text-center mb-14 relative">
        {/* Radial glow behind hero */}
        <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] rounded-full blur-3xl pointer-events-none ${
          isDark ? 'bg-bb-accent/[0.04]' : 'bg-bb-accent-dark/[0.03]'
        }`} />

        {/* Floating particles */}
        <HeroParticles isDark={isDark} />

        {/* Floating hexagons */}
        <FloatingHexagons isDark={isDark} />

        <div className="relative pt-12 pb-8">
          {/* Hexagon icon */}
          <ScrollReveal delay={0} direction="scale">
            <div className="flex items-center justify-center gap-3 mb-6">
              <div className="relative">
                <Hexagon className={`w-14 h-14 animate-float ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
                <div className={`absolute inset-0 w-14 h-14 rounded-full blur-2xl animate-glow-pulse ${
                  isDark ? 'bg-bb-accent/30' : 'bg-bb-accent-dark/20'
                }`} />
                {/* Inner sparkle dots */}
                <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 rounded-full ${
                  isDark ? 'bg-bb-accent' : 'bg-bb-accent-dark'
                } animate-glow-pulse`} />
              </div>
            </div>
          </ScrollReveal>

          {/* Title with glitch effect */}
          <ScrollReveal delay={150} direction="up">
            <GlitchTitle isDark={isDark} />
          </ScrollReveal>

          {/* Subtitle */}
          <ScrollReveal delay={300} direction="up">
            <p className={`text-lg sm:text-xl max-w-2xl mx-auto mb-6 font-light tracking-wide ${
              isDark ? 'text-gray-400' : 'text-gray-600'
            }`}>
              Contamination-Resistant Evaluation of Reasoning in Language Models
            </p>
          </ScrollReveal>

          {/* Badges */}
          <ScrollReveal delay={450} direction="up">
          <div className="flex flex-wrap items-center justify-center gap-2 text-xs">
            <a
              href="https://openreview.net/forum?id=mIKqVWGjwI"
              target="_blank"
              rel="noopener noreferrer"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full font-semibold transition-all hover:scale-105 ${
                isDark
                  ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/20 hover:bg-bb-accent/20 hover:shadow-[0_0_15px_rgba(0,230,118,0.15)]'
                  : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/20 hover:bg-bb-accent-dark/15'
              }`}
            >
              Accepted at ICLR 2026
              <ExternalLink className="w-3 h-3" />
            </a>
            <a
              href="https://arxiv.org/abs/2509.24210"
              target="_blank"
              rel="noopener noreferrer"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono transition-all hover:scale-105 ${
                isDark
                  ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300 hover:border-bb-accent/20'
                  : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700'
              }`}
            >
              arXiv:2509.24210
              <ArrowUpRight className="w-3 h-3" />
            </a>
            <a
              href="https://openreview.net/forum?id=mIKqVWGjwI"
              target="_blank"
              rel="noopener noreferrer"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono transition-all hover:scale-105 ${
                isDark
                  ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300 hover:border-bb-accent/20'
                  : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700'
              }`}
            >
              OpenReview
              <ArrowUpRight className="w-3 h-3" />
            </a>
            <a
              href="https://github.com/ctrl-gaurav/BeyondBench"
              target="_blank"
              rel="noopener noreferrer"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono transition-all hover:scale-105 ${
                isDark
                  ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300 hover:border-bb-accent/20'
                  : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700'
              }`}
            >
              GitHub
              <ArrowUpRight className="w-3 h-3" />
            </a>
            <a
              href="https://pypi.org/project/beyondbench/"
              target="_blank"
              rel="noopener noreferrer"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono transition-all hover:scale-105 ${
                isDark
                  ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300 hover:border-bb-accent/20'
                  : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700'
              }`}
            >
              PyPI
              <ArrowUpRight className="w-3 h-3" />
            </a>
          </div>
          </ScrollReveal>

          {/* Authors & Affiliations */}
          <ScrollReveal delay={600} direction="up">
          <div className={`mt-10 rounded-2xl p-6 sm:p-8 backdrop-blur-xl border transition-all duration-300 hover:shadow-lg ${
            isDark
              ? 'bg-bb-dark-300/40 border-bb-dark-50/20 hover:border-bb-accent/10'
              : 'bg-white/50 border-bb-light-300/40 hover:border-bb-accent-dark/10'
          }`}>
            <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-2">
              {paperInfo.authors.map((a, i) => (
                <span key={a.name} className="inline-flex items-center">
                  <span className={`text-base sm:text-lg font-semibold tracking-tight transition-colors duration-200 hover:${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} ${
                    isDark ? 'text-white' : 'text-gray-800'
                  }`}>{a.name}</span>
                  {i < paperInfo.authors.length - 1 && (
                    <span className={`ml-3 ${isDark ? 'text-gray-600' : 'text-gray-300'}`}>&bull;</span>
                  )}
                </span>
              ))}
            </div>
            <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
              <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-300 hover:scale-105 ${
                isDark
                  ? 'bg-gradient-to-r from-bb-accent/10 to-bb-teal/10 text-gray-200 border border-bb-accent/20'
                  : 'bg-gradient-to-r from-bb-accent-dark/8 to-bb-teal/8 text-gray-700 border border-bb-accent-dark/20'
              }`}>
                <span className={`w-2 h-2 rounded-full animate-pulse ${isDark ? 'bg-bb-accent' : 'bg-bb-accent-dark'}`} />
                Virginia Tech, USA
              </span>
              <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-300 hover:scale-105 ${
                isDark
                  ? 'bg-gradient-to-r from-bb-purple/10 to-bb-cyan/10 text-gray-200 border border-bb-purple/20'
                  : 'bg-gradient-to-r from-bb-purple/8 to-bb-cyan/8 text-gray-700 border border-bb-purple/15'
              }`}>
                <span className="w-2 h-2 rounded-full bg-bb-purple animate-pulse" />
                Amazon AGI, USA
              </span>
            </div>
          </div>
          </ScrollReveal>
        </div>
      </div>

      {/* ============ WAVE DIVIDER ============ */}
      <WaveDivider isDark={isDark} />

      {/* ============ STATS ============ */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
        <StatCard icon={Brain} label="Reasoning Tasks" value="44" rawValue={44} sub="Easy + Medium + Hard" delay={0} />
        <StatCard icon={Target} label="Task Variations" value="117" rawValue={117} sub="Fine-grained evaluation" delay={100} />
        <StatCard icon={BarChart3} label="Models Evaluated" value={`${modelData.length}`} rawValue={modelData.length} sub="Open & Proprietary" delay={200} />
        <StatCard icon={Zap} label="Unique Instances" value=">10^15" sub="Per task" delay={300} />
      </div>

      {/* ============ TOP 3 PODIUM ============ */}
      <ScrollReveal delay={0} direction="up">
      <div className="mb-12">
        <div className="flex items-center gap-3 mb-6">
          <div className={`flex items-center justify-center w-9 h-9 rounded-lg transition-all duration-300 ${
            isDark ? 'bg-bb-accent/10' : 'bg-bb-accent-dark/10'
          }`}>
            <Trophy className={`w-4.5 h-4.5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
          </div>
          <h2 className={`text-xl font-bold tracking-wide ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>
            Top <span className="gradient-text-green">Performers</span>
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {top3.map((m, i) => (
            <PodiumCard key={m.model} model={m} index={i} />
          ))}
        </div>
      </div>
      </ScrollReveal>

      {/* ============ SECTION DIVIDER ============ */}
      <div className="section-divider" />

      {/* ============ FILTERS & SEARCH ============ */}
      <ScrollReveal delay={0} direction="up">
      <div className={`rounded-xl p-4 mb-6 backdrop-blur-xl border transition-all ${
        isDark
          ? 'bg-bb-dark-300/60 border-bb-dark-50/30'
          : 'bg-white/70 border-bb-light-300/60 shadow-sm'
      }`}>
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1 group/search">
            <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 transition-colors duration-300 ${
              isDark ? 'text-gray-500 group-focus-within/search:text-bb-accent' : 'text-gray-400 group-focus-within/search:text-bb-accent-dark'
            }`} />
            <input
              type="text"
              placeholder="Search models..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className={`w-full pl-10 pr-4 py-2.5 rounded-lg text-sm transition-all duration-300 focus:outline-none focus:ring-2 ${
                isDark
                  ? 'bg-bb-dark-400/60 border border-bb-dark-50/20 text-gray-200 placeholder-gray-600 focus:border-bb-accent/40 focus:ring-bb-accent/10 focus:shadow-[0_0_15px_rgba(0,230,118,0.08)]'
                  : 'bg-bb-light-100 border border-bb-light-300 text-gray-900 placeholder-gray-400 focus:border-bb-accent-dark/40 focus:ring-bb-accent-dark/10'
              }`}
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 ${
              showFilters
                ? isDark
                  ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/30 shadow-[0_0_10px_rgba(0,230,118,0.1)]'
                  : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/30'
                : isDark
                  ? 'bg-bb-dark-400/60 border border-bb-dark-50/20 text-gray-400 hover:text-white hover:border-bb-accent/30'
                  : 'bg-bb-light-100 border border-bb-light-300 text-gray-500 hover:text-gray-700 hover:border-bb-accent-dark/30'
            }`}
          >
            <Filter className={`w-4 h-4 transition-transform duration-300 ${showFilters ? 'rotate-180' : ''}`} />
            Filters
            {familyFilter !== 'all' && (
              <span className={`ml-1 w-5 h-5 flex items-center justify-center rounded-full text-[10px] font-bold ${
                isDark ? 'bg-bb-accent/20 text-bb-accent' : 'bg-bb-accent-dark/20 text-bb-accent-dark'
              }`}>1</span>
            )}
          </button>
        </div>
        {/* Animated filter panel */}
        <div className={`overflow-hidden transition-all duration-300 ${showFilters ? 'max-h-96 opacity-100 mt-3' : 'max-h-0 opacity-0'}`}>
          <div className={`pt-3 border-t ${isDark ? 'border-bb-dark-50/10' : 'border-bb-light-300/50'}`}>
            <div className={`text-xs mb-2.5 font-medium tracking-wide ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Model Family</div>
            <div className="flex flex-wrap gap-2">
              {families.map(f => (
                <button
                  key={f}
                  onClick={() => setFamilyFilter(f)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${
                    familyFilter === f
                      ? isDark
                        ? 'bg-bb-accent/20 text-bb-accent border border-bb-accent/30 shadow-[0_0_10px_rgba(0,230,118,0.1)]'
                        : 'bg-bb-accent-dark/15 text-bb-accent-dark border border-bb-accent-dark/30'
                      : isDark
                        ? 'bg-bb-dark-400/40 text-gray-500 border border-bb-dark-50/10 hover:text-gray-300 hover:border-bb-dark-50/30'
                        : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700 hover:border-bb-light-400'
                  }`}
                >
                  {f === 'all' ? 'All Families' : f}
                  {f !== 'all' && (
                    <span className="ml-1 opacity-60">
                      ({modelData.filter(m => getFamily(m.model) === f).length})
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ============ LEADERBOARD TABLE ============ */}
      <div className={`rounded-xl overflow-hidden border transition-all ${
        isDark
          ? 'bg-bb-dark-300/40 border-bb-dark-50/30 neon-border'
          : 'bg-white/80 border-bb-light-300/60 shadow-lg'
      }`}>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className={`border-b ${
                isDark ? 'border-bb-dark-50/20 bg-bb-dark-400/40' : 'border-bb-light-300/50 bg-bb-light-100/80'
              }`}>
                <th className={`px-3 py-3.5 text-[11px] font-semibold uppercase tracking-wider text-center w-12 ${
                  isDark ? 'text-gray-500' : 'text-gray-400'
                }`}>#</th>
                <SortHeader k="model" className="text-left">Model</SortHeader>
                <SortHeader k="accuracy">Overall</SortHeader>
                <SortHeader k="easy_acc">Easy</SortHeader>
                <SortHeader k="medium_acc">Medium</SortHeader>
                <SortHeader k="hard_acc">Hard</SortHeader>
                <SortHeader k="instruction">IF Rate</SortHeader>
                <SortHeader k="tokens">Tokens</SortHeader>
                <SortHeader k="efficiency">Efficiency</SortHeader>
              </tr>
            </thead>
            <tbody>
              {sortedData.map((m, idx) => {
                const rank = globalRanked[m.model]
                const family = getFamily(m.model)
                const familyColor = FAMILY_COLORS[family] || '#6b7280'
                const isExpanded = expandedRow === m.model
                const isTop3 = rank <= 3

                return (
                  <>
                    <tr
                      key={m.model}
                      className={`border-b cursor-pointer transition-all duration-200 table-row-hover ${
                        isDark
                          ? `border-bb-dark-50/10 ${isExpanded ? 'bg-bb-dark-300/40' : isTop3 ? 'bg-bb-accent/[0.02] hover:bg-bb-dark-300/30' : 'hover:bg-bb-dark-300/20'}`
                          : `border-bb-light-300/30 ${isExpanded ? 'bg-bb-light-200/60' : isTop3 ? 'bg-bb-accent-dark/[0.03] hover:bg-bb-light-200/40' : 'hover:bg-bb-light-200/40'}`
                      }`}
                      onClick={() => setExpandedRow(isExpanded ? null : m.model)}
                      style={{ animationDelay: `${Math.min(idx * 20, 500)}ms` }}
                    >
                      <td className="px-3 py-3 text-center">
                        <div className={`inline-flex items-center justify-center w-7 h-7 rounded-lg text-xs font-bold font-mono ${
                          rank === 1
                            ? 'bg-yellow-400/15 text-yellow-400 shadow-[0_0_8px_rgba(250,204,21,0.2)]'
                            : rank === 2
                            ? isDark ? 'bg-gray-400/10 text-gray-300' : 'bg-gray-200 text-gray-500'
                            : rank === 3
                            ? 'bg-amber-600/10 text-amber-600'
                            : isDark ? 'text-gray-600' : 'text-gray-400'
                        }`}>
                          {rank}
                        </div>
                      </td>
                      <td className="px-3 py-3">
                        <div className="flex items-center gap-2.5">
                          <div
                            className="w-2.5 h-2.5 rounded-full shrink-0 ring-2 ring-offset-1 transition-transform duration-300 hover:scale-125"
                            style={{
                              backgroundColor: familyColor,
                              ringColor: familyColor + '40',
                              ringOffsetColor: isDark ? '#0d131b' : '#ffffff',
                            }}
                          />
                          <div className="min-w-0">
                            <div className={`text-sm font-medium truncate max-w-[200px] sm:max-w-none ${
                              isDark ? 'text-gray-200' : 'text-gray-800'
                            }`}>
                              {m.model.split('/')[1] || m.model}
                            </div>
                            <div className={`text-[10px] ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>
                              {family} {m.params && `\u00b7 ${m.params}`} {m.quantization !== 'None' && `\u00b7 ${m.quantization}`}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-3 text-center">
                        <div>
                          <span className={`font-mono text-sm font-semibold ${
                            isDark ? 'text-bb-accent' : 'text-bb-accent-dark'
                          }`}>{m.accuracy.toFixed(2)}%</span>
                          {/* Mini accuracy bar */}
                          <div className={`mini-progress ${isDark ? 'bg-bb-dark-500/60' : 'bg-bb-light-300/40'}`}>
                            <div
                              className={`mini-progress-fill ${isDark ? 'bg-bb-accent/50' : 'bg-bb-accent-dark/50'}`}
                              style={{ width: `${(m.accuracy / maxAccuracy) * 100}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className={`px-3 py-3 text-center font-mono text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{m.easy_acc.toFixed(1)}%</td>
                      <td className={`px-3 py-3 text-center font-mono text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{m.medium_acc.toFixed(1)}%</td>
                      <td className={`px-3 py-3 text-center font-mono text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{m.hard_acc.toFixed(1)}%</td>
                      <td className={`px-3 py-3 text-center font-mono text-sm ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>{m.instruction.toFixed(1)}%</td>
                      <td className={`px-3 py-3 text-center font-mono text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{m.tokens.toFixed(0)}</td>
                      <td className={`px-3 py-3 text-center font-mono text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{m.efficiency.toFixed(3)}</td>
                    </tr>
                    {isExpanded && (
                      <tr key={`${m.model}-detail`} className={
                        isDark ? 'bg-bb-dark-400/30' : 'bg-bb-light-200/40'
                      }>
                        <td colSpan={9} className="px-6 py-5">
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm animate-fade-in">
                            {[
                              { label: 'Easy Suite', acc: m.easy_acc, inst: m.easy_inst, tok: m.easy_tokens, color: isDark ? 'text-bb-accent' : 'text-bb-accent-dark' },
                              { label: 'Medium Suite', acc: m.medium_acc, inst: m.medium_inst, tok: m.medium_tokens, color: 'text-bb-teal' },
                              { label: 'Hard Suite', acc: m.hard_acc, inst: m.hard_inst, tok: m.hard_tokens, color: 'text-bb-mint' },
                              { label: 'Additional', extra: true },
                            ].map(({ label, acc, inst, tok, color, extra }) => (
                              <div key={label} className={`p-3 rounded-lg transition-all duration-300 hover:scale-[1.02] ${
                                isDark ? 'bg-bb-dark-300/40 hover:bg-bb-dark-300/60' : 'bg-white/60 hover:bg-white/80'
                              }`}>
                                <div className={`text-[10px] uppercase tracking-wider mb-2 font-semibold ${
                                  isDark ? 'text-gray-600' : 'text-gray-400'
                                }`}>{label}</div>
                                {extra ? (
                                  <>
                                    <div className={`font-mono text-xs ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
                                      Overthinking: {m.overthinking}
                                    </div>
                                    <div className={`font-mono text-xs mt-1 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                                      Words: {m.words?.toFixed(0)} &bull; Chars: {m.chars?.toFixed(0)}
                                    </div>
                                  </>
                                ) : (
                                  <>
                                    <div className={`${color} font-mono font-semibold`}>{acc.toFixed(2)}% acc</div>
                                    <div className={`font-mono text-xs mt-0.5 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                                      {inst.toFixed(1)}% IF &bull; {tok.toFixed(0)} tokens
                                    </div>
                                    {/* Progress bar for accuracy */}
                                    <div className={`mini-progress mt-2 ${isDark ? 'bg-bb-dark-500/60' : 'bg-bb-light-300/40'}`}>
                                      <div
                                        className={`mini-progress-fill ${color === 'text-bb-teal' ? 'bg-bb-teal/50' : color === 'text-bb-mint' ? 'bg-bb-mint/50' : isDark ? 'bg-bb-accent/50' : 'bg-bb-accent-dark/50'}`}
                                        style={{ width: `${acc}%` }}
                                      />
                                    </div>
                                  </>
                                )}
                              </div>
                            ))}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
        <div className={`px-4 py-3 border-t text-xs text-center ${
          isDark ? 'border-bb-dark-50/10 text-gray-600' : 'border-bb-light-300/30 text-gray-400'
        }`}>
          Showing {sortedData.length} of {modelData.length} models &bull; Click a row for detailed metrics
        </div>
      </div>
      </ScrollReveal>
    </div>
  )
}

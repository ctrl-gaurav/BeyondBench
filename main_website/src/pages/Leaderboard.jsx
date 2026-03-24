import { useState, useMemo, useEffect, useRef } from 'react'
import { Trophy, Search, ChevronDown, ChevronUp, Filter, BarChart3, Zap, Brain, Target, Hexagon, TrendingUp, Award, Crown, Medal, Star, ArrowUpRight, ExternalLink, Check, Copy } from 'lucide-react'
import { modelData, paperInfo } from '../data/modelData'
import { useTheme } from '../context/ThemeContext'

function useStaggeredReveal(count, baseDelay = 100) {
  const [visible, setVisible] = useState([])
  useEffect(() => {
    const timers = []
    for (let i = 0; i < count; i++) {
      timers.push(setTimeout(() => setVisible(v => [...v, i]), baseDelay * (i + 1)))
    }
    return () => timers.forEach(clearTimeout)
  }, [count, baseDelay])
  return visible
}

function AnimateIn({ children, delay = 0, className = '', direction = 'up' }) {
  const ref = useRef(null)
  const [show, setShow] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setShow(true), delay)
    return () => clearTimeout(timer)
  }, [delay])

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
        show ? 'opacity-100 translate-y-0 translate-x-0 scale-100' : `opacity-0 ${transforms[direction]}`
      } ${className}`}
    >
      {children}
    </div>
  )
}

function TypingEffect() {
  const phrases = [
    'Benchmark-Free Reasoning Evaluation',
    '44 Tasks, 117 Variations, 10^15 Instances',
    'Contamination-Resistant by Design',
    '5 Backends: OpenAI, Gemini, Anthropic, vLLM, HF',
    'Dynamic Algorithmic Problem Generation',
  ]
  const [phraseIdx, setPhraseIdx] = useState(0)
  const [charIdx, setCharIdx] = useState(0)
  const [isDeleting, setIsDeleting] = useState(false)

  useEffect(() => {
    const cur = phrases[phraseIdx]
    let timeout
    if (!isDeleting && charIdx < cur.length) {
      timeout = setTimeout(() => setCharIdx(charIdx + 1), 45)
    } else if (!isDeleting && charIdx === cur.length) {
      timeout = setTimeout(() => setIsDeleting(true), 2200)
    } else if (isDeleting && charIdx > 0) {
      timeout = setTimeout(() => setCharIdx(charIdx - 1), 25)
    } else if (isDeleting && charIdx === 0) {
      setIsDeleting(false)
      setPhraseIdx((phraseIdx + 1) % phrases.length)
    }
    return () => clearTimeout(timeout)
  }, [charIdx, isDeleting, phraseIdx])

  return (
    <span className="typing-cursor font-mono text-bb-accent">
      {phrases[phraseIdx].substring(0, charIdx)}
    </span>
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

function StatCard({ icon: Icon, label, value, sub, delay = 0 }) {
  const { isDark } = useTheme()
  return (
    <div
      className={`group relative overflow-hidden rounded-xl p-5 flex flex-col items-center text-center transition-all duration-300 hover:scale-[1.03] hover:-translate-y-1 ${
        isDark
          ? 'bg-bb-dark-300/60 backdrop-blur-xl border border-bb-dark-50/30 hover:border-bb-accent/30'
          : 'bg-white/70 backdrop-blur-xl border border-bb-light-300/60 hover:border-bb-accent-dark/40 shadow-sm hover:shadow-lg'
      }`}
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Shimmer on hover */}
      <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 ${
        isDark
          ? 'bg-gradient-to-br from-bb-accent/5 via-transparent to-bb-teal/5'
          : 'bg-gradient-to-br from-bb-accent-dark/5 via-transparent to-bb-teal/5'
      }`} />

      <div className={`relative w-11 h-11 rounded-xl flex items-center justify-center mb-3 transition-all duration-300 group-hover:scale-110 ${
        isDark
          ? 'bg-bb-accent/10 group-hover:bg-bb-accent/20'
          : 'bg-bb-accent-dark/10 group-hover:bg-bb-accent-dark/15'
      }`}>
        <Icon className={`w-5 h-5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
      </div>
      <div className={`text-2xl font-bold font-mono relative ${isDark ? 'text-white' : 'text-gray-900'}`}>{value}</div>
      <div className={`text-xs mt-1.5 font-medium ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
      {sub && <div className={`text-[10px] mt-0.5 ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{sub}</div>}
    </div>
  )
}

function PodiumCard({ model, index }) {
  const { isDark } = useTheme()
  const configs = [
    {
      medal: <Crown className="w-6 h-6" />,
      color: 'text-yellow-400',
      bg: isDark ? 'from-yellow-400/10 via-yellow-400/5 to-transparent' : 'from-yellow-400/15 via-yellow-400/5 to-transparent',
      border: isDark ? 'border-yellow-400/30 hover:border-yellow-400/50' : 'border-yellow-400/40 hover:border-yellow-400/60',
      glow: 'shadow-[0_0_30px_rgba(250,204,21,0.1)]',
      label: '1st Place'
    },
    {
      medal: <Medal className="w-5 h-5" />,
      color: 'text-gray-300',
      bg: isDark ? 'from-gray-300/10 via-gray-300/5 to-transparent' : 'from-gray-400/10 via-gray-400/5 to-transparent',
      border: isDark ? 'border-gray-400/20 hover:border-gray-400/40' : 'border-gray-400/30 hover:border-gray-400/50',
      glow: '',
      label: '2nd Place'
    },
    {
      medal: <Star className="w-5 h-5" />,
      color: 'text-amber-600',
      bg: isDark ? 'from-amber-600/10 via-amber-600/5 to-transparent' : 'from-amber-600/10 via-amber-600/5 to-transparent',
      border: isDark ? 'border-amber-600/20 hover:border-amber-600/40' : 'border-amber-600/30 hover:border-amber-600/50',
      glow: '',
      label: '3rd Place'
    },
  ]
  const c = configs[index]
  const m = model

  return (
    <div className={`group relative overflow-hidden rounded-xl border p-5 transition-all duration-300 hover:scale-[1.02] hover:-translate-y-1 ${c.border} ${c.glow} ${
      isDark ? 'bg-bb-dark-300/60 backdrop-blur-xl' : 'bg-white/80 backdrop-blur-xl shadow-sm hover:shadow-lg'
    }`}>
      {/* Background gradient */}
      <div className={`absolute inset-0 bg-gradient-to-br ${c.bg} opacity-60`} />
      {/* Corner decoration */}
      <div className={`absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl ${c.bg} rounded-bl-full opacity-40`} />

      <div className="relative">
        <div className="flex items-start gap-3">
          <div className={`flex items-center justify-center w-10 h-10 rounded-lg ${
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
            <div key={label} className={`text-center p-2 rounded-lg ${
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
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function Leaderboard() {
  const { isDark } = useTheme()
  const [copied, setCopied] = useState(false)
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
        className={`px-3 py-3.5 text-[11px] font-semibold uppercase tracking-wider cursor-pointer select-none transition-colors ${
          active
            ? isDark ? 'text-bb-accent' : 'text-bb-accent-dark'
            : isDark ? 'text-gray-500 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600'
        } ${className}`}
        onClick={() => toggleSort(k)}
      >
        <div className="flex items-center gap-1">
          {children}
          {active && (sortDir === 'desc' ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />)}
        </div>
      </th>
    )
  }

  const top3 = useMemo(() => {
    return [...modelData].sort((a, b) => b.accuracy - a.accuracy).slice(0, 3)
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <div className="text-center mb-8 relative">
        {/* Radial glow behind hero */}
        <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] rounded-full blur-3xl pointer-events-none ${
          isDark ? 'bg-bb-accent/[0.04]' : 'bg-bb-accent-dark/[0.03]'
        }`} />

        <div className="relative pt-8 pb-4">
          <AnimateIn delay={0} direction="scale">
            <div className="flex items-center justify-center gap-3 mb-5">
              <div className="relative">
                <Hexagon className={`w-12 h-12 animate-float ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
                <div className={`absolute inset-0 w-12 h-12 rounded-full blur-2xl animate-glow-pulse ${
                  isDark ? 'bg-bb-accent/30' : 'bg-bb-accent-dark/20'
                }`} />
              </div>
            </div>
          </AnimateIn>

          <AnimateIn delay={150} direction="up">
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold mb-4 tracking-tight">
              <span className={isDark ? 'text-white' : 'text-gray-900'}>Beyond</span>
              <span className={`hero-gradient-text`}>Bench</span>
            </h1>
          </AnimateIn>

          <AnimateIn delay={300} direction="up">
            <p className={`text-lg sm:text-xl max-w-2xl mx-auto mb-3 font-light ${
              isDark ? 'text-gray-400' : 'text-gray-600'
            }`}>
              Contamination-Resistant Evaluation of Reasoning in Language Models
            </p>
          </AnimateIn>

          {/* Typing ticker */}
          <AnimateIn delay={400} direction="up">
            <div className={`h-8 flex items-center justify-center mb-5 text-sm ${
              isDark ? 'text-gray-500' : 'text-gray-400'
            }`}>
              <span className="mr-2">&gt;</span>
              <TypingEffect />
            </div>
          </AnimateIn>

          {/* Badges */}
          <AnimateIn delay={450} direction="up">
          <div className="flex flex-wrap items-center justify-center gap-2 text-xs">
            <a
              href="https://openreview.net/forum?id=mIKqVWGjwI"
              target="_blank"
              rel="noopener noreferrer"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full font-semibold transition-all hover:scale-105 ${
                isDark
                  ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/20 hover:bg-bb-accent/20'
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
                  ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300'
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
                  ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300'
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
                  ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300'
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
                  ? 'bg-bb-dark-300/60 text-gray-400 border border-bb-dark-50/20 hover:text-gray-300'
                  : 'bg-bb-light-200 text-gray-500 border border-bb-light-300 hover:text-gray-700'
              }`}
            >
              PyPI
              <ArrowUpRight className="w-3 h-3" />
            </a>
          </div>
          </AnimateIn>

          {/* Install Command */}
          <AnimateIn delay={550} direction="up">
            <div className="max-w-md mx-auto mt-6 mb-2">
              <div className={`group relative rounded-xl overflow-hidden transition-all duration-500 ${
                isDark
                  ? 'bg-[#0a0e18] border border-bb-dark-50/20 hover:border-bb-accent/30 shadow-xl shadow-black/30'
                  : 'bg-[#f7f8fc] border border-gray-200/60 hover:border-bb-accent-dark/30 shadow-md'
              }`}>
                {/* Header bar */}
                <div className={`flex items-center justify-between px-4 py-2 border-b ${
                  isDark ? 'border-bb-dark-50/10 bg-white/[0.02]' : 'border-gray-200/30 bg-gray-50/50'
                }`}>
                  <div className="flex items-center gap-2.5">
                    <div className="flex gap-1.5">
                      <div className="w-2 h-2 rounded-full bg-[#ff5f57]/80" />
                      <div className="w-2 h-2 rounded-full bg-[#febc2e]/80" />
                      <div className="w-2 h-2 rounded-full bg-[#28c840]/80" />
                    </div>
                    <span className={`text-[9px] uppercase tracking-widest font-mono ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>terminal</span>
                  </div>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText('pip install beyondbench')
                      setCopied(true)
                      setTimeout(() => setCopied(false), 2000)
                    }}
                    className={`text-[10px] font-mono px-2 py-0.5 rounded transition-all duration-200 flex items-center gap-1 ${
                      copied
                        ? isDark ? 'text-bb-accent bg-bb-accent/10' : 'text-bb-accent-dark bg-bb-accent-dark/10'
                        : isDark ? 'text-gray-600 hover:text-bb-accent hover:bg-bb-accent/10' : 'text-gray-400 hover:text-bb-accent-dark hover:bg-bb-accent-dark/10'
                    }`}
                  >
                    {copied ? <><Check className="w-3 h-3" /> Copied!</> : <><Copy className="w-3 h-3" /> Copy</>}
                  </button>
                </div>

                {/* Code content */}
                <div className="px-4 py-3">
                  <code className="text-sm font-mono">
                    <span className={isDark ? 'text-gray-500' : 'text-gray-400'}>$</span>
                    {' '}
                    <span style={{ color: isDark ? '#c084fc' : '#7c3aed' }}>pip</span>
                    {' '}
                    <span style={{ color: isDark ? '#60a5fa' : '#2563eb' }}>install</span>
                    {' '}
                    <span style={{ color: isDark ? '#34d399' : '#059669' }}>beyondbench</span>
                    <span className={`inline-block w-[8px] h-[16px] ml-0.5 align-middle rounded-[1px] ${isDark ? 'bg-bb-accent' : 'bg-bb-accent-dark'}`} style={{ animation: 'blink 1s step-end infinite' }} />
                  </code>
                </div>
              </div>
            </div>
          </AnimateIn>

          {/* Authors & Affiliations */}
          <AnimateIn delay={650} direction="up">
          <div className={`mt-6 rounded-2xl p-4 sm:p-6 backdrop-blur-xl border ${
            isDark
              ? 'bg-bb-dark-300/40 border-bb-dark-50/20'
              : 'bg-white/50 border-bb-light-300/40'
          }`}>
            <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1.5">
              {paperInfo.authors.map((a, i) => (
                <span key={a.name} className="inline-flex items-center">
                  <span className={`text-base sm:text-lg font-semibold tracking-tight ${
                    isDark ? 'text-white' : 'text-gray-800'
                  }`}>{a.name}</span>
                  {i < paperInfo.authors.length - 1 && (
                    <span className={`ml-3 ${isDark ? 'text-gray-600' : 'text-gray-300'}`}>&bull;</span>
                  )}
                </span>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-center gap-2">
              <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium ${
                isDark
                  ? 'bg-gradient-to-r from-bb-accent/10 to-bb-teal/10 text-gray-200 border border-bb-accent/20'
                  : 'bg-gradient-to-r from-bb-accent-dark/8 to-bb-teal/8 text-gray-700 border border-bb-accent-dark/20'
              }`}>
                <span className={`w-2 h-2 rounded-full ${isDark ? 'bg-bb-accent' : 'bg-bb-accent-dark'}`} />
                Virginia Tech, USA
              </span>
              <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium ${
                isDark
                  ? 'bg-gradient-to-r from-bb-purple/10 to-bb-cyan/10 text-gray-200 border border-bb-purple/20'
                  : 'bg-gradient-to-r from-bb-purple/8 to-bb-cyan/8 text-gray-700 border border-bb-purple/15'
              }`}>
                <span className="w-2 h-2 rounded-full bg-bb-purple" />
                Amazon AGI, USA
              </span>
            </div>
          </div>
          </AnimateIn>
        </div>
      </div>

      {/* Stats */}
      <AnimateIn delay={700} direction="up">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
        <StatCard icon={Brain} label="Reasoning Tasks" value="44" sub="Easy + Medium + Hard" delay={0} />
        <StatCard icon={Target} label="Task Variations" value="117" sub="Fine-grained evaluation" delay={100} />
        <StatCard icon={BarChart3} label="Models Evaluated" value={`${modelData.length}`} sub="Open & Proprietary" delay={200} />
        <StatCard icon={Zap} label="Unique Instances" value=">10^15" sub="Per task" delay={300} />
      </div>
      </AnimateIn>

      {/* Top 3 Podium */}
      <AnimateIn delay={900} direction="up">
      <div className="mb-12">
        <div className="flex items-center gap-3 mb-5">
          <div className={`flex items-center justify-center w-8 h-8 rounded-lg ${
            isDark ? 'bg-bb-accent/10' : 'bg-bb-accent-dark/10'
          }`}>
            <Trophy className={`w-4 h-4 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
          </div>
          <h2 className={`text-lg font-semibold ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>
            Top Performers
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {top3.map((m, i) => (
            <PodiumCard key={m.model} model={m} index={i} />
          ))}
        </div>
      </div>
      </AnimateIn>

      {/* Filters & Search */}
      <AnimateIn delay={1100} direction="up">
      <div className={`rounded-xl p-4 mb-6 backdrop-blur-xl border transition-all ${
        isDark
          ? 'bg-bb-dark-300/60 border-bb-dark-50/30'
          : 'bg-white/70 border-bb-light-300/60 shadow-sm'
      }`}>
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 ${
              isDark ? 'text-gray-500' : 'text-gray-400'
            }`} />
            <input
              type="text"
              placeholder="Search models..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className={`w-full pl-10 pr-4 py-2.5 rounded-lg text-sm transition-all duration-200 focus:outline-none focus:ring-2 ${
                isDark
                  ? 'bg-bb-dark-400/60 border border-bb-dark-50/20 text-gray-200 placeholder-gray-600 focus:border-bb-accent/40 focus:ring-bb-accent/10'
                  : 'bg-bb-light-100 border border-bb-light-300 text-gray-900 placeholder-gray-400 focus:border-bb-accent-dark/40 focus:ring-bb-accent-dark/10'
              }`}
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
              showFilters
                ? isDark
                  ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/30'
                  : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/30'
                : isDark
                  ? 'bg-bb-dark-400/60 border border-bb-dark-50/20 text-gray-400 hover:text-white hover:border-bb-accent/30'
                  : 'bg-bb-light-100 border border-bb-light-300 text-gray-500 hover:text-gray-700 hover:border-bb-accent-dark/30'
            }`}
          >
            <Filter className="w-4 h-4" />
            Filters
            {familyFilter !== 'all' && (
              <span className={`ml-1 w-5 h-5 flex items-center justify-center rounded-full text-[10px] font-bold ${
                isDark ? 'bg-bb-accent/20 text-bb-accent' : 'bg-bb-accent-dark/20 text-bb-accent-dark'
              }`}>1</span>
            )}
          </button>
        </div>
        {showFilters && (
          <div className={`mt-3 pt-3 border-t ${isDark ? 'border-bb-dark-50/10' : 'border-bb-light-300/50'}`}>
            <div className={`text-xs mb-2.5 font-medium ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Model Family</div>
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
        )}
      </div>

      {/* Leaderboard Table */}
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
              {sortedData.map((m) => {
                const rank = globalRanked[m.model]
                const family = getFamily(m.model)
                const familyColor = FAMILY_COLORS[family] || '#6b7280'
                const isExpanded = expandedRow === m.model
                const isTop3 = rank <= 3

                return (
                  <>
                    <tr
                      key={m.model}
                      className={`border-b cursor-pointer transition-all duration-150 ${
                        isDark
                          ? `border-bb-dark-50/10 ${isExpanded ? 'bg-bb-dark-300/40' : isTop3 ? 'bg-bb-accent/[0.02] hover:bg-bb-dark-300/30' : 'hover:bg-bb-dark-300/20'}`
                          : `border-bb-light-300/30 ${isExpanded ? 'bg-bb-light-200/60' : isTop3 ? 'bg-bb-accent-dark/[0.03] hover:bg-bb-light-200/40' : 'hover:bg-bb-light-200/40'}`
                      }`}
                      onClick={() => setExpandedRow(isExpanded ? null : m.model)}
                    >
                      <td className="px-3 py-3 text-center">
                        <span className={`font-mono text-sm font-bold ${
                          rank === 1 ? 'text-yellow-400' : rank === 2 ? 'text-gray-300' : rank === 3 ? 'text-amber-600'
                            : isDark ? 'text-gray-600' : 'text-gray-400'
                        }`}>
                          {rank}
                        </span>
                      </td>
                      <td className="px-3 py-3">
                        <div className="flex items-center gap-2.5">
                          <div
                            className="w-2.5 h-2.5 rounded-full shrink-0 ring-2 ring-offset-1"
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
                        <span className={`font-mono text-sm font-semibold ${
                          isDark ? 'text-bb-accent' : 'text-bb-accent-dark'
                        }`}>{m.accuracy.toFixed(2)}%</span>
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
                              <div key={label} className={`p-3 rounded-lg ${
                                isDark ? 'bg-bb-dark-300/40' : 'bg-white/60'
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
      </AnimateIn>
    </div>
  )
}

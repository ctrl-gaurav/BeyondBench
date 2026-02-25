import { useState, useMemo } from 'react'
import { Trophy, Search, ArrowUpDown, ChevronDown, ChevronUp, Filter, BarChart3, Zap, Brain, Target, Hexagon, TrendingUp, Award } from 'lucide-react'
import { modelData, paperInfo } from '../data/modelData'

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

function StatCard({ icon: Icon, label, value, sub, color = 'bb-accent' }) {
  return (
    <div className="glass-card p-5 flex flex-col items-center text-center group hover:border-bb-accent/30 transition-all">
      <div className={`w-10 h-10 rounded-lg bg-${color}/10 flex items-center justify-center mb-3`}>
        <Icon className={`w-5 h-5 text-${color}`} />
      </div>
      <div className="text-2xl font-bold text-white font-mono">{value}</div>
      <div className="text-xs text-gray-500 mt-1">{label}</div>
      {sub && <div className="text-[10px] text-gray-600 mt-0.5">{sub}</div>}
    </div>
  )
}

export default function Leaderboard() {
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
        className={`px-3 py-3 text-xs font-semibold uppercase tracking-wider cursor-pointer select-none transition-colors ${active ? 'text-bb-accent' : 'text-gray-500 hover:text-gray-300'} ${className}`}
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
      <div className="text-center mb-12 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-bb-accent/5 to-transparent rounded-3xl -z-10" />
        <div className="pt-8 pb-4">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Hexagon className="w-10 h-10 text-bb-accent animate-float" />
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold mb-3">
            <span className="text-white">Beyond</span>
            <span className="text-bb-accent neon-text">Bench</span>
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto mb-2">
            Contamination-Resistant Evaluation of Reasoning in Language Models
          </p>
          <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
            <span className="px-2 py-1 bg-bb-accent/10 text-bb-accent rounded-full font-semibold">Accepted at ICLR 2026</span>
            <span className="px-2 py-1 bg-bb-dark-300/60 text-gray-400 rounded-full">arXiv:2509.24210</span>
          </div>

          {/* Authors */}
          <div className="mt-6 text-sm text-gray-500">
            {paperInfo.authors.map((a, i) => (
              <span key={a.name}>
                <span className={a.corresponding ? 'text-gray-300' : ''}>{a.name}{a.corresponding ? '*' : ''}</span>
                {i < paperInfo.authors.length - 1 ? ', ' : ''}
              </span>
            ))}
          </div>
          <div className="mt-1 text-xs text-gray-600">
            Virginia Tech &bull; Amazon AGI
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        <StatCard icon={Brain} label="Reasoning Tasks" value="44" sub="Easy + Medium + Hard" />
        <StatCard icon={Target} label="Task Variations" value="117" sub="Fine-grained evaluation" />
        <StatCard icon={BarChart3} label="Models Evaluated" value={`${modelData.length}`} sub="Open & Proprietary" />
        <StatCard icon={Zap} label="Unique Instances" value=">10^15" sub="Per task" />
      </div>

      {/* Top 3 Podium */}
      <div className="mb-10">
        <h2 className="text-lg font-semibold text-gray-300 mb-4 flex items-center gap-2">
          <Trophy className="w-5 h-5 text-bb-accent" />
          Top Performers
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {top3.map((m, i) => {
            const medals = ['text-yellow-400', 'text-gray-300', 'text-amber-600']
            const borders = ['border-yellow-400/30', 'border-gray-400/20', 'border-amber-600/20']
            return (
              <div key={m.model} className={`glass-card p-5 border ${borders[i]} relative overflow-hidden`}>
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-bl from-bb-accent/5 to-transparent rounded-bl-full" />
                <div className="flex items-start gap-3">
                  <div className={`text-2xl font-bold ${medals[i]} font-mono`}>#{i + 1}</div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-white truncate">{m.model.split('/')[1] || m.model}</div>
                    <div className="text-xs text-gray-500">{getFamily(m.model)}</div>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-3 gap-2">
                  <div>
                    <div className="text-lg font-bold text-bb-accent font-mono">{m.accuracy.toFixed(1)}%</div>
                    <div className="text-[10px] text-gray-600">Overall</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-bb-teal font-mono">{m.easy_acc.toFixed(1)}%</div>
                    <div className="text-[10px] text-gray-600">Easy</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold text-bb-mint font-mono">{m.hard_acc.toFixed(1)}%</div>
                    <div className="text-[10px] text-gray-600">Hard</div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Filters & Search */}
      <div className="glass-card p-4 mb-6">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              type="text"
              placeholder="Search models..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-bb-dark-400/60 border border-bb-dark-50/20 rounded-lg text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-bb-accent/40 transition-colors"
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-2 px-4 py-2.5 bg-bb-dark-400/60 border border-bb-dark-50/20 rounded-lg text-sm text-gray-400 hover:text-white hover:border-bb-accent/30 transition-all"
          >
            <Filter className="w-4 h-4" />
            Filters
          </button>
        </div>
        {showFilters && (
          <div className="mt-3 pt-3 border-t border-bb-dark-50/10">
            <div className="text-xs text-gray-500 mb-2">Model Family</div>
            <div className="flex flex-wrap gap-2">
              {families.map(f => (
                <button
                  key={f}
                  onClick={() => setFamilyFilter(f)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                    familyFilter === f
                      ? 'bg-bb-accent/20 text-bb-accent border border-bb-accent/30'
                      : 'bg-bb-dark-400/40 text-gray-500 border border-bb-dark-50/10 hover:text-gray-300'
                  }`}
                >
                  {f === 'all' ? 'All Families' : f}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Leaderboard Table */}
      <div className="glass-card overflow-hidden neon-border">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-bb-dark-50/20 bg-bb-dark-400/30">
                <th className="px-3 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wider text-center w-12">#</th>
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
                return (
                  <>
                    <tr
                      key={m.model}
                      className={`border-b border-bb-dark-50/10 cursor-pointer transition-colors ${
                        isExpanded ? 'bg-bb-dark-300/40' : 'hover:bg-bb-dark-300/20'
                      } ${rank <= 3 ? 'bg-bb-accent/[0.02]' : ''}`}
                      onClick={() => setExpandedRow(isExpanded ? null : m.model)}
                    >
                      <td className="px-3 py-3 text-center">
                        <span className={`font-mono text-sm font-bold ${rank === 1 ? 'text-yellow-400' : rank === 2 ? 'text-gray-300' : rank === 3 ? 'text-amber-600' : 'text-gray-600'}`}>
                          {rank}
                        </span>
                      </td>
                      <td className="px-3 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: familyColor }} />
                          <div>
                            <div className="text-sm font-medium text-gray-200 truncate max-w-[200px] sm:max-w-none">{m.model.split('/')[1] || m.model}</div>
                            <div className="text-[10px] text-gray-600">{family} {m.params && `| ${m.params}`} {m.quantization !== 'None' && `| ${m.quantization}`}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-3 text-center">
                        <span className="font-mono text-sm font-semibold text-bb-accent">{m.accuracy.toFixed(2)}%</span>
                      </td>
                      <td className="px-3 py-3 text-center font-mono text-sm text-gray-300">{m.easy_acc.toFixed(1)}%</td>
                      <td className="px-3 py-3 text-center font-mono text-sm text-gray-300">{m.medium_acc.toFixed(1)}%</td>
                      <td className="px-3 py-3 text-center font-mono text-sm text-gray-300">{m.hard_acc.toFixed(1)}%</td>
                      <td className="px-3 py-3 text-center font-mono text-sm text-gray-400">{m.instruction.toFixed(1)}%</td>
                      <td className="px-3 py-3 text-center font-mono text-xs text-gray-500">{m.tokens.toFixed(0)}</td>
                      <td className="px-3 py-3 text-center font-mono text-xs text-gray-500">{m.efficiency.toFixed(3)}</td>
                    </tr>
                    {isExpanded && (
                      <tr key={`${m.model}-detail`} className="bg-bb-dark-400/20">
                        <td colSpan={9} className="px-6 py-4">
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">Easy Suite</div>
                              <div className="text-bb-accent font-mono">{m.easy_acc.toFixed(2)}% acc</div>
                              <div className="text-gray-500 font-mono text-xs">{m.easy_inst.toFixed(1)}% IF | {m.easy_tokens.toFixed(0)} tokens</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">Medium Suite</div>
                              <div className="text-bb-teal font-mono">{m.medium_acc.toFixed(2)}% acc</div>
                              <div className="text-gray-500 font-mono text-xs">{m.medium_inst.toFixed(1)}% IF | {m.medium_tokens.toFixed(0)} tokens</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">Hard Suite</div>
                              <div className="text-bb-mint font-mono">{m.hard_acc.toFixed(2)}% acc</div>
                              <div className="text-gray-500 font-mono text-xs">{m.hard_inst.toFixed(1)}% IF | {m.hard_tokens.toFixed(0)} tokens</div>
                            </div>
                            <div>
                              <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">Additional Metrics</div>
                              <div className="text-gray-400 font-mono text-xs">Overthinking: {m.overthinking}</div>
                              <div className="text-gray-500 font-mono text-xs">Words: {m.words?.toFixed(0)} | Chars: {m.chars?.toFixed(0)}</div>
                            </div>
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
        <div className="px-4 py-3 border-t border-bb-dark-50/10 text-xs text-gray-600 text-center">
          Showing {sortedData.length} of {modelData.length} models &bull; Click a row for detailed metrics
        </div>
      </div>
    </div>
  )
}

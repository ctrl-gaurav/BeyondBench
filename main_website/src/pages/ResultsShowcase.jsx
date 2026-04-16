import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ScatterChart, Scatter, ZAxis,
} from 'recharts'
import { BarChart3, Target, Zap, TrendingUp, ChevronDown, X, Plus } from 'lucide-react'
import { modelData } from '../data/modelData'
import { useTheme } from '../context/ThemeContext'

// Color palette for selected models
const MODEL_COLORS = [
  '#00e676', // bb-accent green
  '#0ea5e9', // bb-teal blue
  '#a855f7', // bb-purple
  '#06b6d4', // bb-cyan
  '#f59e0b', // amber
  '#ef4444', // red
]

// Shorten model name for display
function shortName(name) {
  return name.replace(/^[^/]+\//, '').replace(/-(?:Instruct|instruct|chat|Chat)$/, '')
}

// Custom tooltip for bar chart
function BarTooltip({ active, payload, label }) {
  const { isDark } = useTheme()
  if (!active || !payload?.length) return null
  return (
    <div className={`px-3 py-2 rounded-lg border text-xs shadow-xl ${
      isDark
        ? 'bg-bb-dark-300/95 border-bb-dark-50/30 text-white'
        : 'bg-white/95 border-bb-light-300 text-gray-900'
    }`}>
      <p className="font-semibold mb-1 truncate max-w-[200px]">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.fill || p.color }}>
          {p.name}: <span className="font-mono font-bold">{p.value?.toFixed(1)}%</span>
        </p>
      ))}
    </div>
  )
}

// Custom tooltip for scatter chart
function ScatterTooltip({ active, payload }) {
  const { isDark } = useTheme()
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className={`px-3 py-2 rounded-lg border text-xs shadow-xl max-w-[220px] ${
      isDark
        ? 'bg-bb-dark-300/95 border-bb-dark-50/30 text-white'
        : 'bg-white/95 border-bb-light-300 text-gray-900'
    }`}>
      <p className="font-semibold truncate">{shortName(d.model)}</p>
      <p className={isDark ? 'text-gray-400' : 'text-gray-500'}>Accuracy: <span className="font-mono text-white font-bold">{d.accuracy?.toFixed(1)}%</span></p>
      <p className={isDark ? 'text-gray-400' : 'text-gray-500'}>Tokens: <span className="font-mono text-white font-bold">{d.tokens?.toFixed(0)}</span></p>
    </div>
  )
}

// Custom tooltip for radar chart
function RadarTooltip({ active, payload }) {
  const { isDark } = useTheme()
  if (!active || !payload?.length) return null
  return (
    <div className={`px-3 py-2 rounded-lg border text-xs shadow-xl ${
      isDark
        ? 'bg-bb-dark-300/95 border-bb-dark-50/30 text-white'
        : 'bg-white/95 border-bb-light-300 text-gray-900'
    }`}>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: <span className="font-mono font-bold">{p.value?.toFixed(1)}%</span>
        </p>
      ))}
    </div>
  )
}

function SectionHeader({ icon: Icon, title, subtitle, isDark }) {
  return (
    <div className="flex items-start gap-4 mb-6">
      <div className={`p-3 rounded-xl flex-shrink-0 ${isDark ? 'bg-bb-accent/10' : 'bg-bb-accent-dark/10'}`}>
        <Icon className={`w-6 h-6 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
      </div>
      <div>
        <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>{title}</h2>
        <p className={`text-sm mt-0.5 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>{subtitle}</p>
      </div>
    </div>
  )
}

function Card({ children, isDark, className = '' }) {
  return (
    <div className={`rounded-2xl border p-6 ${
      isDark
        ? 'bg-bb-dark-300/60 backdrop-blur-xl border-bb-dark-50/30'
        : 'bg-white/70 backdrop-blur-xl border-bb-light-300/60 shadow-sm'
    } ${className}`}>
      {children}
    </div>
  )
}

export default function ResultsShowcase() {
  const { isDark } = useTheme()

  // Top 20 models by accuracy for bar chart
  const top20 = useMemo(() => {
    return [...modelData]
      .sort((a, b) => b.accuracy - a.accuracy)
      .slice(0, 20)
      .map(m => ({ ...m, shortModel: shortName(m.model) }))
  }, [])

  // All models for scatter
  const scatterData = useMemo(() => modelData.map(m => ({ ...m, shortModel: shortName(m.model) })), [])

  // Model selector state
  const [selectedModels, setSelectedModels] = useState(() => {
    // Default: top 4 models by accuracy
    return [...modelData]
      .sort((a, b) => b.accuracy - a.accuracy)
      .slice(0, 4)
      .map(m => m.model)
  })
  const [selectorOpen, setSelectorOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  const filteredOptions = useMemo(() => {
    const q = searchQuery.toLowerCase()
    return modelData.filter(m => m.model.toLowerCase().includes(q)).slice(0, 30)
  }, [searchQuery])

  const addModel = (modelName) => {
    if (selectedModels.length >= 5) return
    if (!selectedModels.includes(modelName)) {
      setSelectedModels(prev => [...prev, modelName])
    }
    setSelectorOpen(false)
    setSearchQuery('')
  }

  const removeModel = (modelName) => {
    setSelectedModels(prev => prev.filter(m => m !== modelName))
  }

  // Radar data for selected models
  const radarData = useMemo(() => {
    const metrics = [
      { key: 'easy_acc', label: 'Easy Acc' },
      { key: 'medium_acc', label: 'Medium Acc' },
      { key: 'hard_acc', label: 'Hard Acc' },
      { key: 'easy_inst', label: 'Easy IF' },
      { key: 'hard_inst', label: 'Hard IF' },
      { key: 'accuracy', label: 'Overall' },
    ]
    return metrics.map(({ key, label }) => {
      const row = { metric: label }
      selectedModels.forEach(modelName => {
        const m = modelData.find(d => d.model === modelName)
        if (m) row[shortName(modelName)] = +(m[key] || 0).toFixed(1)
      })
      return row
    })
  }, [selectedModels])

  // Axis tick style
  const axisStyle = { fill: isDark ? '#9ca3af' : '#6b7280', fontSize: 10 }
  const gridColor = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.08)'

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Page header */}
      <div className="mb-10">
        <div className="flex items-center gap-3 mb-3">
          <div className={`p-2 rounded-xl ${isDark ? 'bg-bb-accent/10' : 'bg-bb-accent-dark/10'}`}>
            <BarChart3 className={`w-7 h-7 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
          </div>
          <h1 className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Results Showcase
          </h1>
        </div>
        <p className={`text-base max-w-2xl ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          Interactive visualizations comparing 101+ models across accuracy, efficiency, and multi-dimensional capability profiles.
        </p>
      </div>

      {/* ── Section 1: Top 20 Bar Chart ── */}
      <Card isDark={isDark} className="mb-8">
        <SectionHeader
          icon={TrendingUp}
          title="Top 20 Models by Overall Accuracy"
          subtitle="Sorted by overall accuracy across all 79 tasks"
          isDark={isDark}
        />
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={top20} margin={{ top: 5, right: 20, left: 0, bottom: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis
                dataKey="shortModel"
                tick={axisStyle}
                angle={-45}
                textAnchor="end"
                interval={0}
                height={90}
              />
              <YAxis tick={axisStyle} domain={[0, 100]} tickFormatter={v => `${v}%`} width={45} />
              <Tooltip content={<BarTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 12, color: isDark ? '#9ca3af' : '#6b7280', paddingTop: 8 }}
              />
              <Bar dataKey="accuracy" name="Overall Accuracy" fill="#00e676" radius={[3, 3, 0, 0]} />
              <Bar dataKey="instruction" name="Instruction Following" fill="#0ea5e9" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* ── Section 2: Model Selector + Radar Chart ── */}
      <Card isDark={isDark} className="mb-8">
        <SectionHeader
          icon={Target}
          title="Multi-Dimensional Model Comparison"
          subtitle="Compare up to 5 models across easy/medium/hard accuracy and instruction following"
          isDark={isDark}
        />

        {/* Selected model chips */}
        <div className="flex flex-wrap gap-2 mb-4">
          {selectedModels.map((modelName, idx) => (
            <span
              key={modelName}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border"
              style={{
                borderColor: MODEL_COLORS[idx % MODEL_COLORS.length] + '60',
                backgroundColor: MODEL_COLORS[idx % MODEL_COLORS.length] + '15',
                color: MODEL_COLORS[idx % MODEL_COLORS.length],
              }}
            >
              <span className="max-w-[160px] truncate">{shortName(modelName)}</span>
              <button onClick={() => removeModel(modelName)} className="hover:opacity-70 transition-opacity">
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}

          {selectedModels.length < 5 && (
            <div className="relative">
              <button
                onClick={() => setSelectorOpen(o => !o)}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${
                  isDark
                    ? 'border-bb-dark-50/40 text-gray-400 hover:border-bb-accent/40 hover:text-bb-accent'
                    : 'border-bb-light-300 text-gray-500 hover:border-bb-accent-dark/40 hover:text-bb-accent-dark'
                }`}
              >
                <Plus className="w-3 h-3" />
                Add model
                <ChevronDown className={`w-3 h-3 transition-transform ${selectorOpen ? 'rotate-180' : ''}`} />
              </button>
              {selectorOpen && (
                <div className={`absolute top-full left-0 mt-1 z-30 w-72 rounded-xl border shadow-2xl overflow-hidden ${
                  isDark
                    ? 'bg-bb-dark-300 border-bb-dark-50/30'
                    : 'bg-white border-bb-light-300'
                }`}>
                  <div className={`p-2 border-b ${isDark ? 'border-bb-dark-50/20' : 'border-bb-light-300/50'}`}>
                    <input
                      autoFocus
                      type="text"
                      placeholder="Search models..."
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      className={`w-full px-3 py-1.5 rounded-lg text-xs outline-none ${
                        isDark
                          ? 'bg-bb-dark-500/50 text-white placeholder-gray-600 border border-bb-dark-50/20'
                          : 'bg-bb-light-100 text-gray-900 placeholder-gray-400 border border-bb-light-300'
                      }`}
                    />
                  </div>
                  <div className="max-h-48 overflow-y-auto">
                    {filteredOptions.map(m => (
                      <button
                        key={m.model}
                        onClick={() => addModel(m.model)}
                        disabled={selectedModels.includes(m.model)}
                        className={`w-full text-left px-3 py-2 text-xs transition-colors ${
                          selectedModels.includes(m.model)
                            ? isDark ? 'text-gray-600 cursor-not-allowed' : 'text-gray-300 cursor-not-allowed'
                            : isDark
                              ? 'text-gray-300 hover:bg-bb-dark-500/50 hover:text-white'
                              : 'text-gray-700 hover:bg-bb-light-100 hover:text-gray-900'
                        }`}
                      >
                        <span className="block truncate">{m.model}</span>
                        <span className={`text-[10px] ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                          Acc: {m.accuracy.toFixed(1)}% · IF: {m.instruction.toFixed(1)}%
                        </span>
                      </button>
                    ))}
                    {filteredOptions.length === 0 && (
                      <p className={`text-center py-4 text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                        No models found
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Radar Chart */}
        {selectedModels.length > 0 ? (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData} margin={{ top: 10, right: 40, bottom: 10, left: 40 }}>
                <PolarGrid stroke={gridColor} />
                <PolarAngleAxis
                  dataKey="metric"
                  tick={{ fill: isDark ? '#d1d5db' : '#374151', fontSize: 12, fontWeight: 500 }}
                />
                <PolarRadiusAxis
                  angle={30}
                  domain={[0, 100]}
                  tick={{ fill: isDark ? '#6b7280' : '#9ca3af', fontSize: 10 }}
                  tickFormatter={v => `${v}%`}
                />
                {selectedModels.map((modelName, idx) => (
                  <Radar
                    key={modelName}
                    name={shortName(modelName)}
                    dataKey={shortName(modelName)}
                    stroke={MODEL_COLORS[idx % MODEL_COLORS.length]}
                    fill={MODEL_COLORS[idx % MODEL_COLORS.length]}
                    fillOpacity={0.08}
                    strokeWidth={2}
                  />
                ))}
                <Legend
                  wrapperStyle={{ fontSize: 11, color: isDark ? '#9ca3af' : '#6b7280' }}
                />
                <Tooltip content={<RadarTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className={`h-80 flex items-center justify-center rounded-xl border-2 border-dashed ${
            isDark ? 'border-bb-dark-50/20 text-gray-600' : 'border-bb-light-300 text-gray-400'
          }`}>
            <p className="text-sm">Add at least one model to see the radar chart</p>
          </div>
        )}
      </Card>

      {/* ── Section 3: Scatter Plot (Accuracy vs Tokens) ── */}
      <Card isDark={isDark}>
        <SectionHeader
          icon={Zap}
          title="Accuracy vs. Token Usage (Efficiency)"
          subtitle="Models in the top-right are both accurate and concise. Bubble size represents efficiency score."
          isDark={isDark}
        />
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis
                dataKey="accuracy"
                name="Accuracy"
                type="number"
                domain={[0, 100]}
                tickFormatter={v => `${v}%`}
                tick={axisStyle}
                label={{ value: 'Overall Accuracy (%)', position: 'insideBottom', offset: -15, fill: isDark ? '#9ca3af' : '#6b7280', fontSize: 11 }}
              />
              <YAxis
                dataKey="tokens"
                name="Avg Tokens"
                type="number"
                tick={axisStyle}
                tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                label={{ value: 'Avg Tokens', angle: -90, position: 'insideLeft', offset: 15, fill: isDark ? '#9ca3af' : '#6b7280', fontSize: 11 }}
              />
              <ZAxis dataKey="efficiency" range={[30, 200]} />
              <Tooltip content={<ScatterTooltip />} />
              <Scatter
                data={scatterData}
                fill="#00e676"
                fillOpacity={0.7}
                stroke={isDark ? '#00e67680' : '#00a85280'}
                strokeWidth={1}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <p className={`text-xs mt-2 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
          * Bubble size proportional to efficiency score (accuracy / avg_tokens × 1000). Hover a point for model details.
        </p>
      </Card>

      {/* Summary stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-8">
        {[
          { label: 'Models Evaluated', value: modelData.length + '+' },
          { label: 'Best Accuracy', value: Math.max(...modelData.map(m => m.accuracy)).toFixed(1) + '%' },
          { label: 'Avg Accuracy', value: (modelData.reduce((s, m) => s + m.accuracy, 0) / modelData.length).toFixed(1) + '%' },
          { label: 'Tasks Covered', value: '79' },
        ].map(({ label, value }) => (
          <div
            key={label}
            className={`rounded-xl p-4 text-center border ${
              isDark
                ? 'bg-bb-dark-300/40 border-bb-dark-50/20'
                : 'bg-white/50 border-bb-light-300/50 shadow-sm'
            }`}
          >
            <div className={`text-2xl font-bold font-mono ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>
              {value}
            </div>
            <div className={`text-xs mt-1 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

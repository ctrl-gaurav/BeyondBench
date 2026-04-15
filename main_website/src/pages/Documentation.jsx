import { useState, useMemo } from 'react'
import { BookOpen, Terminal, Code, Settings, Layers, Cpu, Package, FileText, Hexagon, Copy, Check, Zap, Database, GitBranch, Shield, BarChart3, Wrench, Globe, AlertTriangle, Lightbulb, Search, ChevronDown } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'
import { usePyPIVersion } from '../hooks/usePyPIVersion'

/* ============ SYNTAX HIGHLIGHTING ============ */

const darkColors = {
  comment: '#6b7280', keyword: '#c084fc', string: '#34d399', number: '#f59e0b',
  function: '#60a5fa', operator: '#9ca3af', punctuation: '#9ca3af', key: '#60a5fa',
  boolean: '#f59e0b', decorator: '#c084fc', flag: '#60a5fa', text: '#d1d5db',
}
const lightColors = {
  comment: '#9ca3af', keyword: '#7c3aed', string: '#059669', number: '#d97706',
  function: '#2563eb', operator: '#64748b', punctuation: '#64748b', key: '#2563eb',
  boolean: '#d97706', decorator: '#7c3aed', flag: '#2563eb', text: '#374151',
}

function tokenizePython(code) {
  const tokens = []
  const keywords = new Set(['import','from','as','def','return','class','if','else','elif','for','while','with','try','except','raise','True','False','None','print','self','in','not','and','or','is','lambda','yield','async','await','pass','break','continue','del','global','nonlocal','assert','finally'])
  let i = 0
  while (i < code.length) {
    if (code[i] === '#') { let e = code.indexOf('\n', i); if (e === -1) e = code.length; tokens.push({ type: 'comment', value: code.slice(i, e) }); i = e; continue }
    if (code.slice(i, i+3) === '"""' || code.slice(i, i+3) === "'''") { const q = code.slice(i, i+3); let e = code.indexOf(q, i+3); if (e === -1) e = code.length-3; tokens.push({ type: 'string', value: code.slice(i, e+3) }); i = e+3; continue }
    if (code[i] === '"' || code[i] === "'") { const q = code[i]; let j = i+1; while (j < code.length && code[j] !== q) { if (code[j] === '\\') j++; j++ } tokens.push({ type: 'string', value: code.slice(i, j+1) }); i = j+1; continue }
    if (code[i] === '@' && (i === 0 || code[i-1] === '\n' || /\s/.test(code[i-1]))) { let j = i+1; while (j < code.length && /[\w.]/.test(code[j])) j++; tokens.push({ type: 'decorator', value: code.slice(i, j) }); i = j; continue }
    if (/\d/.test(code[i]) && (i === 0 || !/[\w.]/.test(code[i-1]))) { let j = i; while (j < code.length && /[\d.eE_xXa-fA-F]/.test(code[j])) j++; tokens.push({ type: 'number', value: code.slice(i, j) }); i = j; continue }
    if (/[a-zA-Z_]/.test(code[i])) { let j = i; while (j < code.length && /[\w]/.test(code[j])) j++; const w = code.slice(i, j); if (keywords.has(w)) tokens.push({ type: 'keyword', value: w }); else if (j < code.length && code[j] === '(') tokens.push({ type: 'function', value: w }); else tokens.push({ type: 'text', value: w }); i = j; continue }
    if (/[=+\-*/<>!&|^~%]/.test(code[i])) { tokens.push({ type: 'operator', value: code[i] }); i++; continue }
    if (/[()[\]{},;:.]/.test(code[i])) { tokens.push({ type: 'punctuation', value: code[i] }); i++; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenizeBash(code) {
  const tokens = []
  const keywords = new Set(['python','pip','git','cd','source','export','npm','pytest','beyondbench','CUDA_VISIBLE_DEVICES','rm','tail','mkdir','echo','cat','chmod','sudo','apt','brew'])
  let i = 0
  while (i < code.length) {
    if (code[i] === '#') { let e = code.indexOf('\n', i); if (e === -1) e = code.length; tokens.push({ type: 'comment', value: code.slice(i, e) }); i = e; continue }
    if (code[i] === '"' || code[i] === "'") { const q = code[i]; let j = i+1; while (j < code.length && code[j] !== q) { if (code[j] === '\\') j++; j++ } tokens.push({ type: 'string', value: code.slice(i, j+1) }); i = j+1; continue }
    if (code[i] === '-' && i+1 < code.length && /[a-zA-Z-]/.test(code[i+1])) { let j = i; while (j < code.length && /[\w-]/.test(code[j])) j++; tokens.push({ type: 'flag', value: code.slice(i, j) }); i = j; continue }
    if (/[a-zA-Z_]/.test(code[i])) { let j = i; while (j < code.length && /[\w.\-/]/.test(code[j])) j++; const w = code.slice(i, j); if (keywords.has(w)) tokens.push({ type: 'keyword', value: w }); else tokens.push({ type: 'text', value: w }); i = j; continue }
    if (/\d/.test(code[i])) { let j = i; while (j < code.length && /[\d.]/.test(code[j])) j++; tokens.push({ type: 'number', value: code.slice(i, j) }); i = j; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenizeYaml(code) {
  const tokens = []
  code.split('\n').forEach((line, li) => {
    if (li > 0) tokens.push({ type: 'text', value: '\n' })
    let i = 0
    while (i < line.length && /\s/.test(line[i])) { tokens.push({ type: 'text', value: line[i] }); i++ }
    if (line[i] === '#') { tokens.push({ type: 'comment', value: line.slice(i) }); return }
    const ci = line.indexOf(':', i)
    if (ci > i && /^[\w.-]+$/.test(line.slice(i, ci).trim())) {
      tokens.push({ type: 'key', value: line.slice(i, ci) }); tokens.push({ type: 'punctuation', value: ':' })
      const rest = line.slice(ci+1); const trimmed = rest.trim()
      if (trimmed) {
        if (/^".*"$/.test(trimmed) || /^'.*'$/.test(trimmed)) { tokens.push({ type: 'text', value: rest.slice(0, rest.indexOf(trimmed)) }); tokens.push({ type: 'string', value: trimmed }) }
        else if (/^(true|false|null|none|yes|no)$/i.test(trimmed)) { tokens.push({ type: 'text', value: rest.slice(0, rest.indexOf(trimmed)) }); tokens.push({ type: 'boolean', value: trimmed }) }
        else if (/^-?\d+\.?\d*$/.test(trimmed)) { tokens.push({ type: 'text', value: rest.slice(0, rest.indexOf(trimmed)) }); tokens.push({ type: 'number', value: trimmed }) }
        else { const commentIdx = rest.indexOf(' #'); if (commentIdx > 0) { tokens.push({ type: 'text', value: rest.slice(0, commentIdx) }); tokens.push({ type: 'comment', value: rest.slice(commentIdx) }) } else { tokens.push({ type: 'text', value: rest }) } }
      }
      return
    }
    if (line[i] === '-') { tokens.push({ type: 'punctuation', value: '-' }); tokens.push({ type: 'text', value: line.slice(i+1) }); return }
    tokens.push({ type: 'text', value: line.slice(i) })
  })
  return tokens
}

function tokenizeJson(code) {
  const tokens = []
  let i = 0
  while (i < code.length) {
    if (code[i] === '"') {
      let j = i+1; while (j < code.length && code[j] !== '"') { if (code[j] === '\\') j++; j++ }
      const s = code.slice(i, j+1); let k = j+1; while (k < code.length && /\s/.test(code[k])) k++
      tokens.push({ type: code[k] === ':' ? 'key' : 'string', value: s }); i = j+1; continue
    }
    if (/\d/.test(code[i]) || (code[i] === '-' && i+1 < code.length && /\d/.test(code[i+1]))) { let j = i; if (code[j] === '-') j++; while (j < code.length && /[\d.eE+-]/.test(code[j])) j++; tokens.push({ type: 'number', value: code.slice(i, j) }); i = j; continue }
    if (/[a-zA-Z]/.test(code[i])) { let j = i; while (j < code.length && /[a-zA-Z]/.test(code[j])) j++; const w = code.slice(i, j); tokens.push({ type: (w === 'true' || w === 'false' || w === 'null') ? 'boolean' : 'text', value: w }); i = j; continue }
    if (/[{}[\]:,]/.test(code[i])) { tokens.push({ type: 'punctuation', value: code[i] }); i++; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenizeBibtex(code) {
  const tokens = []
  let i = 0
  while (i < code.length) {
    if (code[i] === '@') { let j = i+1; while (j < code.length && /\w/.test(code[j])) j++; tokens.push({ type: 'decorator', value: code.slice(i, j) }); i = j; continue }
    if (code[i] === '{' || code[i] === '}') { tokens.push({ type: 'punctuation', value: code[i] }); i++; continue }
    if (/[a-zA-Z]/.test(code[i])) { let j = i; while (j < code.length && /[\w]/.test(code[j])) j++; const w = code.slice(i, j); let k = j; while (k < code.length && code[k] === ' ') k++; tokens.push({ type: code[k] === '=' ? 'key' : 'string', value: w }); i = j; continue }
    tokens.push({ type: 'text', value: code[i] }); i++
  }
  return tokens
}

function tokenize(code, language) {
  switch (language) {
    case 'python': case 'py': return tokenizePython(code)
    case 'bash': case 'shell': case 'sh': return tokenizeBash(code)
    case 'yaml': case 'yml': return tokenizeYaml(code)
    case 'json': return tokenizeJson(code)
    case 'bibtex': return tokenizeBibtex(code)
    default: return [{ type: 'text', value: code }]
  }
}

function CodeBlock({ code, language = 'bash' }) {
  const [copied, setCopied] = useState(false)
  const { isDark } = useTheme()
  const colors = isDark ? darkColors : lightColors

  const highlighted = useMemo(() => {
    return tokenize(code, language).map((token, i) => {
      if (token.type === 'text') return <span key={i}>{token.value}</span>
      return <span key={i} style={{ color: colors[token.type] }}>{token.value}</span>
    })
  }, [code, language, colors])

  function handleCopy() {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={`relative group my-3 rounded-xl overflow-hidden transition-all duration-300 ${
      isDark
        ? 'bg-[#0a0e18] border border-bb-dark-50/15 hover:border-bb-accent/20 shadow-lg shadow-black/20'
        : 'bg-[#f7f8fc] border border-gray-200/60 hover:border-bb-accent-dark/30 shadow-sm'
    }`}>
      <div className={`flex items-center justify-between px-4 py-2.5 border-b ${
        isDark ? 'border-bb-dark-50/10 bg-white/[0.02]' : 'border-gray-200/30 bg-gray-50/50'
      }`}>
        <div className="flex items-center gap-2.5">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#febc2e]/80" />
            <div className="w-2.5 h-2.5 rounded-full bg-[#28c840]/80" />
          </div>
          <span className={`text-[10px] uppercase tracking-wider font-mono ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{language}</span>
        </div>
        <button
          onClick={handleCopy}
          className={`text-xs font-mono px-2.5 py-1 rounded-md transition-all duration-200 ${
            copied
              ? isDark ? 'text-green-400 bg-green-500/10' : 'text-green-600 bg-green-500/10'
              : isDark ? 'text-gray-600 hover:text-gray-300 hover:bg-white/5' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
          }`}
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className={`p-4 overflow-x-auto text-[13px] font-mono leading-relaxed ${
        isDark ? 'text-gray-300' : 'text-gray-700'
      }`}>
        <code>{highlighted}</code>
      </pre>
    </div>
  )
}

function SubSection({ title, children }) {
  const { isDark } = useTheme()
  return (
    <div className="mt-6">
      <h3 className={`text-base font-semibold mb-3 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>{title}</h3>
      <div className={`text-sm space-y-3 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
    </div>
  )
}

function Callout({ type = 'info', children }) {
  const { isDark } = useTheme()
  const styles = {
    info: isDark ? 'border-bb-accent/40 bg-bb-accent/5' : 'border-bb-accent-dark/40 bg-bb-accent-dark/5',
    warning: 'border-yellow-500/40 bg-yellow-500/5',
    tip: 'border-green-400/40 bg-green-400/5',
  }
  const icons = {
    info: <Shield className={`w-4 h-4 shrink-0 mt-0.5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />,
    warning: <AlertTriangle className="w-4 h-4 text-yellow-500 shrink-0 mt-0.5" />,
    tip: <Lightbulb className="w-4 h-4 text-green-400 shrink-0 mt-0.5" />,
  }
  const labels = { info: 'Note', warning: 'Warning', tip: 'Tip' }
  const labelColors = {
    info: isDark ? 'text-bb-accent' : 'text-bb-accent-dark',
    warning: 'text-yellow-500',
    tip: 'text-green-400',
  }
  return (
    <div className={`border-l-4 rounded-xl p-4 ${styles[type]}`}>
      <div className={`text-xs font-mono font-bold uppercase tracking-wider mb-1 flex items-center gap-1.5 ${labelColors[type]}`}>
        {icons[type]}
        {labels[type]}
      </div>
      <div className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
    </div>
  )
}

const NAV_ITEMS = [
  { id: 'overview', icon: BookOpen, label: 'Overview', iconPath: 'M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253' },
  { id: 'installation', icon: Package, label: 'Installation', iconPath: 'M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4' },
  { id: 'quickstart', icon: Zap, label: 'Quick Start', iconPath: 'M13 10V3L4 14h7v7l9-11h-7z' },
  { id: 'eval-openai', icon: Globe, label: 'OpenAI Evaluation', iconPath: 'M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9' },
  { id: 'eval-gemini', icon: Globe, label: 'Gemini Evaluation', iconPath: 'M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9' },
  { id: 'eval-anthropic', icon: Globe, label: 'Anthropic Evaluation', iconPath: 'M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9' },
  { id: 'eval-vllm', icon: Cpu, label: 'vLLM Evaluation', iconPath: 'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z' },
  { id: 'eval-transformers', icon: Cpu, label: 'Transformers Evaluation', iconPath: 'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z' },
  { id: 'cli', icon: Terminal, label: 'CLI Reference', iconPath: 'M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z' },
  { id: 'python-api', icon: Code, label: 'Python API', iconPath: 'M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4' },
  { id: 'backends', icon: Cpu, label: 'Backend Overview', iconPath: 'M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2' },
  { id: 'tasks', icon: Layers, label: 'Task Suites', iconPath: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10' },
  { id: 'advanced-eval', icon: BarChart3, label: 'Advanced Evaluation', iconPath: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  { id: 'configuration', icon: Settings, label: 'Configuration', iconPath: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z' },
  { id: 'extending', icon: GitBranch, label: 'Extending', iconPath: 'M6 3v12m0 0a3 3 0 103 3V9m-3 6a3 3 0 01-3-3m12 0a3 3 0 003-3V6a3 3 0 00-3-3m0 0a3 3 0 10-3 3' },
  { id: 'output', icon: Database, label: 'Output Format', iconPath: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4' },
  { id: 'troubleshooting', icon: Wrench, label: 'Troubleshooting', iconPath: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0' },
  { id: 'citation', icon: FileText, label: 'Citation', iconPath: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z' },
]

/* ============ SECTION CONTENT COMPONENTS ============ */

function OverviewContent({ isDark, cardCls, headCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Overview</h2>
        <p className={textCls}>
          <strong className={isDark ? 'text-gray-200' : 'text-gray-800'}>BeyondBench</strong> is a contamination-resistant evaluation framework for assessing reasoning capabilities in Large Language Models. Unlike traditional benchmarks that risk contamination from internet-scale training data, BeyondBench uses <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>algorithmic problem generation</strong> to create mathematically grounded problems on the fly.
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { title: '79 Tasks', desc: '44 easy + 15 medium + 20 hard difficulty tasks covering arithmetic, sequences, constraint satisfaction, and NP-complete problems.' },
          { title: '138 Variations', desc: 'Fine-grained task variations for nuanced capability evaluation across different problem formulations.' },
          { title: '>10^15 Instances', desc: 'Each task generates from a problem space of over 10^15 unique instances, ensuring contamination resistance.' },
          { title: '5 Backends', desc: 'vLLM, Transformers, OpenAI, Gemini, and Anthropic APIs. Evaluate any model from any provider.' },
        ].map(item => (
          <div key={item.title} className={cardCls}>
            <div className={`text-sm font-semibold mb-1 ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>{item.title}</div>
            <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{item.desc}</div>
          </div>
        ))}
      </div>
      <div className={cardCls}>
        <p className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
          Paper accepted at <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>ICLR 2026</strong>. Read the full paper at <a href="https://arxiv.org/abs/2509.24210" target="_blank" rel="noopener noreferrer" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>arXiv:2509.24210</a>.
        </p>
      </div>
    </div>
  )
}

function InstallationContent({ isDark, cardCls, headCls, subheadCls, textCls, pypiVersion }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Installation</h2>
        <p className={textCls}>Get BeyondBench up and running in minutes.{pypiVersion && <span className="ml-2 font-mono text-xs opacity-60">Latest: v{pypiVersion}</span>}</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>From PyPI (Recommended)</h3>
        <CodeBlock code="pip install beyondbench" />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>From Source</h3>
        <CodeBlock code={`git clone https://github.com/ctrl-gaurav/BeyondBench.git
cd BeyondBench
pip install -e .`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>With Optional Dependencies</h3>
        <CodeBlock code={`# All API clients (OpenAI, Gemini, Anthropic)
pip install beyondbench[all-apis]

# vLLM support (requires CUDA)
pip install beyondbench[vllm]

# Development tools
pip install beyondbench[dev]

# Everything included
pip install beyondbench[full]`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Interactive Setup</h3>
        <p className={textCls}>For first-time users, use the interactive setup script:</p>
        <CodeBlock code="./setup_beyondbench.sh" />
        <p className={`text-xs mt-2 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>This will create a conda environment, install dependencies, verify imports, and run tests.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Verify Installation</h3>
        <CodeBlock code={`# Check version
beyondbench --version

# Verify all imports work
python -c "from beyondbench import EvaluationEngine, TaskRegistry; print('OK')"

# List all available tasks
beyondbench list-tasks`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Requirements</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {[
            { label: 'Python', value: '3.8+' },
            { label: 'PyTorch', value: '2.0+' },
            { label: 'Transformers', value: '4.30+' },
            { label: 'CUDA', value: 'For vLLM and local GPU inference' },
          ].map(r => (
            <div key={r.label} className={`flex items-center gap-3 px-4 py-2.5 rounded-lg ${isDark ? 'bg-bb-dark-400/30' : 'bg-gray-50'}`}>
              <span className={`text-xs font-mono font-bold ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`}>{r.label}</span>
              <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{r.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function QuickStartContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Quick Start</h2>
        <p className={textCls}>Get started with BeyondBench in three different ways.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Interactive Wizard</h3>
        <p className={textCls}>Launch the interactive wizard for guided setup:</p>
        <CodeBlock code="beyondbench" />
        <p className={`text-xs mt-2 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>The wizard walks you through model selection, API key configuration, task suite selection, and parameter tuning.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Command Line</h3>
        <CodeBlock code={`# Evaluate GPT-4o on the easy suite
beyondbench evaluate --model-id gpt-4o --api-provider openai --suite easy

# Evaluate a local model with vLLM
beyondbench evaluate --model-id meta-llama/Llama-3.2-3B-Instruct \\
  --backend vllm --suite all

# Evaluate Claude on hard tasks
beyondbench evaluate --model-id claude-sonnet-4-20250514 \\
  --api-provider anthropic --suite hard --datapoints 50

# List all available tasks
beyondbench list-tasks --suite all`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Python API</h3>
        <CodeBlock language="python" code={`from beyondbench import EvaluationEngine, ModelHandler, TaskRegistry

# Initialize model handler
model = ModelHandler(
    model_id="gpt-4o",
    api_provider="openai",
    api_key="your-api-key"
)

# Run evaluation
engine = EvaluationEngine(model_handler=model, output_dir="./results")
results = engine.run_evaluation(suite="easy", datapoints=100)

# Print results
print(f"Average Accuracy: {results['summary']['avg_accuracy']:.2%}")
print(f"Instruction Following: {results['summary']['avg_instruction_following']:.2%}")`} />
      </div>
    </div>
  )
}

function EvalOpenAIContent({ isDark, cardCls, headCls, subheadCls, textCls, labelCls, inlineCodeCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Evaluating OpenAI Models</h2>
        <p className={textCls}>Evaluate GPT-4, GPT-4o, GPT-5, o1, o3, o4-Mini, and other OpenAI models.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Setup</h3>
        <CodeBlock code={`# Set your API key
export OPENAI_API_KEY="sk-..."

# Or pass it directly via CLI
beyondbench evaluate --model-id gpt-4o --api-provider openai --api-key "sk-..."`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Evaluate GPT-4o (All Suites)</h3>
        <CodeBlock code={`beyondbench evaluate \\
  --model-id gpt-4o \\
  --api-provider openai \\
  --suite all \\
  --datapoints 100 \\
  --temperature 0.7 \\
  --max-tokens 512 \\
  --output-dir ./results/gpt-4o \\
  --store-details`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Evaluate GPT-5 with Reasoning Effort</h3>
        <CodeBlock code={`# Minimal reasoning (fast, cheaper)
beyondbench evaluate --model-id gpt-5 --api-provider openai \\
  --reasoning-effort minimal --suite easy

# Medium reasoning (balanced)
beyondbench evaluate --model-id gpt-5 --api-provider openai \\
  --reasoning-effort medium --suite all

# High reasoning (maximum quality)
beyondbench evaluate --model-id gpt-5 --api-provider openai \\
  --reasoning-effort high --suite hard --datapoints 50`} />
        <Callout type="tip">
          Use <code className={inlineCodeCls}>--reasoning-effort high</code> for hard tasks to get the best accuracy. Use <code className={inlineCodeCls}>minimal</code> for cost-efficient easy task evaluation.
        </Callout>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Evaluate o3 / o4-Mini Reasoning Models</h3>
        <CodeBlock code={`# o3 - reasoning model
beyondbench evaluate --model-id o3 --api-provider openai \\
  --suite hard --datapoints 50 --max-tokens 4096

# o4-mini - lightweight reasoning
beyondbench evaluate --model-id o4-mini --api-provider openai \\
  --suite all --datapoints 100`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Python API</h3>
        <CodeBlock language="python" code={`from beyondbench.models.model_handler import ModelHandler
from beyondbench import EvaluationEngine

# Initialize OpenAI model
model = ModelHandler(
    model_id="gpt-4o",
    api_provider="openai",
    api_key="sk-..."  # or set OPENAI_API_KEY env var
)

# Run evaluation
engine = EvaluationEngine(model_handler=model, output_dir="./results/gpt-4o")
results = engine.run_evaluation(
    suite="all",
    datapoints=100,
    temperature=0.7,
    max_tokens=512,
    store_details=True
)

# Results breakdown
for task, metrics in results['task_results'].items():
    print(f"{task}: acc={metrics['accuracy']:.2%}, inst={metrics['instruction_following']:.2%}")`} />
      </div>
      <div className={cardCls}>
        <div className={labelCls}>Supported Models</div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 font-mono text-xs">
          {['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-5', 'o1', 'o1-mini', 'o3', 'o3-mini', 'o4-mini'].map(m => (
            <div key={m} className={`px-3 py-1.5 rounded-lg ${isDark ? 'bg-bb-dark-400/30 text-gray-400' : 'bg-gray-50 text-gray-600'}`}>{m}</div>
          ))}
        </div>
      </div>
    </div>
  )
}

function EvalGeminiContent({ isDark, cardCls, headCls, subheadCls, textCls, labelCls, inlineCodeCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Evaluating Google Gemini Models</h2>
        <p className={textCls}>Evaluate Gemini 2.0 Flash, Gemini 2.5 Pro, Gemini Ultra, and other Google AI models.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Setup</h3>
        <CodeBlock code={`# Set your Gemini API key
export GEMINI_API_KEY="AI..."

# Or use Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Evaluate Gemini 2.0 Flash</h3>
        <CodeBlock code={`beyondbench evaluate \\
  --model-id gemini-2.0-flash \\
  --api-provider gemini \\
  --suite all \\
  --datapoints 100 \\
  --thinking-budget 1024 \\
  --output-dir ./results/gemini-flash`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Evaluate Gemini 2.5 Pro with Extended Thinking</h3>
        <CodeBlock code={`# Low thinking budget (fast)
beyondbench evaluate --model-id gemini-2.5-pro \\
  --api-provider gemini --thinking-budget 512 --suite easy

# High thinking budget (for hard reasoning tasks)
beyondbench evaluate --model-id gemini-2.5-pro \\
  --api-provider gemini --thinking-budget 4096 --suite hard

# Maximum thinking for complex problems
beyondbench evaluate --model-id gemini-2.5-pro \\
  --api-provider gemini --thinking-budget 8192 --suite hard --datapoints 50`} />
        <Callout type="tip">
          Increase <code className={inlineCodeCls}>--thinking-budget</code> for hard tasks like Sudoku, N-Queens, and Boolean SAT. Lower budgets work well for easy arithmetic tasks.
        </Callout>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Python API</h3>
        <CodeBlock language="python" code={`from beyondbench.models.model_handler import ModelHandler
from beyondbench import EvaluationEngine

model = ModelHandler(
    model_id="gemini-2.5-pro",
    api_provider="gemini",
    api_key="AI..."  # or set GEMINI_API_KEY env var
)

engine = EvaluationEngine(model_handler=model, output_dir="./results/gemini")
results = engine.run_evaluation(
    suite="all",
    datapoints=100,
    store_details=True
)
print(f"Overall Accuracy: {results['summary']['avg_accuracy']:.2%}")`} />
      </div>
      <div className={cardCls}>
        <div className={labelCls}>Supported Models</div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 font-mono text-xs">
          {['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-1.5-pro', 'gemini-1.5-flash'].map(m => (
            <div key={m} className={`px-3 py-1.5 rounded-lg ${isDark ? 'bg-bb-dark-400/30 text-gray-400' : 'bg-gray-50 text-gray-600'}`}>{m}</div>
          ))}
        </div>
      </div>
    </div>
  )
}

function EvalAnthropicContent({ isDark, cardCls, headCls, subheadCls, textCls, labelCls, inlineCodeCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Evaluating Anthropic Models</h2>
        <p className={textCls}>Evaluate Claude Sonnet 4, Claude Opus 4, Claude 3.5 Haiku, and other Anthropic models.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Setup</h3>
        <CodeBlock code={`# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Evaluate Claude Sonnet 4</h3>
        <CodeBlock code={`beyondbench evaluate \\
  --model-id claude-sonnet-4-20250514 \\
  --api-provider anthropic \\
  --suite all \\
  --datapoints 100 \\
  --temperature 0.7 \\
  --max-tokens 1024 \\
  --output-dir ./results/claude-sonnet`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Evaluate Claude Opus 4</h3>
        <CodeBlock code={`beyondbench evaluate \\
  --model-id claude-opus-4-20250514 \\
  --api-provider anthropic \\
  --suite hard \\
  --datapoints 50 \\
  --max-tokens 4096 \\
  --output-dir ./results/claude-opus`} />
        <Callout type="info">
          Claude Opus 4 has higher per-token cost. Consider using smaller <code className={inlineCodeCls}>--datapoints</code> for initial evaluation, then scale up.
        </Callout>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Python API</h3>
        <CodeBlock language="python" code={`from beyondbench.models.model_handler import ModelHandler
from beyondbench import EvaluationEngine

model = ModelHandler(
    model_id="claude-sonnet-4-20250514",
    api_provider="anthropic",
    api_key="sk-ant-..."  # or set ANTHROPIC_API_KEY env var
)

engine = EvaluationEngine(model_handler=model, output_dir="./results/claude")
results = engine.run_evaluation(suite="all", datapoints=100)

# Compare suites
for suite in ['easy', 'medium', 'hard']:
    suite_results = {k: v for k, v in results['task_results'].items()
                     if v.get('suite') == suite}
    avg = sum(v['accuracy'] for v in suite_results.values()) / len(suite_results)
    print(f"{suite.upper()}: {avg:.2%}")`} />
      </div>
      <div className={cardCls}>
        <div className={labelCls}>Supported Models</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 font-mono text-xs">
          {['claude-sonnet-4-20250514', 'claude-opus-4-20250514', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'].map(m => (
            <div key={m} className={`px-3 py-1.5 rounded-lg ${isDark ? 'bg-bb-dark-400/30 text-gray-400' : 'bg-gray-50 text-gray-600'}`}>{m}</div>
          ))}
        </div>
      </div>
    </div>
  )
}

function EvalVllmContent({ isDark, cardCls, headCls, subheadCls, textCls, inlineCodeCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Evaluating with vLLM</h2>
        <p className={textCls}>Use vLLM for high-throughput local evaluation of any HuggingFace model with PagedAttention and continuous batching.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Prerequisites</h3>
        <CodeBlock code={`# Install vLLM (requires CUDA)
pip install beyondbench[vllm]

# Or install vLLM separately
pip install vllm`} />
        <Callout type="warning">
          vLLM requires NVIDIA GPUs with CUDA support. Check your GPU memory before loading large models.
        </Callout>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Basic Usage</h3>
        <CodeBlock code={`# Evaluate a 7B model on single GPU
beyondbench evaluate \\
  --model-id Qwen/Qwen2.5-7B-Instruct \\
  --backend vllm \\
  --suite all \\
  --datapoints 100`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Multi-GPU with Tensor Parallelism</h3>
        <CodeBlock code={`# 2 GPUs for a 32B model
beyondbench evaluate \\
  --model-id Qwen/Qwen2.5-32B-Instruct \\
  --backend vllm \\
  --tensor-parallel-size 2 \\
  --gpu-memory-utilization 0.90 \\
  --suite all --datapoints 100

# 4 GPUs for a 70B model
beyondbench evaluate \\
  --model-id meta-llama/Llama-3.1-70B-Instruct \\
  --backend vllm \\
  --tensor-parallel-size 4 \\
  --gpu-memory-utilization 0.95 \\
  --suite all --datapoints 100

# 8 GPUs for a 405B model
beyondbench evaluate \\
  --model-id meta-llama/Llama-3.1-405B-Instruct \\
  --backend vllm \\
  --tensor-parallel-size 8 \\
  --suite easy --datapoints 50`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Python API</h3>
        <CodeBlock language="python" code={`from beyondbench.models.model_handler import ModelHandler
from beyondbench import EvaluationEngine

# Single GPU
model = ModelHandler(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    backend="vllm",
    cuda_device="cuda:0",
    gpu_memory_utilization=0.95
)

# Multi-GPU
model_large = ModelHandler(
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    backend="vllm",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.90,
    trust_remote_code=True
)

engine = EvaluationEngine(model_handler=model, output_dir="./results")
results = engine.run_evaluation(suite="all", datapoints=100, batch_size=8)`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>GPU Memory Guide</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className={`border-b ${isDark ? 'border-bb-dark-50/20' : 'border-gray-200'}`}>
                <th className={`text-left py-2 text-xs font-semibold ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>Model Size</th>
                <th className={`text-left py-2 text-xs font-semibold ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>FP16 VRAM</th>
                <th className={`text-left py-2 text-xs font-semibold ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>Recommended GPUs</th>
              </tr>
            </thead>
            <tbody className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              {[
                ['1-3B', '~6 GB', '1x RTX 3090/4090'],
                ['7-8B', '~16 GB', '1x A100-40G or 1x RTX 4090'],
                ['13-14B', '~28 GB', '1x A100-80G'],
                ['32-34B', '~68 GB', '2x A100-40G or 1x A100-80G'],
                ['65-70B', '~140 GB', '2x A100-80G or 4x A100-40G'],
                ['405B', '~810 GB', '8x A100-80G or 8x H100'],
              ].map(([size, vram, gpus], i) => (
                <tr key={size} className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}>
                  <td className="py-1.5">{size}</td><td className="py-1.5">{vram}</td><td className="py-1.5">{gpus}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function EvalTransformersContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Evaluating with HuggingFace Transformers</h2>
        <p className={textCls}>Use the HuggingFace Transformers backend for flexible local inference, especially useful for debugging and smaller models.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Basic Usage</h3>
        <CodeBlock code={`beyondbench evaluate \\
  --model-id meta-llama/Llama-3.2-3B-Instruct \\
  --backend transformers \\
  --cuda-device cuda:0 \\
  --suite all \\
  --datapoints 100`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Microsoft Phi Models</h3>
        <CodeBlock code={`# Phi-4 Mini
beyondbench evaluate \\
  --model-id microsoft/phi-4-mini-instruct \\
  --backend transformers \\
  --trust-remote-code \\
  --suite all --datapoints 100

# Phi-3 Mini
beyondbench evaluate \\
  --model-id microsoft/phi-3-mini-4k-instruct \\
  --backend transformers \\
  --trust-remote-code \\
  --suite easy --datapoints 50`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Python API with Custom Generation Config</h3>
        <CodeBlock language="python" code={`from beyondbench.models.model_handler import ModelHandler
from beyondbench import EvaluationEngine

model = ModelHandler(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    backend="transformers",
    cuda_device="cuda:0",
    trust_remote_code=True
)

engine = EvaluationEngine(
    model_handler=model,
    output_dir="./results/llama-3.2-3b"
)

results = engine.run_evaluation(
    suite="all",
    datapoints=100,
    temperature=0.1,
    max_tokens=512,
    seed=42,
    store_details=True
)`} />
      </div>
      <div className={cardCls}>
        <Callout type="info">
          The Transformers backend is slower than vLLM but more compatible. Use it when vLLM doesn't support your model architecture, or for debugging purposes.
        </Callout>
      </div>
    </div>
  )
}

function CLIContent({ isDark, cardCls, headCls, subheadCls, textCls, labelCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>CLI Reference</h2>
        <p className={textCls}>Complete command-line interface documentation.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>beyondbench evaluate</h3>
        <p className={`${textCls} mb-4`}>Run model evaluation against BeyondBench tasks.</p>
        <div className={`font-mono text-xs space-y-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          {[
            { group: 'Model & Backend', flags: [
              ['--model-id', 'TEXT', 'Model identifier (required)'],
              ['--backend', '[vllm|transformers]', 'Local inference backend'],
              ['--api-provider', '[openai|gemini|anthropic]', 'API provider'],
              ['--api-key', 'TEXT', 'API key (or set env variable)'],
            ]},
            { group: 'Task Selection', flags: [
              ['--suite', '[easy|medium|hard|all]', 'Task suite to evaluate (default: all)'],
              ['--tasks', 'TEXT...', 'Specific tasks to evaluate'],
              ['--datapoints', 'INT', 'Number of test cases per task (default: 100)'],
              ['--folds', 'INT', 'Number of cross-validation folds (default: 1)'],
            ]},
            { group: 'Generation Parameters', flags: [
              ['--temperature', 'FLOAT', 'Sampling temperature (default: 0.7)'],
              ['--top-p', 'FLOAT', 'Nucleus sampling (default: 0.9)'],
              ['--max-tokens', 'INT', 'Max generation tokens (default: 32768)'],
              ['--seed', 'INT', 'Random seed for reproducibility'],
            ]},
            { group: 'Hardware Configuration', flags: [
              ['--cuda-device', 'TEXT', 'CUDA device (default: cuda:0)'],
              ['--tensor-parallel-size', 'INT', 'Number of GPUs (default: 1)'],
              ['--gpu-memory-utilization', 'FLOAT', 'GPU memory ratio (default: 0.96)'],
              ['--trust-remote-code', '', 'Allow remote code execution'],
              ['--batch-size', 'INT', 'Batch size for inference (default: 1)'],
            ]},
            { group: 'Provider-Specific', flags: [
              ['--reasoning-effort', '[minimal|low|medium|high]', 'OpenAI reasoning effort'],
              ['--thinking-budget', 'INT', 'Gemini thinking budget (default: 1024)'],
            ]},
            { group: 'Output & Logging', flags: [
              ['--output-dir', 'PATH', 'Output directory (default: ./beyondbench_results)'],
              ['--store-details', '', 'Save per-example results'],
              ['--log-level', '[DEBUG|INFO|WARNING|ERROR]', 'Logging level (default: INFO)'],
              ['--max-retries', 'INT', 'Max retries (default: 3)'],
              ['--timeout', 'INT', 'Timeout in seconds (default: 300)'],
            ]},
          ].map(section => (
            <div key={section.group}>
              <div className={`font-semibold mb-2 mt-4 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>{section.group}</div>
              {section.flags.map(([flag, type, desc]) => (
                <div key={flag} className="ml-2"><span className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>{flag}</span> {type} &nbsp;&nbsp; {desc}</div>
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>beyondbench list-tasks</h3>
        <CodeBlock code={`beyondbench list-tasks              # Show all tasks
beyondbench list-tasks --suite easy  # Show only easy tasks
beyondbench list-tasks --suite hard  # Show only hard tasks`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>beyondbench run-config</h3>
        <p className={textCls}>Execute evaluation from a YAML or JSON configuration file.</p>
        <CodeBlock code="beyondbench run-config config.yaml" />
      </div>
    </div>
  )
}

function PythonAPIContent({ isDark, cardCls, headCls, subheadCls, textCls, labelCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Python API</h2>
        <p className={textCls}>Programmatic access to BeyondBench's full evaluation capabilities.</p>
      </div>
      <div className={cardCls}>
        <div className={labelCls}>Core Class</div>
        <h3 className={subheadCls}>ModelHandler</h3>
        <p className={`${textCls} mb-3`}>Unified interface for all model backends.</p>
        <CodeBlock language="python" code={`from beyondbench.models.model_handler import ModelHandler

# OpenAI API
model = ModelHandler(model_id="gpt-4o", api_provider="openai", api_key="sk-...")

# Anthropic API
model = ModelHandler(model_id="claude-sonnet-4-20250514", api_provider="anthropic")

# Google Gemini API
model = ModelHandler(model_id="gemini-2.5-pro", api_provider="gemini")

# vLLM (local)
model = ModelHandler(model_id="Qwen/Qwen2.5-7B-Instruct", backend="vllm", tp_size=1)

# HuggingFace Transformers (local)
model = ModelHandler(model_id="meta-llama/Llama-3.2-3B-Instruct", backend="transformers")`} />
      </div>
      <div className={cardCls}>
        <div className={labelCls}>Core Class</div>
        <h3 className={subheadCls}>EvaluationEngine</h3>
        <p className={`${textCls} mb-3`}>Orchestrates the evaluation process.</p>
        <CodeBlock language="python" code={`from beyondbench import EvaluationEngine

engine = EvaluationEngine(
    model_handler=model,
    output_dir="./results",
    store_details=True
)

# Run full evaluation
results = engine.run_evaluation(
    suite="all",        # "easy", "medium", "hard", or "all"
    datapoints=100,     # test cases per task
    temperature=0.7,
    max_tokens=512,
    seed=42
)

# Access results
print(results['summary'])          # Aggregate metrics
print(results['task_results'])     # Per-task breakdown`} />
      </div>
      <div className={cardCls}>
        <div className={labelCls}>Core Class</div>
        <h3 className={subheadCls}>TaskRegistry</h3>
        <p className={`${textCls} mb-3`}>Discover and access available tasks.</p>
        <CodeBlock language="python" code={`from beyondbench.core.task_registry import TaskRegistry

registry = TaskRegistry()

# List all tasks
all_tasks = registry.get_available_tasks()

# Get tasks by suite
easy_tasks = registry.get_tasks_for_suite("easy")
hard_tasks = registry.get_tasks_for_suite("hard")

# Get suite statistics
stats = registry.get_suite_stats()
print(stats)  # {'easy': 29, 'medium': 5, 'hard': 10}

# Get a specific task class
SumTask = registry.get_task_class("sum_task")`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Parsing Utilities</h3>
        <CodeBlock language="python" code={`from beyondbench.utils.parsing import parse_boxed_answer, extract_number, extract_list

# Parse LaTeX boxed answers
answer = parse_boxed_answer(r"The answer is \\boxed{42}")  # Returns 42

# Extract numbers from text
num = extract_number("The result is 3.14")  # Returns 3.14

# Extract lists from text
lst = extract_list("The sorted list is [1, 2, 3, 4, 5]")  # Returns [1,2,3,4,5]`} />
      </div>
    </div>
  )
}

function BackendsContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Backend Overview</h2>
        <p className={textCls}>BeyondBench supports 5 inference backends for maximum flexibility.</p>
      </div>
      <div className={cardCls}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className={`border-b ${isDark ? 'border-bb-dark-50/20' : 'border-gray-200'}`}>
                {['Backend', 'Models', 'Features', 'Setup'].map(h => (
                  <th key={h} className={`text-left py-2 px-3 text-xs font-semibold ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              {[
                ['OpenAI', 'GPT-4o, GPT-5, o3, o4-Mini', 'Reasoning effort control', 'OPENAI_API_KEY'],
                ['Gemini', 'Gemini 2.5 Pro/Flash', 'Thinking budget config', 'GEMINI_API_KEY'],
                ['Anthropic', 'Claude Sonnet 4, Opus 4', 'Latest Claude models', 'ANTHROPIC_API_KEY'],
                ['vLLM', 'Any HuggingFace model', 'Batch, tensor parallel, high throughput', 'pip install beyondbench[vllm]'],
                ['Transformers', 'Any HuggingFace model', 'CPU/GPU, auto device map', 'Included by default'],
              ].map(([name, models, features, setup]) => (
                <tr key={name} className={`border-b ${isDark ? 'border-bb-dark-50/10' : 'border-gray-100'}`}>
                  <td className={`py-2 px-3 font-semibold ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>{name}</td>
                  <td className="py-2 px-3">{models}</td>
                  <td className="py-2 px-3">{features}</td>
                  <td className="py-2 px-3 font-mono text-[10px]">{setup}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Environment Variables</h3>
        <CodeBlock code={`# API Provider Keys
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."

# For gated HuggingFace models (e.g., Llama, Mistral)
export HF_TOKEN="hf_..."

# GPU selection for local models
export CUDA_VISIBLE_DEVICES="0,1"  # Use specific GPUs`} />
      </div>
    </div>
  )
}

function TaskSuitesContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Task Suites</h2>
        <p className={textCls}>BeyondBench organizes 79 tasks into three difficulty suites with 138 total variations.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Easy Suite (29 Tasks)</h3>
        <p className={textCls}>Fundamental arithmetic and statistical operations with scalable complexity.</p>
        <div className={`mt-3 grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Arithmetic:</span> sum, multiplication, subtraction, division, absolute_difference</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Statistics:</span> mean, median, mode</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Counting:</span> odd_count, even_count, count_negative, count_unique, count_greater_than_previous, count_palindromic, count_perfect_squares, count_multiples, local_maxima_count</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Extrema:</span> find_maximum, find_minimum, second_maximum, range, index_of_maximum, max_adjacent_difference, sum_of_max_indices</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Sequences:</span> sorting, longest_increasing_subsequence, alternating_sum, sum_of_digits</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Comparison:</span> comparison</div>
        </div>
        <CodeBlock code="beyondbench evaluate --model-id MODEL --suite easy --datapoints 100" />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Medium Suite (5 Tasks, 49 Variations)</h3>
        <p className={textCls}>Sequence patterns and recursive reasoning requiring multi-step problem solving.</p>
        <div className={`mt-3 space-y-1 text-xs font-mono ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>fibonacci_sequence</span> - 6 variations (Tribonacci, Lucas, modified recursive)</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>algebraic_sequence</span> - 10 variations (polynomial, arithmetic, quadratic)</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>geometric_sequence</span> - 10 variations (exponential, compound, factorial)</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>prime_sequence</span> - 11 variations (prime gaps, twin primes, Sophie Germain)</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>complex_pattern</span> - 12 variations (interleaved, conditional, multi-rule)</div>
        </div>
        <CodeBlock code="beyondbench evaluate --model-id MODEL --suite medium --datapoints 100" />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Hard Suite (10 Tasks, 68 Variations)</h3>
        <p className={textCls}>NP-complete and constraint satisfaction problems testing advanced reasoning.</p>
        <div className={`mt-3 space-y-1 text-xs font-mono ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>tower_hanoi</span> - 6 variations | O(2^n) computational complexity</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>n_queens</span> - 4 variations | NP-complete board placement</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>graph_coloring</span> - 10 variations | Chromatic number computation</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>boolean_sat</span> - 5 variations | 2-SAT, 3-SAT, Horn clauses</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>sudoku</span> - 8 variations | Standard, diagonal, irregular</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>cryptarithmetic</span> - 12 variations | Letter-to-digit mapping</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>matrix_chain</span> - 5 variations | Dynamic programming optimization</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>modular_systems</span> - 5 variations | Chinese Remainder Theorem</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>constraint_optimization</span> - 5 variations | Knapsack, scheduling</div>
          <div><span className={isDark ? 'text-gray-400' : 'text-gray-600'}>logic_grid_puzzles</span> - 8 variations | Einstein puzzles, deductive reasoning</div>
        </div>
        <CodeBlock code="beyondbench evaluate --model-id MODEL --suite hard --datapoints 50" />
      </div>
    </div>
  )
}

function AdvancedEvalContent({ isDark, cardCls, headCls, subheadCls, textCls, inlineCodeCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Advanced Evaluation</h2>
        <p className={textCls}>Fine-tune your evaluation strategy for optimal results.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Scalable Complexity Testing</h3>
        <p className={textCls}>Test how model performance scales with problem size:</p>
        <CodeBlock code={`# Test with different list sizes
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --list-sizes "8,16,32,64,128" \\
  --suite easy --datapoints 100

# Adjust number ranges
beyondbench evaluate --model-id MODEL \\
  --range-min -1000 --range-max 1000 \\
  --suite easy --datapoints 100`} />
        <Callout type="tip">
          Scaling list sizes reveals whether models use genuine algorithms or rely on pattern matching. Performance should degrade gracefully for true reasoning.
        </Callout>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Cross-Validation Folds</h3>
        <CodeBlock code={`beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --suite all --datapoints 100 --folds 5 \\
  --seed 42 --store-details`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Comparing Multiple Models</h3>
        <CodeBlock language="python" code={`from beyondbench import EvaluationEngine
from beyondbench.models.model_handler import ModelHandler

models = {
    "gpt-4o": ModelHandler(model_id="gpt-4o", api_provider="openai"),
    "claude-sonnet": ModelHandler(model_id="claude-sonnet-4-20250514", api_provider="anthropic"),
    "gemini-pro": ModelHandler(model_id="gemini-2.5-pro", api_provider="gemini"),
}

all_results = {}
for name, model in models.items():
    engine = EvaluationEngine(model_handler=model, output_dir=f"./results/{name}")
    all_results[name] = engine.run_evaluation(suite="all", datapoints=100, seed=42)

# Compare results
print(f"{'Model':<20} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>8}")
print("-" * 52)
for name, results in all_results.items():
    summary = results['summary']
    print(f"{name:<20} {summary.get('easy_acc', 0):>7.1%} {summary.get('medium_acc', 0):>7.1%} {summary.get('hard_acc', 0):>7.1%} {summary['avg_accuracy']:>7.1%}")`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Cost-Efficient Evaluation Strategy</h3>
        <div className={`space-y-2 text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          {[
            ['Step 1:', 'Run easy suite with 50 datapoints to verify setup'],
            ['Step 2:', 'Run all suites with 20 datapoints for quick overview'],
            ['Step 3:', 'Scale up to 100+ datapoints for publishable results'],
            ['Step 4:', 'Use --store-details to analyze failures without re-running'],
          ].map(([step, desc]) => (
            <div key={step}><strong className={isDark ? 'text-gray-300' : 'text-gray-700'}>{step}</strong> {desc}</div>
          ))}
        </div>
      </div>
    </div>
  )
}

function ConfigurationContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Configuration</h2>
        <p className={textCls}>Configure BeyondBench via YAML or JSON files for reproducible evaluations.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>YAML Configuration File</h3>
        <CodeBlock language="yaml" code={`model:
  id: gpt-4o
  provider: openai
  # api_key: sk-...  # Or use environment variable

evaluation:
  suite: all          # easy, medium, hard, or all
  datapoints: 100     # test cases per task
  temperature: 0.7    # generation temperature
  max_tokens: 512     # max generation tokens
  seed: 42            # random seed
  num_folds: 1        # evaluation folds

output:
  dir: ./results
  store_details: true # save per-example results`} />
        <p className={textCls}>Run with:</p>
        <CodeBlock code="beyondbench run-config config.yaml" />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Multi-Model Batch Configuration</h3>
        <CodeBlock language="yaml" code={`models:
  - id: gpt-4o
    provider: openai
  - id: claude-sonnet-4-20250514
    provider: anthropic
  - id: gemini-2.5-pro
    provider: gemini

evaluation:
  suite: all
  datapoints: 100
  seed: 42

output:
  dir: ./comparison_results
  store_details: true`} />
      </div>
    </div>
  )
}

function ExtendingContent({ isDark, cardCls, headCls, subheadCls, textCls, inlineCodeCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Extending BeyondBench</h2>
        <p className={textCls}>Add custom tasks and backends to the framework.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Adding Custom Tasks</h3>
        <p className={textCls}>Create new tasks by extending the <code className={inlineCodeCls}>BaseTask</code> class:</p>
        <CodeBlock language="python" code={`from beyondbench.core.base_task import BaseTask

class MyCustomTask(BaseTask):
    @property
    def task_name(self):
        return "my_custom_task"

    @property
    def task_suite(self):
        return "easy"  # or "medium", "hard"

    def generate_data(self, num_samples=100, **kwargs):
        """Generate problem instances with ground truth."""
        data = []
        for _ in range(num_samples):
            numbers = self.rng.randint(1, 100, size=5).tolist()
            answer = sum(numbers)
            data.append({
                "input": numbers,
                "ground_truth": answer
            })
        return data

    def create_prompt(self, data_point):
        """Create the evaluation prompt."""
        nums = data_point["input"]
        return f"Calculate the sum of {nums}. Answer with just the number."

    def evaluate_response(self, response, data_point):
        """Evaluate the model's response."""
        from beyondbench.utils.parsing import extract_number
        extracted = extract_number(response)
        correct = extracted == data_point["ground_truth"]
        return {
            "accuracy": 1.0 if correct else 0.0,
            "extracted_answer": extracted,
            "ground_truth": data_point["ground_truth"]
        }`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Task Registration</h3>
        <p className={textCls}>Place your task file in the appropriate suite directory:</p>
        <CodeBlock code={`beyondbench/tasks/easy/my_custom_task.py    # For easy tasks
beyondbench/tasks/medium/my_task.py         # For medium tasks
beyondbench/tasks/hard/my_task.py           # For hard tasks`} />
        <p className={`text-xs mt-2 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>Tasks are auto-discovered by the <code className={inlineCodeCls}>TaskRegistry</code> when placed in the correct directory.</p>
      </div>
    </div>
  )
}

function OutputContent({ isDark, cardCls, headCls, subheadCls, textCls, pypiVersion }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Output Format</h2>
        <p className={textCls}>Understanding BeyondBench's result structure.</p>
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Results Structure</h3>
        <CodeBlock language="json" code={`{
  "summary": {
    "model_id": "gpt-4o",
    "suite": "all",
    "avg_accuracy": 0.5701,
    "avg_instruction_following": 0.9659,
    "avg_tokens": 458.45,
    "total_tasks": 44,
    "total_datapoints": 4400,
    "easy_acc": 0.8210,
    "medium_acc": 0.5540,
    "hard_acc": 0.2130
  },
  "task_results": {
    "sum": { "accuracy": 0.95, "instruction_following": 1.0, "suite": "easy" },
    "sudoku": { "accuracy": 0.12, "instruction_following": 0.98, "suite": "hard" }
  },
  "metadata": {
    "version": "${pypiVersion || '0.1.0'}",
    "timestamp": "2025-02-05T10:30:00Z",
    "seed": 42
  }
}`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Output Directory Structure</h3>
        <CodeBlock code={`beyondbench_results/
  gpt-4o/
    summary.json          # Aggregate results
    easy/
      sum_results.json    # Per-task results
      sum_details.json    # Per-example details (if --store-details)
    medium/
      fibonacci_results.json
    hard/
      sudoku_results.json
      n_queens_results.json`} />
      </div>
    </div>
  )
}

function TroubleshootingContent({ isDark, cardCls, headCls, subheadCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Troubleshooting</h2>
        <p className={textCls}>Solutions to common issues.</p>
      </div>
      {[
        { title: 'CUDA out of memory', desc: 'Model too large for available GPU memory.', code: `# Solution 1: Use tensor parallelism\nbeyondbench evaluate --model-id MODEL --backend vllm --tensor-parallel-size 2\n\n# Solution 2: Reduce GPU memory utilization\nbeyondbench evaluate --model-id MODEL --backend vllm --gpu-memory-utilization 0.85\n\n# Solution 3: Use a quantized model\nbeyondbench evaluate --model-id MODEL-GPTQ --backend vllm` },
        { title: 'API Rate Limit Errors', desc: 'Too many API requests in a short time.', code: `# Increase timeout and retries\nbeyondbench evaluate --model-id gpt-4o --api-provider openai \\\n  --max-retries 5 --timeout 600\n\n# Reduce datapoints for initial testing\nbeyondbench evaluate --model-id gpt-4o --api-provider openai \\\n  --datapoints 20 --suite easy` },
        { title: 'Import Errors', desc: 'Missing dependencies or incorrect installation.', code: `# Reinstall with all dependencies\npip install --force-reinstall beyondbench[full]\n\n# Verify installation\npython -c "from beyondbench import EvaluationEngine, TaskRegistry; print('OK')"` },
        { title: 'vLLM Model Loading Failures', desc: 'Model architecture not supported or trust_remote_code needed.', code: `# Enable remote code execution\nbeyondbench evaluate --model-id MODEL --backend vllm --trust-remote-code\n\n# Try transformers backend as fallback\nbeyondbench evaluate --model-id MODEL --backend transformers` },
      ].map(issue => (
        <div key={issue.title} className={cardCls}>
          <div className="text-xs font-semibold text-red-400 mb-1">{issue.title}</div>
          <p className={`text-xs mb-2 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{issue.desc}</p>
          <CodeBlock code={issue.code} />
        </div>
      ))}
      <div className={cardCls}>
        <h3 className={subheadCls}>Debug Mode</h3>
        <CodeBlock code={`# Enable debug logging for detailed output
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --log-level DEBUG --suite easy --datapoints 5`} />
      </div>
      <div className={cardCls}>
        <h3 className={subheadCls}>Getting Help</h3>
        <div className={`text-xs space-y-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          <div>Report issues: <a href="https://github.com/ctrl-gaurav/BeyondBench/issues" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>GitHub Issues</a></div>
          <div>Email: <a href="mailto:gks@vt.edu" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>gks@vt.edu</a></div>
          <div>Paper: <a href="https://arxiv.org/abs/2509.24210" className={`${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'} hover:underline`}>arXiv:2509.24210</a></div>
        </div>
      </div>
    </div>
  )
}

function CitationContent({ isDark, cardCls, headCls, textCls }) {
  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <h2 className={headCls}>Citation</h2>
        <p className={textCls}>If you use BeyondBench in your research, please cite our paper (accepted at <strong className={isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}>ICLR 2026</strong>):</p>
        <CodeBlock language="bibtex" code={`@misc{srivastava2025beyondbenchbenchmarkfreeevaluationreasoning,
      title={BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models},
      author={Gaurav Srivastava and Aafiya Hussain and Zhenyu Bi and Swastik Roy and Priya Pitre and Meng Lu and Morteza Ziyadi and Xuan Wang},
      year={2025},
      eprint={2509.24210},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.24210},
}`} />
      </div>
    </div>
  )
}

/* ============ MAIN COMPONENT ============ */

export default function Documentation() {
  const [activeSection, setActiveSection] = useState('overview')
  const [searchQuery, setSearchQuery] = useState('')
  const { isDark } = useTheme()
  const pypiVersion = usePyPIVersion()

  const cardCls = `p-6 sm:p-8 rounded-2xl ${isDark ? 'glass-card' : 'bg-white/70 backdrop-blur-xl border border-gray-200/60 shadow-sm'}`
  const headCls = `text-xl sm:text-2xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`
  const subheadCls = `text-lg font-semibold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`
  const textCls = `text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`
  const labelCls = `text-xs font-mono uppercase tracking-wider mb-2 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`
  const inlineCodeCls = `font-mono text-xs px-1.5 py-0.5 rounded-md ${isDark ? 'bg-bb-dark-400/60 text-bb-accent border border-bb-dark-50/20' : 'bg-gray-100 text-bb-accent-dark border border-gray-200'}`

  const contentProps = { isDark, cardCls, headCls, subheadCls, textCls, labelCls, inlineCodeCls, pypiVersion }

  const sectionContent = {
    'overview': <OverviewContent {...contentProps} />,
    'installation': <InstallationContent {...contentProps} />,
    'quickstart': <QuickStartContent {...contentProps} />,
    'eval-openai': <EvalOpenAIContent {...contentProps} />,
    'eval-gemini': <EvalGeminiContent {...contentProps} />,
    'eval-anthropic': <EvalAnthropicContent {...contentProps} />,
    'eval-vllm': <EvalVllmContent {...contentProps} />,
    'eval-transformers': <EvalTransformersContent {...contentProps} />,
    'cli': <CLIContent {...contentProps} />,
    'python-api': <PythonAPIContent {...contentProps} />,
    'backends': <BackendsContent {...contentProps} />,
    'tasks': <TaskSuitesContent {...contentProps} />,
    'advanced-eval': <AdvancedEvalContent {...contentProps} />,
    'configuration': <ConfigurationContent {...contentProps} />,
    'extending': <ExtendingContent {...contentProps} />,
    'output': <OutputContent {...contentProps} />,
    'troubleshooting': <TroubleshootingContent {...contentProps} />,
    'citation': <CitationContent {...contentProps} />,
  }

  const filteredItems = useMemo(() => {
    if (!searchQuery.trim()) return NAV_ITEMS
    const q = searchQuery.toLowerCase()
    return NAV_ITEMS.filter(item => item.label.toLowerCase().includes(q) || item.id.toLowerCase().includes(q))
  }, [searchQuery])

  return (
    <section className="pt-24 pb-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <span className={`inline-block px-4 py-1.5 rounded-full text-xs font-semibold tracking-wider uppercase mb-4 ${
            isDark ? 'bg-bb-accent/10 text-bb-accent border border-bb-accent/20' : 'bg-bb-accent-dark/10 text-bb-accent-dark border border-bb-accent-dark/20'
          }`}>Documentation</span>
          <h1 className={`text-3xl sm:text-4xl lg:text-5xl font-bold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            BeyondBench Docs
          </h1>
          <p className={`text-lg max-w-2xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Everything you need to evaluate reasoning in language models.
          </p>
        </div>

        {/* Search */}
        <div className="max-w-md mx-auto mb-10">
          <div className={`relative rounded-xl transition-all duration-300 ${
            isDark
              ? 'bg-bb-dark-400/30 border border-bb-dark-50/20 focus-within:border-bb-accent/25'
              : 'bg-white border border-gray-200 focus-within:border-bb-accent-dark/40 shadow-sm'
          }`}>
            <Search className={`absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
            <input
              type="text"
              placeholder="Search documentation..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`w-full pl-10 pr-4 py-2.5 rounded-xl text-sm bg-transparent outline-none ${
                isDark ? 'text-white placeholder:text-gray-600' : 'text-gray-900 placeholder:text-gray-400'
              }`}
            />
          </div>
        </div>

        {/* Mobile section selector */}
        <div className="lg:hidden mb-6">
          <div className="relative">
            <select
              value={activeSection}
              onChange={(e) => { setActiveSection(e.target.value); window.scrollTo({ top: 0, behavior: 'smooth' }) }}
              className={`w-full px-4 py-2.5 rounded-xl text-sm font-medium appearance-none ${
                isDark
                  ? 'bg-bb-dark-400/30 text-white border border-bb-dark-50/20'
                  : 'bg-white text-gray-900 border border-gray-200 shadow-sm'
              }`}
            >
              {NAV_ITEMS.map(item => (
                <option key={item.id} value={item.id}>{item.label}</option>
              ))}
            </select>
            <ChevronDown className={`absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none ${isDark ? 'text-gray-500' : 'text-gray-400'}`} />
          </div>
        </div>

        {/* Layout: sidebar + content */}
        <div className="flex gap-8">
          {/* Sidebar */}
          <nav className={`hidden lg:block w-56 shrink-0 sticky top-24 self-start rounded-xl p-2 max-h-[calc(100vh-7rem)] overflow-y-auto ${
            isDark
              ? 'bg-bb-dark-400/20 border border-bb-dark-50/10'
              : 'bg-white/60 border border-gray-200/30 shadow-sm'
          }`}>
            <div className="space-y-0.5">
              {filteredItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => { setActiveSection(item.id); window.scrollTo({ top: 0, behavior: 'smooth' }) }}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all duration-200 flex items-center gap-2.5 ${
                    activeSection === item.id
                      ? isDark
                        ? 'bg-bb-accent/10 text-bb-accent font-semibold'
                        : 'bg-bb-accent-dark/10 text-bb-accent-dark font-semibold'
                      : isDark
                        ? 'text-gray-500 hover:text-gray-300 hover:bg-bb-dark-300/30'
                        : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d={item.iconPath} />
                  </svg>
                  {item.label}
                </button>
              ))}
            </div>
          </nav>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div
              key={activeSection}
              className="animate-fade-in"
            >
              {sectionContent[activeSection]}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

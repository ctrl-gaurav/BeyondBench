import { useState, useEffect, useRef, useCallback } from 'react'
import { BookOpen, Terminal, Code, Settings, Layers, Cpu, Package, FileText, ChevronRight, Hexagon, Copy, Check, Zap, Database, GitBranch, Shield, BarChart3, Wrench, Globe, AlertTriangle, Lightbulb, ArrowUp } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'
import { usePyPIVersion } from '../hooks/usePyPIVersion'

function CodeBlock({ code, language = 'bash' }) {
  const [copied, setCopied] = useState(false)
  const { isDark } = useTheme()
  function handleCopy() {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <div className="relative group my-3">
      <div className={`flex items-center justify-between px-4 py-2 rounded-t-lg border border-b-0 ${
        isDark ? 'bg-bb-dark-600/80 border-bb-dark-50/20' : 'bg-gray-100 border-gray-200'
      }`}>
        <span className={`text-[10px] uppercase tracking-wider font-mono ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>{language}</span>
        <button
          onClick={handleCopy}
          className={`transition-all duration-300 ${
            copied
              ? isDark ? 'text-bb-accent scale-110' : 'text-bb-accent-dark scale-110'
              : isDark ? 'text-gray-600 hover:text-bb-accent hover:scale-110' : 'text-gray-400 hover:text-bb-accent-dark hover:scale-110'
          } ${copied ? 'copy-success' : ''}`}
        >
          {copied ? <Check className={`w-3.5 h-3.5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} /> : <Copy className="w-3.5 h-3.5" />}
        </button>
      </div>
      <pre className={`rounded-b-lg p-4 overflow-x-auto text-sm font-mono leading-relaxed border transition-all duration-300 group-hover:shadow-lg ${
        isDark
          ? 'bg-bb-dark-500/80 border-bb-dark-50/20 text-gray-300 group-hover:border-bb-accent/10'
          : 'bg-gray-50 border-gray-200 text-gray-700 group-hover:border-bb-accent-dark/10'
      }`}>
        <code>{code}</code>
      </pre>
    </div>
  )
}

function Section({ id, icon: Icon, title, children }) {
  const { isDark } = useTheme()
  const ref = useRef(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setVisible(true)
        observer.unobserve(el)
      }
    }, { threshold: 0.05 })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <section
      ref={ref}
      id={id}
      className={`mb-12 scroll-mt-24 transition-all duration-700 ease-out ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
      }`}
    >
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 hover:scale-110 ${isDark ? 'bg-bb-accent/10' : 'bg-bb-accent-dark/10'}`}>
          <Icon className={`w-4 h-4 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
        </div>
        <h2 className={`text-xl font-bold tracking-wide ${isDark ? 'text-white' : 'text-gray-900'}`}>{title}</h2>
      </div>
      <div className={`leading-relaxed text-sm space-y-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>{children}</div>
    </section>
  )
}

function SubSection({ title, children }) {
  const { isDark } = useTheme()
  return (
    <div className="mt-6">
      <h3 className={`text-base font-semibold mb-3 tracking-wide ${isDark ? 'text-gray-200' : 'text-gray-800'}`}>{title}</h3>
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
  return (
    <div className={`border-l-2 ${styles[type]} rounded-r-lg p-3 flex gap-2 text-xs transition-all duration-300 hover:translate-x-1 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
      {icons[type]}
      <div>{children}</div>
    </div>
  )
}

function BackToTop({ isDark }) {
  const [show, setShow] = useState(false)

  useEffect(() => {
    const handleScroll = () => setShow(window.scrollY > 400)
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  if (!show) return null

  return (
    <button
      onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      className={`back-to-top ${
        isDark
          ? 'bg-bb-accent/20 text-bb-accent border border-bb-accent/30 hover:bg-bb-accent/30'
          : 'bg-bb-accent-dark/20 text-bb-accent-dark border border-bb-accent-dark/30 hover:bg-bb-accent-dark/30'
      }`}
    >
      <ArrowUp className="w-4 h-4" />
    </button>
  )
}

const NAV_ITEMS = [
  { id: 'overview', icon: BookOpen, label: 'Overview' },
  { id: 'installation', icon: Package, label: 'Installation' },
  { id: 'quickstart', icon: Zap, label: 'Quick Start' },
  { id: 'eval-openai', icon: Globe, label: 'OpenAI Evaluation' },
  { id: 'eval-gemini', icon: Globe, label: 'Gemini Evaluation' },
  { id: 'eval-anthropic', icon: Globe, label: 'Anthropic Evaluation' },
  { id: 'eval-vllm', icon: Cpu, label: 'vLLM Evaluation' },
  { id: 'eval-transformers', icon: Cpu, label: 'Transformers Evaluation' },
  { id: 'cli', icon: Terminal, label: 'CLI Reference' },
  { id: 'python-api', icon: Code, label: 'Python API' },
  { id: 'backends', icon: Cpu, label: 'Backend Overview' },
  { id: 'tasks', icon: Layers, label: 'Task Suites' },
  { id: 'advanced-eval', icon: BarChart3, label: 'Advanced Evaluation' },
  { id: 'configuration', icon: Settings, label: 'Configuration' },
  { id: 'extending', icon: GitBranch, label: 'Extending' },
  { id: 'output', icon: Database, label: 'Output Format' },
  { id: 'troubleshooting', icon: Wrench, label: 'Troubleshooting' },
  { id: 'citation', icon: FileText, label: 'Citation' },
]

export default function Documentation() {
  const [activeSection, setActiveSection] = useState('overview')
  const { isDark } = useTheme()
  const pypiVersion = usePyPIVersion('beyondbench')

  // Scroll spy using IntersectionObserver
  useEffect(() => {
    const sectionIds = NAV_ITEMS.map(item => item.id)
    const observers = []

    sectionIds.forEach(id => {
      const el = document.getElementById(id)
      if (!el) return
      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setActiveSection(id)
          }
        },
        { rootMargin: '-20% 0px -70% 0px', threshold: 0 }
      )
      observer.observe(el)
      observers.push(observer)
    })

    return () => observers.forEach(obs => obs.disconnect())
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:pl-64">
      {/* Sidebar - fixed full-height panel with scroll spy */}
      <nav className={`hidden lg:flex flex-col fixed left-0 top-16 bottom-0 w-56 z-40 border-r px-4 pt-8 pb-6 overflow-y-auto ${
        isDark
          ? 'bg-bb-dark-500/90 backdrop-blur-xl border-bb-dark-50/20'
          : 'bg-white/90 backdrop-blur-xl border-bb-light-300/50'
      }`}>
        <div className="flex items-center gap-2 mb-6 group">
          <Hexagon className={`w-5 h-5 transition-all duration-500 group-hover:rotate-[30deg] ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
          <span className={`text-sm font-bold tracking-wide ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Documentation</span>
        </div>
        <ul className="space-y-0.5 flex-1">
          {NAV_ITEMS.map(item => (
            <li key={item.id}>
              <button
                onClick={() => {
                  setActiveSection(item.id)
                  const el = document.getElementById(item.id)
                  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
                }}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-300 text-left scroll-spy-item ${
                  activeSection === item.id
                    ? isDark
                      ? 'bg-bb-accent/10 text-bb-accent shadow-[0_0_10px_rgba(0,230,118,0.08)] active'
                      : 'bg-bb-accent-dark/10 text-bb-accent-dark active'
                    : isDark ? 'text-gray-500 hover:text-gray-300 hover:bg-bb-dark-300/30' : 'text-gray-500 hover:text-gray-700 hover:bg-bb-light-200'
                }`}
              >
                <item.icon className={`w-3.5 h-3.5 shrink-0 transition-transform duration-300 ${activeSection === item.id ? 'scale-110' : ''}`} />
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* Back to top button */}
      <BackToTop isDark={isDark} />

      {/* Content */}
      <div className="min-w-0">
          {/* Header */}
          <div className="glass-card p-6 mb-8 shine-effect">
            <div className="flex items-center gap-3 mb-3">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300 hover:scale-110 ${
                isDark ? 'bg-gradient-to-br from-bb-accent/20 to-bb-teal/20' : 'bg-gradient-to-br from-bb-accent-dark/15 to-bb-teal/15'
              }`}>
                <BookOpen className={`w-5 h-5 ${isDark ? 'text-bb-accent' : 'text-bb-accent-dark'}`} />
              </div>
              <div>
                <h1 className={`text-2xl font-bold tracking-tight ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  BeyondBench <span className="gradient-text-green">Documentation</span>
                </h1>
                <p className={`text-xs font-mono ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>v{pypiVersion || '...'} &bull; pip install beyondbench</p>
              </div>
            </div>
            <p className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Complete documentation for the BeyondBench evaluation framework. BeyondBench provides benchmark-free evaluation of reasoning capabilities in language models using dynamic algorithmic problem generation across 44 tasks, 117 variations, and over 10^15 unique instances.
            </p>
          </div>

          {/* Overview */}
          <Section id="overview" icon={BookOpen} title="Overview">
            <p>
              <strong className="text-gray-200">BeyondBench</strong> is a contamination-resistant evaluation framework for assessing reasoning capabilities in Large Language Models. Unlike traditional benchmarks that risk contamination from internet-scale training data, BeyondBench uses <strong className="text-bb-accent">algorithmic problem generation</strong> to create mathematically grounded problems on the fly.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mt-4">
              <div className="glass-card p-4">
                <div className="text-sm font-semibold text-gray-200 mb-1">44 Tasks</div>
                <div className="text-xs text-gray-500">29 easy + 5 medium + 10 hard difficulty tasks covering arithmetic, sequences, constraint satisfaction, and NP-complete problems.</div>
              </div>
              <div className="glass-card p-4">
                <div className="text-sm font-semibold text-gray-200 mb-1">117 Variations</div>
                <div className="text-xs text-gray-500">Fine-grained task variations for nuanced capability evaluation across different problem formulations.</div>
              </div>
              <div className="glass-card p-4">
                <div className="text-sm font-semibold text-gray-200 mb-1">&gt;10^15 Instances</div>
                <div className="text-xs text-gray-500">Each task generates from a problem space of over 10^15 unique instances, ensuring contamination resistance.</div>
              </div>
              <div className="glass-card p-4">
                <div className="text-sm font-semibold text-gray-200 mb-1">5 Backends</div>
                <div className="text-xs text-gray-500">vLLM, Transformers, OpenAI, Gemini, and Anthropic APIs. Evaluate any model from any provider.</div>
              </div>
            </div>
            <p className="mt-4 text-xs text-gray-500">
              Paper accepted at <strong className="text-bb-accent">ICLR 2026</strong>. Read the full paper at <a href="https://arxiv.org/abs/2509.24210" target="_blank" rel="noopener noreferrer" className="text-bb-accent hover:underline">arXiv:2509.24210</a>.
            </p>
          </Section>

          {/* Installation */}
          <Section id="installation" icon={Package} title="Installation">
            <SubSection title="From PyPI (Recommended)">
              <CodeBlock code="pip install beyondbench" />
            </SubSection>

            <SubSection title="From Source">
              <CodeBlock code={`git clone https://github.com/ctrl-gaurav/BeyondBench.git
cd BeyondBench
pip install -e .`} />
            </SubSection>

            <SubSection title="With Optional Dependencies">
              <CodeBlock code={`# All API clients (OpenAI, Gemini, Anthropic)
pip install beyondbench[all-apis]

# vLLM support (requires CUDA)
pip install beyondbench[vllm]

# Development tools
pip install beyondbench[dev]

# Everything included
pip install beyondbench[full]`} />
            </SubSection>

            <SubSection title="Interactive Setup">
              <p>For first-time users, use the interactive setup script:</p>
              <CodeBlock code="./setup_beyondbench.sh" />
              <p>This will create a conda environment, install dependencies, verify imports, and run tests.</p>
            </SubSection>

            <SubSection title="Verify Installation">
              <CodeBlock code={`# Check version
beyondbench --version

# Verify all imports work
python -c "from beyondbench import EvaluationEngine, TaskRegistry; print('OK')"

# List all available tasks
beyondbench list-tasks`} />
            </SubSection>

            <SubSection title="Requirements">
              <ul className="list-disc list-inside text-gray-500 space-y-1 text-xs">
                <li>Python 3.8+</li>
                <li>PyTorch 2.0+</li>
                <li>Transformers 4.30+</li>
                <li>CUDA (for vLLM and local GPU inference)</li>
              </ul>
            </SubSection>
          </Section>

          {/* Quick Start */}
          <Section id="quickstart" icon={Zap} title="Quick Start">
            <SubSection title="Interactive Wizard">
              <p>Launch the interactive wizard for guided setup:</p>
              <CodeBlock code="beyondbench" />
              <p>The wizard walks you through model selection, API key configuration, task suite selection, and parameter tuning.</p>
            </SubSection>

            <SubSection title="Command Line">
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
            </SubSection>

            <SubSection title="Python API">
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
            </SubSection>
          </Section>

          {/* ===== EVALUATION GUIDES ===== */}

          {/* OpenAI Evaluation */}
          <Section id="eval-openai" icon={Globe} title="Evaluating OpenAI Models">
            <p>Evaluate GPT-4, GPT-4o, GPT-5, o1, o3, o4-Mini, and other OpenAI models.</p>

            <SubSection title="Setup">
              <CodeBlock code={`# Set your API key
export OPENAI_API_KEY="sk-..."

# Or pass it directly via CLI
beyondbench evaluate --model-id gpt-4o --api-provider openai --api-key "sk-..."`} />
            </SubSection>

            <SubSection title="Evaluate GPT-4o (All Suites)">
              <CodeBlock code={`beyondbench evaluate \\
  --model-id gpt-4o \\
  --api-provider openai \\
  --suite all \\
  --datapoints 100 \\
  --temperature 0.7 \\
  --max-tokens 512 \\
  --output-dir ./results/gpt-4o \\
  --store-details`} />
            </SubSection>

            <SubSection title="Evaluate GPT-5 with Reasoning Effort">
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
                Use <code className="text-bb-accent font-mono">--reasoning-effort high</code> for hard tasks to get the best accuracy. Use <code className="text-bb-accent font-mono">minimal</code> for cost-efficient easy task evaluation.
              </Callout>
            </SubSection>

            <SubSection title="Evaluate o3 / o4-Mini Reasoning Models">
              <CodeBlock code={`# o3 - reasoning model
beyondbench evaluate --model-id o3 --api-provider openai \\
  --suite hard --datapoints 50 --max-tokens 4096

# o4-mini - lightweight reasoning
beyondbench evaluate --model-id o4-mini --api-provider openai \\
  --suite all --datapoints 100`} />
            </SubSection>

            <SubSection title="Python API">
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
            </SubSection>

            <SubSection title="Supported OpenAI Models">
              <div className="glass-card p-4 text-xs">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 font-mono text-gray-500">
                  <div>gpt-4o</div>
                  <div>gpt-4o-mini</div>
                  <div>gpt-4-turbo</div>
                  <div>gpt-5</div>
                  <div>o1</div>
                  <div>o1-mini</div>
                  <div>o3</div>
                  <div>o3-mini</div>
                  <div>o4-mini</div>
                </div>
              </div>
            </SubSection>
          </Section>

          {/* Gemini Evaluation */}
          <Section id="eval-gemini" icon={Globe} title="Evaluating Google Gemini Models">
            <p>Evaluate Gemini 2.0 Flash, Gemini 2.5 Pro, Gemini Ultra, and other Google AI models.</p>

            <SubSection title="Setup">
              <CodeBlock code={`# Set your Gemini API key
export GEMINI_API_KEY="AI..."

# Or use Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"`} />
            </SubSection>

            <SubSection title="Evaluate Gemini 2.0 Flash">
              <CodeBlock code={`beyondbench evaluate \\
  --model-id gemini-2.0-flash \\
  --api-provider gemini \\
  --suite all \\
  --datapoints 100 \\
  --thinking-budget 1024 \\
  --output-dir ./results/gemini-flash`} />
            </SubSection>

            <SubSection title="Evaluate Gemini 2.5 Pro with Extended Thinking">
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
                Increase <code className="text-bb-accent font-mono">--thinking-budget</code> for hard tasks like Sudoku, N-Queens, and Boolean SAT. Lower budgets work well for easy arithmetic tasks.
              </Callout>
            </SubSection>

            <SubSection title="Python API">
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
            </SubSection>

            <SubSection title="Supported Gemini Models">
              <div className="glass-card p-4 text-xs">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 font-mono text-gray-500">
                  <div>gemini-2.0-flash</div>
                  <div>gemini-2.0-flash-lite</div>
                  <div>gemini-2.5-pro</div>
                  <div>gemini-2.5-flash</div>
                  <div>gemini-1.5-pro</div>
                  <div>gemini-1.5-flash</div>
                </div>
              </div>
            </SubSection>
          </Section>

          {/* Anthropic Evaluation */}
          <Section id="eval-anthropic" icon={Globe} title="Evaluating Anthropic Models">
            <p>Evaluate Claude Sonnet 4, Claude Opus 4, Claude 3.5 Haiku, and other Anthropic models.</p>

            <SubSection title="Setup">
              <CodeBlock code={`# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."`} />
            </SubSection>

            <SubSection title="Evaluate Claude Sonnet 4">
              <CodeBlock code={`beyondbench evaluate \\
  --model-id claude-sonnet-4-20250514 \\
  --api-provider anthropic \\
  --suite all \\
  --datapoints 100 \\
  --temperature 0.7 \\
  --max-tokens 1024 \\
  --output-dir ./results/claude-sonnet`} />
            </SubSection>

            <SubSection title="Evaluate Claude Opus 4">
              <CodeBlock code={`beyondbench evaluate \\
  --model-id claude-opus-4-20250514 \\
  --api-provider anthropic \\
  --suite hard \\
  --datapoints 50 \\
  --max-tokens 4096 \\
  --output-dir ./results/claude-opus`} />
              <Callout type="info">
                Claude Opus 4 has higher per-token cost. Consider using smaller <code className="text-bb-accent font-mono">--datapoints</code> for initial evaluation, then scale up.
              </Callout>
            </SubSection>

            <SubSection title="Python API">
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
            </SubSection>

            <SubSection title="Supported Anthropic Models">
              <div className="glass-card p-4 text-xs">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 font-mono text-gray-500">
                  <div>claude-sonnet-4-20250514</div>
                  <div>claude-opus-4-20250514</div>
                  <div>claude-3-5-sonnet-20241022</div>
                  <div>claude-3-5-haiku-20241022</div>
                  <div>claude-3-opus-20240229</div>
                  <div>claude-3-haiku-20240307</div>
                </div>
              </div>
            </SubSection>
          </Section>

          {/* vLLM Evaluation */}
          <Section id="eval-vllm" icon={Cpu} title="Evaluating with vLLM">
            <p>Use vLLM for high-throughput local evaluation of any HuggingFace model with PagedAttention and continuous batching.</p>

            <SubSection title="Prerequisites">
              <CodeBlock code={`# Install vLLM (requires CUDA)
pip install beyondbench[vllm]

# Or install vLLM separately
pip install vllm`} />
              <Callout type="warning">
                vLLM requires NVIDIA GPUs with CUDA support. Check your GPU memory before loading large models.
              </Callout>
            </SubSection>

            <SubSection title="Basic Usage">
              <CodeBlock code={`# Evaluate a 7B model on single GPU
beyondbench evaluate \\
  --model-id Qwen/Qwen2.5-7B-Instruct \\
  --backend vllm \\
  --suite all \\
  --datapoints 100`} />
            </SubSection>

            <SubSection title="Multi-GPU with Tensor Parallelism">
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
            </SubSection>

            <SubSection title="Batch Processing">
              <CodeBlock code={`# Increase batch size for throughput
beyondbench evaluate \\
  --model-id meta-llama/Llama-3.2-3B-Instruct \\
  --backend vllm \\
  --batch-size 16 \\
  --suite all --datapoints 200`} />
            </SubSection>

            <SubSection title="Specific GPU Selection">
              <CodeBlock code={`# Use specific CUDA device
beyondbench evaluate \\
  --model-id microsoft/phi-4 \\
  --backend vllm \\
  --cuda-device cuda:1 \\
  --suite all

# With CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=2,3 beyondbench evaluate \\
  --model-id Qwen/Qwen2.5-14B-Instruct \\
  --backend vllm \\
  --tensor-parallel-size 2`} />
            </SubSection>

            <SubSection title="Quantized Models">
              <CodeBlock code={`# GPTQ quantized model
beyondbench evaluate \\
  --model-id TheBloke/Llama-2-70B-Chat-GPTQ \\
  --backend vllm \\
  --trust-remote-code \\
  --suite all

# AWQ quantized model
beyondbench evaluate \\
  --model-id casperhansen/llama-3-70b-instruct-awq \\
  --backend vllm \\
  --suite all`} />
            </SubSection>

            <SubSection title="Python API">
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
            </SubSection>

            <SubSection title="GPU Memory Guide">
              <div className="glass-card p-4 text-xs">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-bb-dark-50/20">
                      <th className="text-left py-2 text-gray-500">Model Size</th>
                      <th className="text-left py-2 text-gray-500">FP16 VRAM</th>
                      <th className="text-left py-2 text-gray-500">Recommended GPUs</th>
                    </tr>
                  </thead>
                  <tbody className="text-gray-400">
                    <tr className="border-b border-bb-dark-50/10"><td className="py-1.5">1-3B</td><td className="py-1.5">~6 GB</td><td className="py-1.5">1x RTX 3090/4090</td></tr>
                    <tr className="border-b border-bb-dark-50/10"><td className="py-1.5">7-8B</td><td className="py-1.5">~16 GB</td><td className="py-1.5">1x A100-40G or 1x RTX 4090</td></tr>
                    <tr className="border-b border-bb-dark-50/10"><td className="py-1.5">13-14B</td><td className="py-1.5">~28 GB</td><td className="py-1.5">1x A100-80G</td></tr>
                    <tr className="border-b border-bb-dark-50/10"><td className="py-1.5">32-34B</td><td className="py-1.5">~68 GB</td><td className="py-1.5">2x A100-40G or 1x A100-80G</td></tr>
                    <tr className="border-b border-bb-dark-50/10"><td className="py-1.5">65-70B</td><td className="py-1.5">~140 GB</td><td className="py-1.5">2x A100-80G or 4x A100-40G</td></tr>
                    <tr><td className="py-1.5">405B</td><td className="py-1.5">~810 GB</td><td className="py-1.5">8x A100-80G or 8x H100</td></tr>
                  </tbody>
                </table>
              </div>
            </SubSection>
          </Section>

          {/* Transformers Evaluation */}
          <Section id="eval-transformers" icon={Cpu} title="Evaluating with HuggingFace Transformers">
            <p>Use the HuggingFace Transformers backend for flexible local inference, especially useful for debugging and smaller models.</p>

            <SubSection title="Basic Usage">
              <CodeBlock code={`beyondbench evaluate \\
  --model-id meta-llama/Llama-3.2-3B-Instruct \\
  --backend transformers \\
  --cuda-device cuda:0 \\
  --suite all \\
  --datapoints 100`} />
            </SubSection>

            <SubSection title="Microsoft Phi Models">
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
            </SubSection>

            <SubSection title="Mistral Models">
              <CodeBlock code={`beyondbench evaluate \\
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \\
  --backend transformers \\
  --suite all --datapoints 100`} />
            </SubSection>

            <SubSection title="Python API with Custom Generation Config">
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
            </SubSection>

            <Callout type="info">
              The Transformers backend is slower than vLLM but more compatible. Use it when vLLM doesn't support your model architecture, or for debugging purposes.
            </Callout>
          </Section>

          {/* CLI Reference */}
          <Section id="cli" icon={Terminal} title="CLI Reference">
            <SubSection title="beyondbench evaluate">
              <p>Run model evaluation against BeyondBench tasks.</p>
              <div className="glass-card p-4 font-mono text-xs space-y-2 text-gray-400">
                <div className="text-gray-300 font-semibold mb-2">Model & Backend</div>
                <div><span className="text-bb-accent">--model-id</span> TEXT &nbsp;&nbsp; Model identifier (required)</div>
                <div><span className="text-bb-accent">--backend</span> [vllm|transformers] &nbsp;&nbsp; Local inference backend</div>
                <div><span className="text-bb-accent">--api-provider</span> [openai|gemini|anthropic] &nbsp;&nbsp; API provider</div>
                <div><span className="text-bb-accent">--api-key</span> TEXT &nbsp;&nbsp; API key (or set env variable)</div>
                <div className="text-gray-300 font-semibold mb-2 mt-4">Task Selection</div>
                <div><span className="text-bb-accent">--suite</span> [easy|medium|hard|all] &nbsp;&nbsp; Task suite to evaluate (default: all)</div>
                <div><span className="text-bb-accent">--tasks</span> TEXT... &nbsp;&nbsp; Specific tasks to evaluate</div>
                <div><span className="text-bb-accent">--datapoints</span> INT &nbsp;&nbsp; Number of test cases per task (default: 100)</div>
                <div><span className="text-bb-accent">--folds</span> INT &nbsp;&nbsp; Number of cross-validation folds (default: 1)</div>
                <div className="text-gray-300 font-semibold mb-2 mt-4">Generation Parameters</div>
                <div><span className="text-bb-accent">--temperature</span> FLOAT &nbsp;&nbsp; Sampling temperature (default: 0.7)</div>
                <div><span className="text-bb-accent">--top-p</span> FLOAT &nbsp;&nbsp; Nucleus sampling (default: 0.9)</div>
                <div><span className="text-bb-accent">--max-tokens</span> INT &nbsp;&nbsp; Max generation tokens (default: 32768, falls back to 8192 on error)</div>
                <div><span className="text-bb-accent">--seed</span> INT &nbsp;&nbsp; Random seed for reproducibility</div>
                <div className="text-gray-300 font-semibold mb-2 mt-4">Hardware Configuration</div>
                <div><span className="text-bb-accent">--cuda-device</span> TEXT &nbsp;&nbsp; CUDA device for local models (default: cuda:0)</div>
                <div><span className="text-bb-accent">--tensor-parallel-size</span> INT &nbsp;&nbsp; Number of GPUs for tensor parallelism (default: 1)</div>
                <div><span className="text-bb-accent">--gpu-memory-utilization</span> FLOAT &nbsp;&nbsp; GPU memory ratio (default: 0.96)</div>
                <div><span className="text-bb-accent">--trust-remote-code</span> &nbsp;&nbsp; Allow remote code execution</div>
                <div><span className="text-bb-accent">--batch-size</span> INT &nbsp;&nbsp; Batch size for inference (default: 1)</div>
                <div className="text-gray-300 font-semibold mb-2 mt-4">Provider-Specific</div>
                <div><span className="text-bb-accent">--reasoning-effort</span> [minimal|low|medium|high] &nbsp;&nbsp; OpenAI reasoning effort</div>
                <div><span className="text-bb-accent">--thinking-budget</span> INT &nbsp;&nbsp; Gemini thinking budget (default: 1024, 0 to disable, -1 for dynamic)</div>
                <div className="text-gray-300 font-semibold mb-2 mt-4">Scalable Task Options</div>
                <div><span className="text-bb-accent">--list-sizes</span> TEXT &nbsp;&nbsp; Comma-separated list sizes (e.g., "8,16,32,64")</div>
                <div><span className="text-bb-accent">--range-min</span> INT &nbsp;&nbsp; Min value for number generation (default: -100)</div>
                <div><span className="text-bb-accent">--range-max</span> INT &nbsp;&nbsp; Max value for number generation (default: 100)</div>
                <div className="text-gray-300 font-semibold mb-2 mt-4">Output & Logging</div>
                <div><span className="text-bb-accent">--output-dir</span> PATH &nbsp;&nbsp; Output directory (default: ./beyondbench_results)</div>
                <div><span className="text-bb-accent">--store-details</span> &nbsp;&nbsp; Save per-example results</div>
                <div><span className="text-bb-accent">--log-level</span> [DEBUG|INFO|WARNING|ERROR] &nbsp;&nbsp; Logging level (default: INFO)</div>
                <div><span className="text-bb-accent">--max-retries</span> INT &nbsp;&nbsp; Max retries for failed ops (default: 3)</div>
                <div><span className="text-bb-accent">--timeout</span> INT &nbsp;&nbsp; Timeout in seconds (default: 300)</div>
              </div>
            </SubSection>

            <SubSection title="beyondbench list-tasks">
              <p>Display available tasks and variations.</p>
              <CodeBlock code={`beyondbench list-tasks              # Show all tasks
beyondbench list-tasks --suite easy  # Show only easy tasks
beyondbench list-tasks --suite hard  # Show only hard tasks`} />
            </SubSection>

            <SubSection title="beyondbench run-config">
              <p>Execute evaluation from a YAML or JSON configuration file.</p>
              <CodeBlock code="beyondbench run-config config.yaml" />
            </SubSection>

            <SubSection title="beyondbench --version">
              <CodeBlock code={`beyondbench --version
# Output: beyondbench, version ${pypiVersion || '<latest>'}`} />
            </SubSection>
          </Section>

          {/* Python API */}
          <Section id="python-api" icon={Code} title="Python API">
            <SubSection title="Core Classes">
              <div className="space-y-4">
                <div className="glass-card p-4">
                  <div className="text-sm font-semibold text-bb-accent font-mono mb-2">ModelHandler</div>
                  <p className="text-xs text-gray-500 mb-2">Unified interface for all model backends.</p>
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

                <div className="glass-card p-4">
                  <div className="text-sm font-semibold text-bb-accent font-mono mb-2">EvaluationEngine</div>
                  <p className="text-xs text-gray-500 mb-2">Orchestrates the evaluation process.</p>
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

                <div className="glass-card p-4">
                  <div className="text-sm font-semibold text-bb-accent font-mono mb-2">TaskRegistry</div>
                  <p className="text-xs text-gray-500 mb-2">Discover and access available tasks.</p>
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
              </div>
            </SubSection>

            <SubSection title="Parsing Utilities">
              <CodeBlock language="python" code={`from beyondbench.utils.parsing import parse_boxed_answer, extract_number, extract_list

# Parse LaTeX boxed answers
answer = parse_boxed_answer(r"The answer is \\boxed{42}")  # Returns 42

# Extract numbers from text
num = extract_number("The result is 3.14")  # Returns 3.14

# Extract lists from text
lst = extract_list("The sorted list is [1, 2, 3, 4, 5]")  # Returns [1,2,3,4,5]`} />
            </SubSection>

            <SubSection title="Logging Utilities">
              <CodeBlock language="python" code={`from beyondbench.utils.logging_utils import setup_logging, get_logger

# Setup logging with custom level
setup_logging(level="DEBUG")
logger = get_logger("my_evaluation")
logger.info("Starting evaluation...")`} />
            </SubSection>
          </Section>

          {/* Backend Overview */}
          <Section id="backends" icon={Cpu} title="Backend Overview">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-bb-dark-50/20">
                    <th className="text-left py-2 px-3 text-xs font-semibold text-gray-500">Backend</th>
                    <th className="text-left py-2 px-3 text-xs font-semibold text-gray-500">Models</th>
                    <th className="text-left py-2 px-3 text-xs font-semibold text-gray-500">Features</th>
                    <th className="text-left py-2 px-3 text-xs font-semibold text-gray-500">Setup</th>
                  </tr>
                </thead>
                <tbody className="text-xs text-gray-400">
                  <tr className="border-b border-bb-dark-50/10"><td className="py-2 px-3 font-semibold text-gray-300">OpenAI</td><td className="py-2 px-3">GPT-4o, GPT-5, o3, o4-Mini</td><td className="py-2 px-3">Reasoning effort control</td><td className="py-2 px-3 font-mono text-[10px]">OPENAI_API_KEY</td></tr>
                  <tr className="border-b border-bb-dark-50/10"><td className="py-2 px-3 font-semibold text-gray-300">Gemini</td><td className="py-2 px-3">Gemini 2.5 Pro/Flash</td><td className="py-2 px-3">Thinking budget config</td><td className="py-2 px-3 font-mono text-[10px]">GEMINI_API_KEY</td></tr>
                  <tr className="border-b border-bb-dark-50/10"><td className="py-2 px-3 font-semibold text-gray-300">Anthropic</td><td className="py-2 px-3">Claude Sonnet 4, Opus 4</td><td className="py-2 px-3">Latest Claude models</td><td className="py-2 px-3 font-mono text-[10px]">ANTHROPIC_API_KEY</td></tr>
                  <tr className="border-b border-bb-dark-50/10"><td className="py-2 px-3 font-semibold text-gray-300">vLLM</td><td className="py-2 px-3">Any HuggingFace model</td><td className="py-2 px-3">Batch, tensor parallel, high throughput</td><td className="py-2 px-3 font-mono text-[10px]">pip install beyondbench[vllm]</td></tr>
                  <tr className="border-b border-bb-dark-50/10"><td className="py-2 px-3 font-semibold text-gray-300">Transformers</td><td className="py-2 px-3">Any HuggingFace model</td><td className="py-2 px-3">CPU/GPU, auto device map</td><td className="py-2 px-3 font-mono text-[10px]">Included by default</td></tr>
                </tbody>
              </table>
            </div>

            <SubSection title="Environment Variables">
              <CodeBlock code={`# API Provider Keys
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."

# For gated HuggingFace models (e.g., Llama, Mistral)
export HF_TOKEN="hf_..."

# GPU selection for local models
export CUDA_VISIBLE_DEVICES="0,1"  # Use specific GPUs`} />
            </SubSection>
          </Section>

          {/* Task Suites */}
          <Section id="tasks" icon={Layers} title="Task Suites">
            <SubSection title="Easy Suite (29 Tasks)">
              <p>Fundamental arithmetic and statistical operations with scalable complexity.</p>
              <div className="glass-card p-4 mt-2">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono text-gray-500">
                  <div><span className="text-gray-400">Arithmetic:</span> sum, multiplication, subtraction, division, absolute_difference</div>
                  <div><span className="text-gray-400">Statistics:</span> mean, median, mode</div>
                  <div><span className="text-gray-400">Counting:</span> odd_count, even_count, count_negative, count_unique, count_greater_than_previous, count_palindromic, count_perfect_squares, count_multiples, local_maxima_count</div>
                  <div><span className="text-gray-400">Extrema:</span> find_maximum, find_minimum, second_maximum, range, index_of_maximum, max_adjacent_difference, sum_of_max_indices</div>
                  <div><span className="text-gray-400">Sequences:</span> sorting, longest_increasing_subsequence, alternating_sum, sum_of_digits</div>
                  <div><span className="text-gray-400">Comparison:</span> comparison</div>
                </div>
              </div>
              <CodeBlock code="beyondbench evaluate --model-id MODEL --suite easy --datapoints 100" />
            </SubSection>

            <SubSection title="Medium Suite (5 Tasks, 49 Variations)">
              <p>Sequence patterns and recursive reasoning requiring multi-step problem solving.</p>
              <div className="glass-card p-4 mt-2">
                <div className="space-y-1 text-xs font-mono text-gray-500">
                  <div><span className="text-gray-400">fibonacci_sequence</span> - 6 variations (Tribonacci, Lucas, modified recursive)</div>
                  <div><span className="text-gray-400">algebraic_sequence</span> - 10 variations (polynomial, arithmetic, quadratic)</div>
                  <div><span className="text-gray-400">geometric_sequence</span> - 10 variations (exponential, compound, factorial)</div>
                  <div><span className="text-gray-400">prime_sequence</span> - 11 variations (prime gaps, twin primes, Sophie Germain)</div>
                  <div><span className="text-gray-400">complex_pattern</span> - 12 variations (interleaved, conditional, multi-rule)</div>
                </div>
              </div>
              <CodeBlock code="beyondbench evaluate --model-id MODEL --suite medium --datapoints 100" />
            </SubSection>

            <SubSection title="Hard Suite (10 Tasks, 68 Variations)">
              <p>NP-complete and constraint satisfaction problems testing advanced reasoning.</p>
              <div className="glass-card p-4 mt-2">
                <div className="space-y-1 text-xs font-mono text-gray-500">
                  <div><span className="text-gray-400">tower_hanoi</span> - 6 variations | O(2^n) computational complexity</div>
                  <div><span className="text-gray-400">n_queens</span> - 4 variations | NP-complete board placement</div>
                  <div><span className="text-gray-400">graph_coloring</span> - 10 variations | Chromatic number computation</div>
                  <div><span className="text-gray-400">boolean_sat</span> - 5 variations | 2-SAT, 3-SAT, Horn clauses</div>
                  <div><span className="text-gray-400">sudoku</span> - 8 variations | Standard, diagonal, irregular</div>
                  <div><span className="text-gray-400">cryptarithmetic</span> - 12 variations | Letter-to-digit mapping</div>
                  <div><span className="text-gray-400">matrix_chain</span> - 5 variations | Dynamic programming optimization</div>
                  <div><span className="text-gray-400">modular_systems</span> - 5 variations | Chinese Remainder Theorem</div>
                  <div><span className="text-gray-400">constraint_optimization</span> - 5 variations | Knapsack, scheduling</div>
                  <div><span className="text-gray-400">logic_grid_puzzles</span> - 8 variations | Einstein puzzles, deductive reasoning</div>
                </div>
              </div>
              <CodeBlock code="beyondbench evaluate --model-id MODEL --suite hard --datapoints 50" />
            </SubSection>

            <SubSection title="Running Specific Tasks">
              <CodeBlock code={`# Run only specific tasks
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --tasks sum_task sorting_task sudoku_task

# Combine with suite filter
beyondbench evaluate --model-id MODEL --suite easy \\
  --tasks mean_task median_task mode_task`} />
            </SubSection>
          </Section>

          {/* Advanced Evaluation */}
          <Section id="advanced-eval" icon={BarChart3} title="Advanced Evaluation">
            <SubSection title="Scalable Complexity Testing">
              <p>Test how model performance scales with problem size by varying list sizes:</p>
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
            </SubSection>

            <SubSection title="Cross-Validation Folds">
              <p>Use multiple folds for statistically robust results:</p>
              <CodeBlock code={`beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --suite all --datapoints 100 --folds 5 \\
  --seed 42 --store-details`} />
            </SubSection>

            <SubSection title="Reproducible Evaluation">
              <CodeBlock code={`# Set seed for reproducibility
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --seed 42 --temperature 0.0 --suite all

# Temperature 0.0 ensures deterministic outputs from the model`} />
            </SubSection>

            <SubSection title="Comparing Multiple Models">
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
            </SubSection>

            <SubSection title="Cost-Efficient Evaluation Strategy">
              <p>Minimize API costs while getting meaningful results:</p>
              <div className="glass-card p-4 text-xs text-gray-400 space-y-2">
                <div><strong className="text-gray-300">Step 1:</strong> Run easy suite with 50 datapoints to verify setup</div>
                <div><strong className="text-gray-300">Step 2:</strong> Run all suites with 20 datapoints for quick overview</div>
                <div><strong className="text-gray-300">Step 3:</strong> Scale up to 100+ datapoints for publishable results</div>
                <div><strong className="text-gray-300">Step 4:</strong> Use --store-details to analyze failures without re-running</div>
              </div>
              <CodeBlock code={`# Step 1: Quick sanity check
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --suite easy --datapoints 20

# Step 2: Full overview
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --suite all --datapoints 50

# Step 3: Publication-quality
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --suite all --datapoints 100 --folds 3 --store-details`} />
            </SubSection>
          </Section>

          {/* Configuration */}
          <Section id="configuration" icon={Settings} title="Configuration">
            <SubSection title="YAML Configuration File">
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
              <p>Run with:</p>
              <CodeBlock code="beyondbench run-config config.yaml" />
            </SubSection>

            <SubSection title="vLLM-Specific Configuration">
              <CodeBlock language="yaml" code={`model:
  id: Qwen/Qwen2.5-72B-Instruct
  backend: vllm
  tp_size: 4              # tensor parallel GPUs
  gpu_memory_utilization: 0.9
  max_model_len: 32768

evaluation:
  suite: all
  datapoints: 100
  temperature: 0.1`} />
            </SubSection>

            <SubSection title="Multi-Model Batch Configuration">
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
            </SubSection>
          </Section>

          {/* Extending */}
          <Section id="extending" icon={GitBranch} title="Extending BeyondBench">
            <SubSection title="Adding Custom Tasks">
              <p>Create new tasks by extending the <code className="text-bb-accent font-mono text-xs bg-bb-dark-400/60 px-1.5 py-0.5 rounded">BaseTask</code> class:</p>
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
            # Generate a problem
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
            </SubSection>

            <SubSection title="Task Registration">
              <p>Place your task file in the appropriate suite directory:</p>
              <CodeBlock code={`beyondbench/tasks/easy/my_custom_task.py    # For easy tasks
beyondbench/tasks/medium/my_task.py         # For medium tasks
beyondbench/tasks/hard/my_task.py           # For hard tasks`} />
              <p>Tasks are auto-discovered by the <code className="text-bb-accent font-mono text-xs bg-bb-dark-400/60 px-1.5 py-0.5 rounded">TaskRegistry</code> when placed in the correct directory.</p>
            </SubSection>

            <SubSection title="Adding Custom Backends">
              <p>Extend the ModelHandler to support additional inference providers:</p>
              <CodeBlock language="python" code={`from beyondbench.models.model_handler import ModelHandler

class MyCustomBackend:
    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        # Initialize your custom backend

    def generate(self, prompt, **kwargs):
        """Generate a response for the given prompt."""
        # Your inference logic here
        return "model response"

# Register and use
handler = ModelHandler(model_id="my-model", backend="custom")
handler._backend = MyCustomBackend("my-model")`} />
            </SubSection>
          </Section>

          {/* Output Format */}
          <Section id="output" icon={Database} title="Output Format">
            <SubSection title="Results Structure">
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
    "sum": {
      "accuracy": 0.95,
      "instruction_following": 1.0,
      "avg_tokens": 120.5,
      "suite": "easy"
    },
    "sudoku": {
      "accuracy": 0.12,
      "instruction_following": 0.98,
      "avg_tokens": 890.3,
      "suite": "hard"
    }
  },
  "metadata": {
    "version": "${pypiVersion || '<latest>'}",
    "timestamp": "2025-02-05T10:30:00Z",
    "seed": 42,
    "temperature": 0.7,
    "max_tokens": 512
  }
}`} />
            </SubSection>

            <SubSection title="Output Directory Structure">
              <CodeBlock code={`beyondbench_results/
  gpt-4o/
    summary.json          # Aggregate results
    easy/
      sum_results.json    # Per-task results
      sum_details.json    # Per-example details (if --store-details)
      sorting_results.json
      ...
    medium/
      fibonacci_results.json
      algebraic_results.json
      ...
    hard/
      sudoku_results.json
      n_queens_results.json
      ...`} />
            </SubSection>

            <SubSection title="Detailed Results (--store-details)">
              <CodeBlock language="json" code={`{
  "task": "sum",
  "suite": "easy",
  "examples": [
    {
      "input": [12, 45, 78, 3, 56],
      "ground_truth": 194,
      "model_response": "The sum is 194.",
      "extracted_answer": 194,
      "correct": true,
      "instruction_following": true,
      "tokens_used": 15
    }
  ]
}`} />
            </SubSection>
          </Section>

          {/* Troubleshooting */}
          <Section id="troubleshooting" icon={Wrench} title="Troubleshooting">
            <SubSection title="Common Issues">
              <div className="space-y-4">
                <div className="glass-card p-4">
                  <div className="text-xs font-semibold text-red-400 mb-1">CUDA out of memory</div>
                  <p className="text-xs text-gray-500 mb-2">Model too large for available GPU memory.</p>
                  <CodeBlock code={`# Solution 1: Use tensor parallelism
beyondbench evaluate --model-id MODEL --backend vllm --tensor-parallel-size 2

# Solution 2: Reduce GPU memory utilization
beyondbench evaluate --model-id MODEL --backend vllm --gpu-memory-utilization 0.85

# Solution 3: Use a quantized model
beyondbench evaluate --model-id MODEL-GPTQ --backend vllm`} />
                </div>

                <div className="glass-card p-4">
                  <div className="text-xs font-semibold text-red-400 mb-1">API Rate Limit Errors</div>
                  <p className="text-xs text-gray-500 mb-2">Too many API requests in a short time.</p>
                  <CodeBlock code={`# Increase timeout and retries
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --max-retries 5 --timeout 600

# Reduce datapoints for initial testing
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --datapoints 20 --suite easy`} />
                </div>

                <div className="glass-card p-4">
                  <div className="text-xs font-semibold text-red-400 mb-1">Import Errors</div>
                  <p className="text-xs text-gray-500 mb-2">Missing dependencies or incorrect installation.</p>
                  <CodeBlock code={`# Reinstall with all dependencies
pip install --force-reinstall beyondbench[full]

# Verify installation
python -c "from beyondbench import EvaluationEngine, TaskRegistry; print('OK')"

# Check version
beyondbench --version`} />
                </div>

                <div className="glass-card p-4">
                  <div className="text-xs font-semibold text-red-400 mb-1">vLLM Model Loading Failures</div>
                  <p className="text-xs text-gray-500 mb-2">Model architecture not supported or trust_remote_code needed.</p>
                  <CodeBlock code={`# Enable remote code execution
beyondbench evaluate --model-id MODEL --backend vllm --trust-remote-code

# Try transformers backend as fallback
beyondbench evaluate --model-id MODEL --backend transformers`} />
                </div>
              </div>
            </SubSection>

            <SubSection title="Debug Mode">
              <CodeBlock code={`# Enable debug logging for detailed output
beyondbench evaluate --model-id gpt-4o --api-provider openai \\
  --log-level DEBUG --suite easy --datapoints 5`} />
            </SubSection>

            <SubSection title="Getting Help">
              <div className="glass-card p-4 text-xs text-gray-400 space-y-2">
                <div>Report issues: <a href="https://github.com/ctrl-gaurav/BeyondBench/issues" className="text-bb-accent hover:underline">GitHub Issues</a></div>
                <div>Email: <a href="mailto:gks@vt.edu" className="text-bb-accent hover:underline">gks@vt.edu</a></div>
                <div>Paper: <a href="https://arxiv.org/abs/2509.24210" className="text-bb-accent hover:underline">arXiv:2509.24210</a></div>
              </div>
            </SubSection>
          </Section>

          {/* Citation */}
          <Section id="citation" icon={FileText} title="Citation">
            <p>If you use BeyondBench in your research, please cite our paper (accepted at <strong className="text-bb-accent">ICLR 2026</strong>):</p>
            <CodeBlock language="bibtex" code={`@misc{srivastava2025beyondbenchbenchmarkfreeevaluationreasoning,
      title={BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models},
      author={Gaurav Srivastava and Aafiya Hussain and Zhenyu Bi and Swastik Roy and Priya Pitre and Meng Lu and Morteza Ziyadi and Xuan Wang},
      year={2025},
      eprint={2509.24210},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.24210},
}`} />
          </Section>
        </div>
    </div>
  )
}

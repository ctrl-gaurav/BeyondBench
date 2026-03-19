import { Routes, Route, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Leaderboard from './pages/Leaderboard'
import Documentation from './pages/Documentation'
import { useTheme } from './context/ThemeContext'

function ScrollProgress() {
  const [progress, setProgress] = useState(0)
  const { isDark } = useTheme()

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY
      const docHeight = document.documentElement.scrollHeight - window.innerHeight
      setProgress(docHeight > 0 ? (scrollTop / docHeight) * 100 : 0)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <div
      className="scroll-progress"
      style={{ width: `${progress}%` }}
    />
  )
}

function PageTransition({ children }) {
  const location = useLocation()
  const [phase, setPhase] = useState('done') // 'hidden' | 'animating' | 'done'

  useEffect(() => {
    setPhase('hidden')
    const t = requestAnimationFrame(() => {
      requestAnimationFrame(() => setPhase('animating'))
    })
    return () => cancelAnimationFrame(t)
  }, [location.pathname])

  useEffect(() => {
    if (phase === 'animating') {
      const timer = setTimeout(() => setPhase('done'), 300)
      return () => clearTimeout(timer)
    }
  }, [phase])

  // When 'done', remove transform entirely so fixed children (e.g. docs sidebar) work correctly
  const className = phase === 'hidden'
    ? 'opacity-0 translate-y-2 transition-all duration-300 ease-out'
    : phase === 'animating'
    ? 'opacity-100 translate-y-0 transition-all duration-300 ease-out'
    : 'opacity-100'

  return (
    <div className={className}>
      {children}
    </div>
  )
}

function App() {
  const { isDark } = useTheme()

  return (
    <div className={`min-h-screen ${isDark ? 'bg-bb-dark-500' : 'bg-bb-light-100'} grid-bg relative overflow-hidden noise-overlay`}>
      {/* Background orbs - enhanced with morphing animation */}
      <div className={`orb w-96 h-96 -top-48 -left-48 ${isDark ? 'bg-bb-accent' : 'bg-bb-accent-dark'}`} style={{ animationDelay: '0s' }} />
      <div className={`orb w-72 h-72 top-1/3 -right-36 ${isDark ? 'bg-bb-teal' : 'bg-bb-teal'}`} style={{ animationDelay: '5s' }} />
      <div className={`orb w-64 h-64 bottom-0 left-1/4 ${isDark ? 'bg-bb-purple' : 'bg-bb-purple'}`} style={{ animationDelay: '10s' }} />
      <div className={`orb w-48 h-48 top-2/3 right-1/4 ${isDark ? 'bg-bb-cyan/50' : 'bg-bb-cyan/30'}`} style={{ animationDelay: '7s' }} />

      <div className="relative z-10">
        <Navbar />
        <ScrollProgress />
        <PageTransition>
          <Routes>
            <Route path="/" element={<Leaderboard />} />
            <Route path="/docs" element={<Documentation />} />
          </Routes>
        </PageTransition>
        <Footer />
      </div>
    </div>
  )
}

export default App

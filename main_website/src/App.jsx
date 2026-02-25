import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Leaderboard from './pages/Leaderboard'
import Documentation from './pages/Documentation'
import { useTheme } from './context/ThemeContext'

function App() {
  const { isDark } = useTheme()

  return (
    <div className={`min-h-screen ${isDark ? 'bg-bb-dark-500' : 'bg-bb-light-100'} grid-bg relative overflow-hidden`}>
      {/* Background orbs */}
      <div className={`orb w-96 h-96 -top-48 -left-48 ${isDark ? 'bg-bb-accent' : 'bg-bb-accent-dark'}`} />
      <div className={`orb w-72 h-72 top-1/3 -right-36 ${isDark ? 'bg-bb-teal' : 'bg-bb-teal'}`} />
      <div className={`orb w-64 h-64 bottom-0 left-1/4 ${isDark ? 'bg-bb-purple' : 'bg-bb-purple'}`} />

      <div className="relative z-10">
        <Navbar />
        <Routes>
          <Route path="/" element={<Leaderboard />} />
          <Route path="/docs" element={<Documentation />} />
        </Routes>
        <Footer />
      </div>
    </div>
  )
}

export default App

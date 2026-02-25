import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Leaderboard from './pages/Leaderboard'
import Documentation from './pages/Documentation'

function App() {
  return (
    <div className="min-h-screen bg-bb-dark-500 grid-bg">
      <Navbar />
      <Routes>
        <Route path="/" element={<Leaderboard />} />
        <Route path="/docs" element={<Documentation />} />
      </Routes>
      <Footer />
    </div>
  )
}

export default App

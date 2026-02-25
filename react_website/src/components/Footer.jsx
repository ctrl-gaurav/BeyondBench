import { Hexagon } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-bb-dark-50/20 bg-bb-dark-600/50 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Hexagon className="w-6 h-6 text-bb-accent" />
              <span className="text-lg font-bold text-white">BeyondBench</span>
            </div>
            <p className="text-gray-500 text-sm leading-relaxed">
              Contamination-Resistant Evaluation of Reasoning in Language Models.
              Accepted at ICLR 2026.
            </p>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Links</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="https://arxiv.org/abs/2509.24210" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-bb-accent transition-colors">Paper (arXiv)</a></li>
              <li><a href="https://github.com/ctrl-gaurav/BeyondBench" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-bb-accent transition-colors">GitHub Repository</a></li>
              <li><a href="https://pypi.org/project/beyondbench/" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-bb-accent transition-colors">PyPI Package</a></li>
              <li><a href="https://ctrl-gaurav.github.io/BeyondBench/" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-bb-accent transition-colors">Leaderboard (GitHub Pages)</a></li>
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-4">Contact</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="mailto:gks@vt.edu" className="text-gray-500 hover:text-bb-accent transition-colors">gks@vt.edu</a></li>
              <li><a href="mailto:xuanw@vt.edu" className="text-gray-500 hover:text-bb-accent transition-colors">xuanw@vt.edu</a></li>
              <li className="text-gray-600">Virginia Tech, CS Department</li>
              <li className="text-gray-600">Amazon AGI</li>
            </ul>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t border-bb-dark-50/10 text-center text-xs text-gray-600">
          Built by the BeyondBench Team &bull; MIT License
        </div>
      </div>
    </footer>
  )
}

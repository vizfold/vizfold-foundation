import { Link } from 'react-router-dom'

export default function HomePage() {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-black mb-4">
          OpenFold Attention Visualization
        </h1>
        <p className="text-xl text-gray-700 max-w-3xl mx-auto">
          Interactive visualization platform for exploring attention mechanisms in
          OpenFold protein structure predictions
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
        <div className="card">
          <div className="text-3xl mb-4">🧬</div>
          <h3 className="text-xl font-semibold mb-2 text-black">
            Upload Proteins
          </h3>
          <p className="text-gray-700 mb-4">
            Upload protein sequences in FASTA format or PDB structures to analyze
          </p>
          <Link to="/proteins" className="btn-primary inline-block">
            Manage Proteins
          </Link>
        </div>

        <div className="card">
          <div className="text-3xl mb-4">⚡</div>
          <h3 className="text-xl font-semibold mb-2 text-black">
            Run Inference
          </h3>
          <p className="text-gray-700 mb-4">
            Execute OpenFold predictions and extract attention weights automatically
          </p>
          <Link to="/inference" className="btn-primary inline-block">
            Start Inference
          </Link>
        </div>

        <div className="card">
          <div className="text-3xl mb-4">📊</div>
          <h3 className="text-xl font-semibold mb-2 text-black">
            Visualize Attention
          </h3>
          <p className="text-gray-700 mb-4">
            Explore attention patterns with heatmaps, arc diagrams, and 3D views
          </p>
          <Link to="/proteins" className="btn-primary inline-block">
            View Visualizations
          </Link>
        </div>
      </div>

      <div className="card mt-12">
        <h2 className="text-2xl font-bold mb-4 text-black">
          Features
        </h2>
        <ul className="grid grid-cols-1 md:grid-cols-2 gap-4 text-gray-700">
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>Interactive 3D protein structure visualization</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>Multiple attention visualization modes</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>MSA row and triangle attention support</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>Layer and head-specific exploration</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>Export visualizations and attention data</span>
          </li>
          <li className="flex items-start">
            <span className="mr-2">✓</span>
            <span>Real-time inference with progress tracking</span>
          </li>
        </ul>
      </div>
    </div>
  )
}

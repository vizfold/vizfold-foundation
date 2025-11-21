import { Link, useLocation } from 'react-router-dom'
import type { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/proteins', label: 'Proteins' },
    { path: '/inference', label: 'Inference' },
  ]

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/'
    }

    return location.pathname.startsWith(path)
  }

  return (
    <div className="min-h-screen bg-white text-black flex flex-col">
      <header className="border-b border-gray-200">
        <div className="container-wide">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="no-underline">
              <h1 className="text-lg font-semibold tracking-tight text-black">
                OpenFold Attention
              </h1>
            </Link>

            <nav className="flex items-center gap-1">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-4 py-2 text-sm font-medium no-underline transition-colors ${
                    isActive(item.path)
                      ? 'bg-black text-white'
                      : 'text-black hover:bg-gray-100'
                  }`}
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </div>
      </header>

      <main className="flex-1">
        <div className="container-wide py-8">{children}</div>
      </main>

      <footer className="border-t border-gray-200">
        <div className="container-wide">
          <div className="py-6 text-sm text-gray-600 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <span>OpenFold Attention Visualization Tool</span>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/vizfold/attention-viz-demo"
                target="_blank"
                rel="noreferrer"
              >
                Repository
              </a>
              <a
                href="https://github.com/vizfold/attention-viz-demo/issues"
                target="_blank"
                rel="noreferrer"
              >
                Issues
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

import { Link, Outlet, useLocation } from 'react-router-dom'

export default function Layout() {
  const location = useLocation()

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/')
  }

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Header - minimal, clean */}
      <header className="border-b border-gray-300">
        <div className="container-wide">
          <div className="flex items-center justify-between h-16">
            {/* Logo/Title */}
            <Link to="/" className="no-underline">
              <h1 className="text-lg font-semibold text-black tracking-tight">
                OpenFold Attention Visualization
              </h1>
            </Link>

            {/* Navigation */}
            <nav className="flex items-center space-x-1">
              <NavLink to="/" active={isActive('/')  && location.pathname === '/'}>
                Home
              </NavLink>
              <NavLink to="/proteins" active={isActive('/proteins') || isActive('/visualization')}>
                Proteins
              </NavLink>
              <NavLink to="/inference" active={isActive('/inference')}>
                Inference
              </NavLink>
            </nav>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1">
        <div className="container-wide py-8">
          <Outlet />
        </div>
      </main>

      {/* Footer - simple */}
      <footer className="border-t border-gray-300 mt-auto">
        <div className="container-wide">
          <div className="py-6 flex items-center justify-between text-sm text-gray-600">
            <div>
              <p>
                OpenFold Attention Visualization Tool
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <a
                href="https://github.com/vizfold/attention-viz-demo"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-black transition-colors"
              >
                Documentation
              </a>
              <a
                href="https://github.com/vizfold/attention-viz-demo/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-black transition-colors"
              >
                Report Issue
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

interface NavLinkProps {
  to: string
  active: boolean
  children: React.ReactNode
}

function NavLink({ to, active, children }: NavLinkProps) {
  return (
    <Link
      to={to}
      className={`
        px-4 py-2 text-sm font-medium no-underline transition-colors
        ${active
          ? 'bg-black text-white'
          : 'text-black hover:bg-gray-100'
        }
      `}
    >
      {children}
    </Link>
  )
}

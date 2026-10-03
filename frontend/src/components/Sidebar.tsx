import { NavLink, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, Mic2, Users, Search, Bell, Shield,
  ChevronLeft, ChevronRight, Brain, Upload, Settings2
} from 'lucide-react'
import { useAppStore } from '../store'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/meetings', icon: Mic2, label: 'Meetings' },
  { to: '/upload', icon: Upload, label: 'Upload Meeting' },
  { to: '/employees', icon: Users, label: 'Employees' },
  { to: '/search', icon: Search, label: 'Knowledge Base' },
  { to: '/notifications', icon: Bell, label: 'Notifications' },
  { to: '/admin', icon: Shield, label: 'Admin' },
]

export default function Sidebar() {
  const { sidebarOpen, toggleSidebar } = useAppStore()

  return (
    <aside
      className={`flex flex-col bg-slate-900 border-r border-slate-800 transition-all duration-300 ${
        sidebarOpen ? 'w-60' : 'w-16'
      } min-h-screen shrink-0`}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 h-16 border-b border-slate-800">
        <div className="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center shrink-0">
          <Brain className="w-5 h-5 text-white" />
        </div>
        {sidebarOpen && (
          <span className="font-bold text-slate-100 text-base tracking-tight truncate">
            MeetWise AI
          </span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 px-2 space-y-0.5 overflow-y-auto">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors group ${
                isActive
                  ? 'bg-brand-600/20 text-brand-400'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon className={`w-5 h-5 shrink-0 ${isActive ? 'text-brand-400' : ''}`} />
                {sidebarOpen && <span className="truncate">{label}</span>}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <div className="p-2 border-t border-slate-800">
        <button
          onClick={toggleSidebar}
          className="btn-ghost w-full flex items-center justify-center gap-2 text-xs"
        >
          {sidebarOpen ? (
            <><ChevronLeft className="w-4 h-4" /> <span>Collapse</span></>
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </button>
      </div>
    </aside>
  )
}

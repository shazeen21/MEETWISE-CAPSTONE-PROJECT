import { ReactNode } from 'react'
import { Loader2 } from 'lucide-react'

// ─── Spinner ──────────────────────────────────────────────────────────────────
export function Spinner({ className = 'w-5 h-5' }: { className?: string }) {
  return <Loader2 className={`animate-spin text-brand-400 ${className}`} />
}

// ─── Loading Page ─────────────────────────────────────────────────────────────
export function LoadingPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
      <Spinner className="w-8 h-8" />
      <p className="muted">Loading…</p>
    </div>
  )
}

// ─── Error Box ────────────────────────────────────────────────────────────────
export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-xl bg-red-900/20 border border-red-800 p-4 text-red-300 text-sm">
      {message}
    </div>
  )
}

// ─── Empty State ──────────────────────────────────────────────────────────────
export function EmptyState({ icon, title, description }: {
  icon: ReactNode
  title: string
  description?: string
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center text-slate-500">
        {icon}
      </div>
      <h3 className="font-semibold text-slate-300">{title}</h3>
      {description && <p className="muted max-w-xs">{description}</p>}
    </div>
  )
}

// ─── Stat Card ────────────────────────────────────────────────────────────────
export function StatCard({
  label, value, icon, sub, color = 'text-brand-400',
}: {
  label: string
  value: string | number
  icon: ReactNode
  sub?: string
  color?: string
}) {
  return (
    <div className="card p-5 flex items-start gap-4">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center bg-slate-800 ${color}`}>
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-2xl font-bold text-slate-100">{value}</p>
        <p className="muted truncate">{label}</p>
        {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

// ─── Section Header ───────────────────────────────────────────────────────────
export function SectionHeader({ title, subtitle, action }: {
  title: string
  subtitle?: string
  action?: ReactNode
}) {
  return (
    <div className="flex items-start justify-between gap-4 mb-6">
      <div>
        <h2 className="text-xl font-bold text-slate-100">{title}</h2>
        {subtitle && <p className="muted mt-0.5">{subtitle}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  )
}

// ─── Page Layout ──────────────────────────────────────────────────────────────
export function Page({ children, title, subtitle, action }: {
  children: ReactNode
  title?: string
  subtitle?: string
  action?: ReactNode
}) {
  return (
    <div className="p-6 max-w-screen-xl mx-auto">
      {title && <SectionHeader title={title} subtitle={subtitle} action={action} />}
      {children}
    </div>
  )
}

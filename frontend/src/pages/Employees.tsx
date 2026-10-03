import { useState } from 'react'
import { useQuery } from 'react-query'
import { useNavigate } from 'react-router-dom'
import { Users, UserPlus, Mic, CheckCircle, XCircle, Search, Eye, ChevronRight } from 'lucide-react'
import { employeesApi, Employee } from '../api'
import { formatDate, formatTimeAgo } from '../utils'
import { LoadingPage, EmptyState, Page } from '../components/UI'

// ─── Avatar colour palette ────────────────────────────────────────────────────

const AVATAR_COLORS = [
  'bg-violet-700/30 text-violet-300',
  'bg-sky-700/30 text-sky-300',
  'bg-emerald-700/30 text-emerald-300',
  'bg-rose-700/30 text-rose-300',
  'bg-amber-700/30 text-amber-300',
  'bg-teal-700/30 text-teal-300',
]

function avatarColor(name: string): string {
  let hash = 0
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash)
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length]
}

function initials(name: string): string {
  return name.slice(0, 2).toUpperCase()
}

// ─── Employee Card ────────────────────────────────────────────────────────────

function EmployeeCard({ employee }: { employee: Employee }) {
  const navigate = useNavigate()
  const colorClass = avatarColor(employee.name)

  return (
    <div className="card p-5 flex flex-col gap-4 hover:border-slate-600 transition-colors">
      {/* Top row – avatar + identity */}
      <div className="flex items-start gap-4">
        <div
          className={`w-12 h-12 rounded-full flex items-center justify-center text-base font-bold shrink-0 ${colorClass}`}
        >
          {initials(employee.name)}
        </div>

        <div className="min-w-0 flex-1">
          <p className="font-semibold text-slate-100 truncate">{employee.name}</p>
          <p className="text-sm text-slate-400 truncate">{employee.designation}</p>
          <p className="text-xs text-slate-500 truncate">
            {employee.department}
            {employee.team ? ` · ${employee.team}` : ''}
          </p>
        </div>

        {/* Enrollment badge */}
        {employee.has_enrolled_voice ? (
          <span className="badge badge-green flex items-center gap-1 shrink-0">
            <CheckCircle className="w-3 h-3" />
            Enrolled
          </span>
        ) : (
          <span className="badge badge-yellow flex items-center gap-1 shrink-0">
            <XCircle className="w-3 h-3" />
            Pending
          </span>
        )}
      </div>

      {/* Voice sample count + registration date */}
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <Mic className="w-3.5 h-3.5" />
          {employee.voice_samples_count}{' '}
          {employee.voice_samples_count === 1 ? 'voice sample' : 'voice samples'}
        </span>
        <span title={formatTimeAgo(employee.created_at)}>
          Registered {formatDate(employee.created_at)}
        </span>
      </div>

      {/* View Profile button */}
      <button
        onClick={() => navigate(`/employees/${employee.id}`)}
        className="btn-secondary w-full flex items-center justify-center gap-2 text-sm py-2"
      >
        <Eye className="w-4 h-4" />
        View Profile
        <ChevronRight className="w-4 h-4 ml-auto" />
      </button>
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Employees() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [deptFilter, setDeptFilter] = useState('')

  const { data: employees, isLoading, isError } = useQuery<Employee[]>(
    ['employees'],
    () => employeesApi.list(),
    { staleTime: 30_000 },
  )

  // ── Derived stats ──────────────────────────────────────────────────────────
  const total = employees?.length ?? 0
  const enrolled = employees?.filter(e => e.has_enrolled_voice).length ?? 0
  const pending = total - enrolled

  // ── Department options for filter dropdown ─────────────────────────────────
  const departments = Array.from(
    new Set(employees?.map(e => e.department).filter(Boolean) ?? []),
  ).sort()

  // ── Filtered list ──────────────────────────────────────────────────────────
  const filtered = (employees ?? []).filter(emp => {
    const q = search.toLowerCase()
    const matchesSearch =
      !q ||
      emp.name.toLowerCase().includes(q) ||
      emp.department.toLowerCase().includes(q) ||
      emp.designation.toLowerCase().includes(q)
    const matchesDept = !deptFilter || emp.department === deptFilter
    return matchesSearch && matchesDept
  })

  // ── Render ─────────────────────────────────────────────────────────────────
  if (isLoading) return <LoadingPage />

  if (isError) {
    return (
      <Page title="Employees" subtitle="Voice-enrolled employee directory">
        <div className="rounded-xl bg-red-900/20 border border-red-800 p-4 text-red-300 text-sm">
          Failed to load employees. Please try again.
        </div>
      </Page>
    )
  }

  return (
    <Page
      title="Employees"
      subtitle="Voice-enrolled employee directory"
      action={
        <button
          onClick={() => navigate('/employees/register')}
          className="btn-primary flex items-center gap-2"
        >
          <UserPlus className="w-4 h-4" />
          Register Employee
        </button>
      }
    >
      {/* ── Stats bar ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="card p-4 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-slate-800 flex items-center justify-center text-brand-400">
            <Users className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xl font-bold text-slate-100">{total}</p>
            <p className="muted text-xs">Total Employees</p>
          </div>
        </div>

        <div className="card p-4 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-900/30 flex items-center justify-center text-emerald-400">
            <CheckCircle className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xl font-bold text-slate-100">{enrolled}</p>
            <p className="muted text-xs">Voice Enrolled</p>
          </div>
        </div>

        <div className="card p-4 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-amber-900/30 flex items-center justify-center text-amber-400">
            <Mic className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xl font-bold text-slate-100">{pending}</p>
            <p className="muted text-xs">Pending Enrollment</p>
          </div>
        </div>
      </div>

      {/* ── Search + filter bar ────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
          <input
            className="input pl-9 w-full"
            placeholder="Search by name, department, or designation…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>

        <select
          className="input w-52 bg-slate-800"
          value={deptFilter}
          onChange={e => setDeptFilter(e.target.value)}
        >
          <option value="">All Departments</option>
          {departments.map(d => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </div>

      {/* ── Employee grid ─────────────────────────────────────────────────── */}
      {filtered.length === 0 ? (
        <EmptyState
          icon={<Users className="w-7 h-7" />}
          title={
            search || deptFilter
              ? 'No employees match your filters'
              : 'No employees registered yet'
          }
          description={
            search || deptFilter
              ? 'Try adjusting your search or department filter.'
              : 'Click "Register Employee" to add your first team member.'
          }
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map(emp => (
            <EmployeeCard key={emp.id} employee={emp} />
          ))}
        </div>
      )}
    </Page>
  )
}

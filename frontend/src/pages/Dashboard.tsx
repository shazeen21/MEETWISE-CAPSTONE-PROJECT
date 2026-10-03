import { useQuery } from 'react-query'
import { meetingsApi, employeesApi, adminApi } from '../api'
import { formatDuration, formatTimeAgo, emotionColor } from '../utils'
import { LoadingPage, StatCard, Page, ErrorBox } from '../components/UI'
import {
  Mic2,
  Users,
  CheckSquare,
  TrendingUp,
  Clock,
  Calendar,
  Brain,
  Activity,
  FileText,
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${ok ? 'bg-emerald-400' : 'bg-red-400'}`}
    />
  )
}

function HealthRow({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-800 last:border-0">
      <span className="muted text-sm">{label}</span>
      <div className="flex items-center gap-2">
        {ok !== undefined && <StatusDot ok={ok} />}
        <span className="text-slate-200 text-sm font-medium">{value}</span>
      </div>
    </div>
  )
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const navigate = useNavigate()

  const {
    data: meetings,
    isLoading: meetingsLoading,
    error: meetingsError,
  } = useQuery(['meetings'], () => meetingsApi.list(100, 0))

  const {
    data: employees,
    isLoading: employeesLoading,
    error: employeesError,
  } = useQuery(['employees'], () => employeesApi.list())

  const {
    data: health,
    isLoading: healthLoading,
    error: healthError,
  } = useQuery(['health'], () => adminApi.healthCheck(), {
    retry: 1,
    refetchInterval: 60_000,
  })

  const isLoading = meetingsLoading || employeesLoading || healthLoading

  if (isLoading) return <LoadingPage />

  if (meetingsError || employeesError) {
    return (
      <Page title="Meeting Intelligence Dashboard" subtitle="AI-powered enterprise meeting analytics">
        <ErrorBox
          message={
            meetingsError
              ? 'Failed to load meetings. Please try again.'
              : 'Failed to load employees. Please try again.'
          }
        />
      </Page>
    )
  }

  // ── Derived stats ──────────────────────────────────────────────────────────

  const totalMeetings = meetings?.length ?? 0
  const totalEmployees = employees?.length ?? 0
  const enrolledVoices = employees?.filter((e) => e.has_enrolled_voice).length ?? 0

  const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
  const recentActivity =
    meetings?.filter((m) => new Date(m.meeting_date) >= sevenDaysAgo).length ?? 0

  const recentMeetings = [...(meetings ?? [])]
    .sort((a, b) => new Date(b.meeting_date).getTime() - new Date(a.meeting_date).getTime())
    .slice(0, 5)

  // ── Chart data ─────────────────────────────────────────────────────────────

  // Meetings per day (last 7 days)
  const dailyCounts: Record<string, number> = {}
  for (let i = 6; i >= 0; i--) {
    const d = new Date(Date.now() - i * 24 * 60 * 60 * 1000)
    dailyCounts[d.toLocaleDateString('en-US', { weekday: 'short' })] = 0
  }
  meetings?.forEach((m) => {
    const d = new Date(m.meeting_date)
    if (d >= sevenDaysAgo) {
      const key = d.toLocaleDateString('en-US', { weekday: 'short' })
      if (key in dailyCounts) dailyCounts[key]++
    }
  })
  const barData = Object.entries(dailyCounts).map(([day, count]) => ({ day, count }))

  // Speaker enrollment pie
  const pieData = [
    { name: 'Enrolled', value: enrolledVoices, color: '#22d3ee' },
    { name: 'Not Enrolled', value: totalEmployees - enrolledVoices, color: '#334155' },
  ]

  // ── Health ─────────────────────────────────────────────────────────────────

  const dbOk = health?.database === 'connected' || health?.status === 'ok'
  const vectorOk = health?.vector_store === 'connected' || health?.vector_store_status === 'ok'
  const whisperModel: string = health?.whisper_model ?? health?.models?.whisper ?? '—'
  const geminiConfigured: boolean =
    health?.gemini_configured ?? health?.models?.gemini_configured ?? false

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <Page
      title="Meeting Intelligence Dashboard"
      subtitle="AI-powered enterprise meeting analytics"
    >
      {/* ── Stats Row ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
        <StatCard
          label="Total Meetings"
          value={totalMeetings}
          icon={<Mic2 className="w-5 h-5" />}
          sub="All time"
          color="text-brand-400"
        />
        <StatCard
          label="Total Employees"
          value={totalEmployees}
          icon={<Users className="w-5 h-5" />}
          sub="Registered profiles"
          color="text-violet-400"
        />
        <StatCard
          label="Enrolled Voices"
          value={enrolledVoices}
          icon={<CheckSquare className="w-5 h-5" />}
          sub={`${totalEmployees ? Math.round((enrolledVoices / totalEmployees) * 100) : 0}% coverage`}
          color="text-emerald-400"
        />
        <StatCard
          label="Recent Activity"
          value={recentActivity}
          icon={<TrendingUp className="w-5 h-5" />}
          sub="Last 7 days"
          color="text-amber-400"
        />
      </div>

      {/* ── Middle Row: Charts + System Health ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">

        {/* Meetings per day bar chart */}
        <div className="card p-5 lg:col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-4 h-4 text-brand-400" />
            <h3 className="font-semibold text-slate-200 text-sm">Meetings — Last 7 Days</h3>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={barData} barSize={28}>
              <XAxis
                dataKey="day"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                allowDecimals={false}
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                width={24}
              />
              <Tooltip
                contentStyle={{
                  background: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0',
                  fontSize: 13,
                }}
                cursor={{ fill: 'rgba(148,163,184,0.05)' }}
              />
              <Bar dataKey="count" fill="#22d3ee" radius={[4, 4, 0, 0]} name="Meetings" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Voice Enrollment pie */}
        <div className="card p-5 flex flex-col">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-4 h-4 text-violet-400" />
            <h3 className="font-semibold text-slate-200 text-sm">Voice Enrollment</h3>
          </div>
          <div className="flex-1 flex items-center justify-center">
            <PieChart width={160} height={160}>
              <Pie
                data={pieData}
                cx={80}
                cy={80}
                innerRadius={48}
                outerRadius={72}
                paddingAngle={3}
                dataKey="value"
                startAngle={90}
                endAngle={-270}
              >
                {pieData.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0',
                  fontSize: 13,
                }}
              />
            </PieChart>
          </div>
          <div className="flex justify-center gap-4 mt-2">
            {pieData.map((d) => (
              <div key={d.name} className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-sm"
                  style={{ background: d.color }}
                />
                <span className="text-xs text-slate-400">{d.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Bottom Row: Recent Meetings + Health + Quick Actions ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Recent Meetings */}
        <div className="card p-5 lg:col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <FileText className="w-4 h-4 text-brand-400" />
            <h3 className="font-semibold text-slate-200 text-sm">Recent Meetings</h3>
          </div>

          {recentMeetings.length === 0 ? (
            <p className="muted text-sm text-center py-10">No meetings recorded yet.</p>
          ) : (
            <div className="space-y-2">
              {recentMeetings.map((m) => (
                <div
                  key={m.id}
                  className="flex items-center justify-between gap-3 rounded-lg bg-slate-800/50 px-4 py-3 hover:bg-slate-800 transition-colors group"
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-slate-100 font-medium text-sm truncate">{m.title}</p>
                    <div className="flex items-center gap-3 mt-1 flex-wrap">
                      <span className="flex items-center gap-1 text-xs text-slate-400">
                        <Clock className="w-3 h-3" />
                        {formatDuration(m.duration_seconds)}
                      </span>
                      <span className="flex items-center gap-1 text-xs text-slate-400">
                        <Users className="w-3 h-3" />
                        {m.speakers.length} speaker{m.speakers.length !== 1 ? 's' : ''}
                      </span>
                      {m.speakers.some((s) => s.dominant_emotion) && (
                        <span
                          className={`text-xs font-medium ${emotionColor(
                            m.speakers[0]?.dominant_emotion ?? 'neutral'
                          )}`}
                        >
                          {m.speakers[0]?.dominant_emotion}
                        </span>
                      )}
                      <span className="flex items-center gap-1 text-xs text-slate-500">
                        <Calendar className="w-3 h-3" />
                        {formatTimeAgo(m.meeting_date)}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => navigate(`/meetings/${m.id}`)}
                    className="btn-secondary text-xs py-1 px-3 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    View
                  </button>
                </div>
              ))}
            </div>
          )}

          {totalMeetings > 5 && (
            <button
              onClick={() => navigate('/meetings')}
              className="mt-3 w-full text-xs text-brand-400 hover:text-brand-300 transition-colors py-2"
            >
              View all {totalMeetings} meetings →
            </button>
          )}
        </div>

        {/* Right column: System Health + Quick Actions */}
        <div className="flex flex-col gap-4">

          {/* System Health */}
          <div className="card p-5">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-4 h-4 text-emerald-400" />
              <h3 className="font-semibold text-slate-200 text-sm">System Health</h3>
              {healthLoading && (
                <span className="text-xs muted ml-auto">checking…</span>
              )}
            </div>

            {healthError ? (
              <p className="text-xs text-red-400">Health check unavailable.</p>
            ) : (
              <div>
                <HealthRow
                  label="Database"
                  value={health?.database ?? (dbOk ? 'Connected' : 'Unknown')}
                  ok={dbOk}
                />
                <HealthRow
                  label="Vector Store"
                  value={health?.vector_store ?? (vectorOk ? 'Connected' : 'Unknown')}
                  ok={vectorOk}
                />
                <HealthRow
                  label="Whisper Model"
                  value={whisperModel}
                />
                <HealthRow
                  label="Gemini AI"
                  value={geminiConfigured ? 'Configured' : 'Not Configured'}
                  ok={geminiConfigured}
                />
              </div>
            )}
          </div>

          {/* Quick Actions */}
          <div className="card p-5 flex-1">
            <h3 className="font-semibold text-slate-200 text-sm mb-3">Quick Actions</h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={() => navigate('/upload')}
                className="btn-primary w-full flex items-center gap-2 justify-center"
              >
                <Mic2 className="w-4 h-4" />
                Upload Meeting
              </button>
              <button
                onClick={() => navigate('/employees/register')}
                className="btn-secondary w-full flex items-center gap-2 justify-center"
              >
                <Users className="w-4 h-4" />
                Register Employee
              </button>
              <button
                onClick={() => navigate('/search')}
                className="btn-secondary w-full flex items-center gap-2 justify-center"
              >
                <Brain className="w-4 h-4" />
                Search Knowledge Base
              </button>
            </div>
          </div>
        </div>
      </div>
    </Page>
  )
}

import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from 'react-query'
import { meetingsApi, TranscriptSegment, EmployeeReport } from '../api'
import {
  formatDuration,
  formatTimestamp,
  formatDate,
  priorityColor,
  statusColor,
  emotionColor,
  accentBadge,
  categoryColor,
} from '../utils'
import { LoadingPage, ErrorBox, EmptyState, Spinner } from '../components/UI'
import {
  ArrowLeft,
  Download,
  Mic2,
  Clock,
  Users,
  FileText,
  BarChart2,
  BookOpen,
  ChevronDown,
  CheckSquare,
  AlertTriangle,
  Tag,
  MessageSquare,
  User,
  Calendar,
  Lightbulb,
  Target,
  TrendingUp,
  Brain,
} from 'lucide-react'
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

// ─── Types ────────────────────────────────────────────────────────────────────

type TabId =
  | 'overview'
  | 'transcript'
  | 'actions'
  | 'topics'
  | 'analytics'
  | 'reports'
  | 'manager'

interface Tab {
  id: TabId
  label: string
  icon: React.ReactNode
}

// ─── Constants ────────────────────────────────────────────────────────────────

const TABS: Tab[] = [
  { id: 'overview', label: 'Overview', icon: <BookOpen size={15} /> },
  { id: 'transcript', label: 'Transcript', icon: <MessageSquare size={15} /> },
  { id: 'actions', label: 'Action Items', icon: <CheckSquare size={15} /> },
  { id: 'topics', label: 'Topics & Keywords', icon: <Tag size={15} /> },
  { id: 'analytics', label: 'Analytics', icon: <BarChart2 size={15} /> },
  { id: 'reports', label: 'Employee Reports', icon: <Users size={15} /> },
  { id: 'manager', label: 'Manager Summary', icon: <Brain size={15} /> },
]

const PIE_COLORS = ['#6366f1', '#22d3ee', '#a78bfa', '#34d399', '#fb923c', '#f472b6', '#facc15']

// ─── Custom Tooltip for Charts ────────────────────────────────────────────────

function DarkTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      {label && <p className="text-slate-400 mb-1">{label}</p>}
      {payload.map((p: any, i: number) => (
        <p key={i} style={{ color: p.color ?? p.fill ?? '#e2e8f0' }}>
          {p.name}: <span className="font-semibold">{p.value}</span>
        </p>
      ))}
    </div>
  )
}

// ─── Overview Tab ─────────────────────────────────────────────────────────────

function OverviewTab({ meeting }: { meeting: any }) {
  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="card p-6">
        <h3 className="section-title mb-3 flex items-center gap-2">
          <FileText size={16} className="text-brand-400" />
          Meeting Summary
        </h3>
        {meeting.summary ? (
          <p className="text-slate-300 leading-relaxed text-sm">{meeting.summary}</p>
        ) : (
          <p className="muted italic">No summary available yet.</p>
        )}
      </div>

      {/* Speakers */}
      {meeting.speakers?.length > 0 && (
        <div>
          <h3 className="section-title mb-3 flex items-center gap-2">
            <Mic2 size={16} className="text-brand-400" />
            Speakers
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {meeting.speakers.map((sp: any) => (
              <div key={sp.id} className="card p-4 flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-full bg-brand-600/30 flex items-center justify-center text-brand-400 font-bold text-sm shrink-0">
                    {(sp.speaker_name ?? sp.speaker_label).charAt(0).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <p className="font-medium text-slate-100 truncate text-sm">
                      {sp.speaker_name ?? sp.speaker_label}
                    </p>
                    {sp.speaker_name && (
                      <p className="text-xs text-slate-500">{sp.speaker_label}</p>
                    )}
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 mt-1">
                  {sp.confidence_score != null && (
                    <span className="badge badge-blue text-xs">
                      {Math.round(sp.confidence_score * 100)}% conf.
                    </span>
                  )}
                  {sp.detected_accent && (
                    <span className={`badge ${accentBadge(sp.detected_accent)} text-xs`}>
                      {sp.detected_accent}
                    </span>
                  )}
                  {sp.dominant_emotion && (
                    <span className={`text-xs font-medium ${emotionColor(sp.dominant_emotion)}`}>
                      {sp.dominant_emotion}
                    </span>
                  )}
                </div>
                {sp.total_speaking_time != null && (
                  <p className="text-xs text-slate-500 flex items-center gap-1 mt-1">
                    <Clock size={11} />
                    {formatDuration(sp.total_speaking_time)} speaking
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Decisions & Next Steps */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Decisions */}
        <div className="card p-5">
          <h3 className="section-title mb-3 flex items-center gap-2">
            <Target size={16} className="text-emerald-400" />
            Decisions Made
          </h3>
          {meeting.decisions?.length > 0 ? (
            <ul className="space-y-2">
              {meeting.decisions.map((d: any, i: number) => (
                <li key={d.id ?? i} className="flex items-start gap-2 text-sm text-slate-300">
                  <span className="mt-0.5 w-5 h-5 rounded-full bg-emerald-900/40 text-emerald-400 flex items-center justify-center text-xs shrink-0 font-bold">
                    {i + 1}
                  </span>
                  <span className="leading-relaxed">{d.decision}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted text-sm">No decisions recorded.</p>
          )}
        </div>

        {/* Next Steps (Action Items Preview) */}
        <div className="card p-5">
          <h3 className="section-title mb-3 flex items-center gap-2">
            <TrendingUp size={16} className="text-violet-400" />
            Next Steps
          </h3>
          {meeting.action_items?.length > 0 ? (
            <ul className="space-y-2">
              {meeting.action_items.slice(0, 6).map((item: any, i: number) => (
                <li key={item.id ?? i} className="flex items-start gap-2 text-sm">
                  <span className="mt-0.5 shrink-0">
                    <CheckSquare size={14} className="text-violet-400" />
                  </span>
                  <div className="min-w-0">
                    <span className="text-slate-300">{item.task}</span>
                    {item.owner && (
                      <span className="text-slate-500 text-xs ml-2">→ {item.owner}</span>
                    )}
                  </div>
                </li>
              ))}
              {meeting.action_items.length > 6 && (
                <p className="muted text-xs">+{meeting.action_items.length - 6} more…</p>
              )}
            </ul>
          ) : (
            <p className="muted text-sm">No action items recorded.</p>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Transcript Tab ───────────────────────────────────────────────────────────

function TranscriptTab({ meetingId }: { meetingId: string }) {
  const [search, setSearch] = useState('')
  const { data, isLoading, error } = useQuery(
    ['transcript', meetingId],
    () => meetingsApi.getTranscript(meetingId),
    { staleTime: 5 * 60 * 1000 }
  )

  const segments: TranscriptSegment[] = data?.segments ?? []

  const filtered = segments.filter((seg) => {
    if (!search.trim()) return true
    const q = search.toLowerCase()
    return (
      seg.speaker.toLowerCase().includes(q) ||
      seg.text.toLowerCase().includes(q) ||
      seg.emotion?.toLowerCase().includes(q)
    )
  })

  if (isLoading) return <div className="flex justify-center py-16"><Spinner className="w-8 h-8" /></div>
  if (error) return <ErrorBox message="Failed to load transcript." />
  if (!segments.length)
    return (
      <EmptyState
        icon={<MessageSquare size={24} />}
        title="No transcript available"
        description="The transcript has not been generated yet."
      />
    )

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <input
          type="text"
          className="input w-full pl-9 text-sm"
          placeholder="Search by speaker, keyword, or emotion…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <MessageSquare size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
      </div>

      {/* Segments */}
      <div className="space-y-2">
        {filtered.length === 0 ? (
          <p className="muted text-center py-10">No segments match your search.</p>
        ) : (
          filtered.map((seg, i) => (
            <div
              key={seg.id ?? i}
              className="card p-4 flex gap-3 hover:border-slate-600 transition-colors"
            >
              {/* Timestamp */}
              <span className="shrink-0 font-mono text-xs bg-slate-800 text-slate-400 px-2 py-1 rounded-md h-fit mt-0.5 border border-slate-700">
                {formatTimestamp(seg.start_time)}
              </span>

              {/* Content */}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="text-sm font-semibold text-slate-100">{seg.speaker}</span>
                  {seg.emotion && (
                    <span className={`text-xs font-medium ${emotionColor(seg.emotion)}`}>
                      {seg.emotion}
                    </span>
                  )}
                  {seg.speaking_style && (
                    <span className="badge badge-slate text-xs">{seg.speaking_style}</span>
                  )}
                </div>
                <p className="text-slate-300 text-sm leading-relaxed">"{seg.text}"</p>
              </div>
            </div>
          ))
        )}
      </div>

      {filtered.length > 0 && (
        <p className="muted text-xs text-center">
          Showing {filtered.length} of {segments.length} segments
        </p>
      )}
    </div>
  )
}

// ─── Action Items Tab ─────────────────────────────────────────────────────────

function ActionItemsTab({ meeting }: { meeting: any }) {
  const items: any[] = meeting.action_items ?? []

  if (!items.length)
    return (
      <EmptyState
        icon={<CheckSquare size={24} />}
        title="No action items"
        description="No action items were extracted from this meeting."
      />
    )

  const sorted = [...items].sort((a, b) => {
    const order = { High: 0, Medium: 1, Low: 2 }
    return (order[a.priority as keyof typeof order] ?? 3) - (order[b.priority as keyof typeof order] ?? 3)
  })

  const groups = ['High', 'Medium', 'Low'] as const

  return (
    <div className="space-y-6">
      {groups.map((priority) => {
        const group = sorted.filter((i) => i.priority === priority)
        if (!group.length) return null
        return (
          <div key={priority}>
            <h3 className="section-title mb-3 flex items-center gap-2">
              <AlertTriangle
                size={15}
                className={
                  priority === 'High'
                    ? 'text-red-400'
                    : priority === 'Medium'
                    ? 'text-yellow-400'
                    : 'text-emerald-400'
                }
              />
              {priority} Priority
              <span className="badge badge-slate ml-1">{group.length}</span>
            </h3>
            <div className="card overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800">
                    <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium w-[40%]">Task</th>
                    <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium">Owner</th>
                    <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium">Deadline</th>
                    <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium">Priority</th>
                    <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {group.map((item, i) => (
                    <tr key={item.id ?? i} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-4 py-3 text-slate-200 font-medium">{item.task}</td>
                      <td className="px-4 py-3 text-slate-400">
                        <span className="flex items-center gap-1">
                          <User size={12} className="shrink-0" />
                          {item.owner || '—'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-slate-400">
                        {item.deadline ? (
                          <span className="flex items-center gap-1">
                            <Calendar size={12} className="shrink-0" />
                            {item.deadline}
                          </span>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`badge ${priorityColor(item.priority)}`}>
                          {item.priority}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`badge ${statusColor(item.status)}`}>
                          {item.status?.replace('_', ' ')}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ─── Topics & Keywords Tab ────────────────────────────────────────────────────

function TopicsTab({ meeting }: { meeting: any }) {
  const topics: any[] = meeting.topics ?? []
  const keywords: any[] = meeting.keywords ?? []

  const keywordChartData = keywords
    .sort((a, b) => b.relevance_score - a.relevance_score)
    .slice(0, 15)
    .map((kw) => ({ name: kw.keyword, score: Math.round(kw.relevance_score * 100) }))

  return (
    <div className="space-y-8">
      {/* Topics */}
      <div>
        <h3 className="section-title mb-3 flex items-center gap-2">
          <Lightbulb size={16} className="text-yellow-400" />
          Topics Discussed
          <span className="badge badge-slate ml-1">{topics.length}</span>
        </h3>
        {topics.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {topics.map((topic) => (
              <div key={topic.id} className="card p-4">
                <p className="font-semibold text-slate-100 text-sm mb-2">{topic.topic}</p>
                <div className="flex items-center gap-2 text-xs text-slate-500 mb-3">
                  <Clock size={11} />
                  {formatTimestamp(topic.start_time)} – {formatTimestamp(topic.end_time)}
                  <span className="badge badge-slate ml-auto">
                    {formatDuration(topic.duration_seconds)}
                  </span>
                </div>
                {topic.participants?.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {topic.participants.map((p: string, i: number) => (
                      <span key={i} className="badge badge-purple text-xs">{p}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="muted text-sm">No topics extracted.</p>
        )}
      </div>

      {/* Keywords */}
      <div>
        <h3 className="section-title mb-3 flex items-center gap-2">
          <Tag size={16} className="text-blue-400" />
          Keywords
          <span className="badge badge-slate ml-1">{keywords.length}</span>
        </h3>

        {keywords.length > 0 ? (
          <>
            {/* Grouped keyword badges */}
            {(() => {
              const grouped: Record<string, any[]> = {}
              keywords.forEach((kw) => {
                const cat = kw.category ?? 'general'
                ;(grouped[cat] = grouped[cat] ?? []).push(kw)
              })
              return (
                <div className="card p-5 space-y-3 mb-6">
                  {Object.entries(grouped).map(([cat, kws]) => (
                    <div key={cat} className="flex items-start gap-3">
                      <span className="text-xs text-slate-500 w-20 shrink-0 pt-0.5 capitalize">{cat}</span>
                      <div className="flex flex-wrap gap-2">
                        {kws.map((kw) => (
                          <span key={kw.id} className={`badge ${categoryColor(kw.category)} text-xs`}>
                            {kw.keyword}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )
            })()}

            {/* Bar chart */}
            {keywordChartData.length > 0 && (
              <div className="card p-5">
                <p className="text-sm font-medium text-slate-300 mb-4">Keyword Relevance Scores</p>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={keywordChartData}
                    layout="vertical"
                    margin={{ top: 0, right: 20, left: 10, bottom: 0 }}
                  >
                    <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={90}
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                    />
                    <Tooltip content={<DarkTooltip />} />
                    <Bar dataKey="score" fill="#6366f1" radius={[0, 4, 4, 0]} name="Score" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </>
        ) : (
          <p className="muted text-sm">No keywords extracted.</p>
        )}
      </div>
    </div>
  )
}

// ─── Analytics Tab ────────────────────────────────────────────────────────────

function AnalyticsTab({ meetingId }: { meetingId: string }) {
  const { data, isLoading, error } = useQuery(
    ['analytics', meetingId],
    () => meetingsApi.getAnalytics(meetingId),
    { staleTime: 5 * 60 * 1000 }
  )

  if (isLoading) return <div className="flex justify-center py-16"><Spinner className="w-8 h-8" /></div>
  if (error) return <ErrorBox message="Failed to load analytics." />
  if (!data) return <EmptyState icon={<BarChart2 size={24} />} title="No analytics available" />

  const pieData = data.speakers.map((sp) => ({
    name: sp.name,
    value: sp.speaking_time,
  }))

  const emotionData = data.overall_sentiment?.emotions_breakdown
    ? Object.entries(data.overall_sentiment.emotions_breakdown).map(([name, value]) => ({
        name,
        value: Math.round((value as number) * 100),
      }))
    : []

  const sentiment = data.overall_sentiment

  return (
    <div className="space-y-6">
      {/* Speaking time pie */}
      {pieData.length > 0 && (
        <div className="card p-5">
          <p className="text-sm font-medium text-slate-300 mb-4 flex items-center gap-2">
            <Mic2 size={15} className="text-brand-400" />
            Speaking Time Distribution
          </p>
          <div className="flex items-center gap-6 flex-wrap">
            <ResponsiveContainer width={220} height={220}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {pieData.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  content={<DarkTooltip />}
                  formatter={(v: number) => formatDuration(v)}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-col gap-2">
              {pieData.map((entry, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <span
                    className="w-3 h-3 rounded-full shrink-0"
                    style={{ background: PIE_COLORS[i % PIE_COLORS.length] }}
                  />
                  <span className="text-slate-300">{entry.name}</span>
                  <span className="text-slate-500 text-xs ml-1">
                    {formatDuration(entry.value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Sentiment breakdown */}
      {sentiment && (
        <div className="card p-5">
          <p className="text-sm font-medium text-slate-300 mb-4 flex items-center gap-2">
            <TrendingUp size={15} className="text-emerald-400" />
            Sentiment Breakdown
            <span className="badge badge-slate ml-auto capitalize">{sentiment.overall_sentiment}</span>
          </p>
          <div className="space-y-3">
            {[
              { label: 'Positive', value: sentiment.positive_percentage, color: 'bg-emerald-500' },
              { label: 'Neutral', value: sentiment.neutral_percentage, color: 'bg-slate-500' },
              { label: 'Negative', value: sentiment.negative_percentage, color: 'bg-red-500' },
            ].map(({ label, value, color }) => (
              <div key={label}>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>{label}</span>
                  <span>{Math.round(value)}%</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${color} rounded-full transition-all duration-700`}
                    style={{ width: `${Math.round(value)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Emotion breakdown bar chart */}
      {emotionData.length > 0 && (
        <div className="card p-5">
          <p className="text-sm font-medium text-slate-300 mb-4 flex items-center gap-2">
            <Brain size={15} className="text-violet-400" />
            Emotion Distribution
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={emotionData} margin={{ top: 0, right: 10, left: -10, bottom: 0 }}>
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip content={<DarkTooltip />} />
              <Bar dataKey="value" name="Score" radius={[4, 4, 0, 0]}>
                {emotionData.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Speaker comparison table */}
      {data.speakers.length > 0 && (
        <div className="card overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-800">
            <p className="text-sm font-medium text-slate-300 flex items-center gap-2">
              <Users size={15} className="text-cyan-400" />
              Speaker Comparison
            </p>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800">
                <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium">Speaker</th>
                <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium">Speaking Time</th>
                <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium">Confidence</th>
                <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium">Accent</th>
                <th className="text-left px-4 py-3 text-xs text-slate-400 font-medium">Emotion</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {data.speakers.map((sp, i) => (
                <tr key={i} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-4 py-3 font-medium text-slate-200">{sp.name}</td>
                  <td className="px-4 py-3 text-slate-400">{formatDuration(sp.speaking_time)}</td>
                  <td className="px-4 py-3 text-slate-400">
                    {sp.confidence != null ? `${Math.round(sp.confidence * 100)}%` : '—'}
                  </td>
                  <td className="px-4 py-3">
                    {sp.accent ? (
                      <span className={`badge ${accentBadge(sp.accent)}`}>{sp.accent}</span>
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {sp.dominant_emotion ? (
                      <span className={`text-xs font-medium ${emotionColor(sp.dominant_emotion)}`}>
                        {sp.dominant_emotion}
                      </span>
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ─── Employee Reports Tab ─────────────────────────────────────────────────────

function EmployeeReportsTab({ meetingId }: { meetingId: string }) {
  const { data, isLoading, error } = useQuery(
    ['employee-reports', meetingId],
    () => meetingsApi.getEmployeeReports(meetingId),
    { staleTime: 5 * 60 * 1000 }
  )

  if (isLoading) return <div className="flex justify-center py-16"><Spinner className="w-8 h-8" /></div>
  if (error) return <ErrorBox message="Failed to load employee reports." />
  if (!data?.length)
    return (
      <EmptyState
        icon={<Users size={24} />}
        title="No employee reports"
        description="Employee-specific reports are generated after processing."
      />
    )

  const reports: EmployeeReport[] = data

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      {reports.map((report) => (
        <div key={report.id} className="card p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-9 h-9 rounded-full bg-violet-900/40 flex items-center justify-center text-violet-400 font-bold shrink-0">
              {report.employee_name.charAt(0).toUpperCase()}
            </div>
            <div>
              <p className="font-semibold text-slate-100">{report.employee_name}</p>
              {report.employee_id && (
                <p className="text-xs text-slate-500">{report.employee_id}</p>
              )}
            </div>
          </div>

          <div className="space-y-3 text-sm">
            {report.topics_discussed?.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide mb-1.5 flex items-center gap-1">
                  <Tag size={10} /> Topics
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {report.topics_discussed.map((t, i) => (
                    <span key={i} className="badge badge-blue text-xs">{t}</span>
                  ))}
                </div>
              </div>
            )}

            {report.decisions_affecting?.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide mb-1.5 flex items-center gap-1">
                  <Target size={10} /> Decisions
                </p>
                <ul className="space-y-1">
                  {report.decisions_affecting.map((d, i) => (
                    <li key={i} className="text-slate-400 text-xs flex items-start gap-1.5">
                      <span className="text-emerald-500 mt-0.5">•</span>
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {report.assigned_tasks?.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide mb-1.5 flex items-center gap-1">
                  <CheckSquare size={10} /> Assigned Tasks
                </p>
                <ul className="space-y-1">
                  {report.assigned_tasks.map((t, i) => (
                    <li key={i} className="text-slate-400 text-xs flex items-start gap-1.5">
                      <span className="text-violet-400 mt-0.5">→</span>
                      {t}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {report.mentioned_deadlines?.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide mb-1.5 flex items-center gap-1">
                  <Calendar size={10} /> Deadlines
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {report.mentioned_deadlines.map((d, i) => (
                    <span key={i} className="badge badge-yellow text-xs">{d}</span>
                  ))}
                </div>
              </div>
            )}

            {report.follow_ups?.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-wide mb-1.5 flex items-center gap-1">
                  <TrendingUp size={10} /> Follow-ups
                </p>
                <ul className="space-y-1">
                  {report.follow_ups.map((f, i) => (
                    <li key={i} className="text-slate-400 text-xs flex items-start gap-1.5">
                      <span className="text-cyan-400 mt-0.5">↻</span>
                      {f}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Manager Summary Tab ──────────────────────────────────────────────────────

function ManagerSummaryTab({ meetingId, meetingTitle }: { meetingId: string; meetingTitle: string }) {
  const { data, isLoading, error } = useQuery(
    ['manager-summary', meetingId],
    () => meetingsApi.getManagerSummary(meetingId),
    { staleTime: 5 * 60 * 1000 }
  )

  if (isLoading) return <div className="flex justify-center py-16"><Spinner className="w-8 h-8" /></div>
  if (error) return <ErrorBox message="Failed to load manager summary." />
  if (!data)
    return (
      <EmptyState
        icon={<Brain size={24} />}
        title="No manager summary available"
        description="The manager summary is generated after full meeting processing."
      />
    )

  const sections = [
    {
      icon: <FileText size={15} className="text-brand-400" />,
      title: 'Executive Summary',
      content: data.executive_summary ? (
        <p className="text-slate-300 text-sm leading-relaxed">{data.executive_summary}</p>
      ) : null,
    },
    {
      icon: <Target size={15} className="text-emerald-400" />,
      title: 'Key Decisions',
      items: data.key_decisions,
      color: 'text-emerald-400',
    },
    {
      icon: <AlertTriangle size={15} className="text-red-400" />,
      title: 'Risks Identified',
      items: data.risks,
      color: 'text-red-400',
    },
    {
      icon: <Clock size={15} className="text-yellow-400" />,
      title: 'Delays',
      items: data.delays,
      color: 'text-yellow-400',
    },
    {
      icon: <AlertTriangle size={15} className="text-orange-400" />,
      title: 'Team Blockers',
      items: data.team_blockers,
      color: 'text-orange-400',
    },
    {
      icon: <CheckSquare size={15} className="text-violet-400" />,
      title: 'Assigned Tasks',
      items: data.assigned_tasks,
      color: 'text-violet-400',
    },
    {
      icon: <TrendingUp size={15} className="text-cyan-400" />,
      title: 'Follow-up Actions',
      items: data.follow_up_actions,
      color: 'text-cyan-400',
    },
  ]

  return (
    <div className="space-y-5">
      {/* Export button */}
      <div className="flex justify-end">
        <button
          onClick={() => window.print()}
          className="btn-secondary flex items-center gap-2 text-sm"
        >
          <Download size={14} />
          Print / Export Summary
        </button>
      </div>

      {sections.map((section, i) => {
        if (section.content !== undefined) {
          // Executive summary (raw content)
          return section.content ? (
            <div key={i} className="card p-5">
              <h3 className="section-title mb-3 flex items-center gap-2">
                {section.icon}
                {section.title}
              </h3>
              {section.content}
            </div>
          ) : null
        }

        const items = (section.items ?? []) as string[]
        if (!items.length) return null

        return (
          <div key={i} className="card p-5">
            <h3 className="section-title mb-3 flex items-center gap-2">
              {section.icon}
              {section.title}
              <span className="badge badge-slate ml-1">{items.length}</span>
            </h3>
            <ul className="space-y-2">
              {items.map((item, j) => (
                <li key={j} className="flex items-start gap-2 text-sm text-slate-300">
                  <span className={`shrink-0 mt-1 ${section.color}`}>•</span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )
      })}
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function MeetingDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<TabId>('overview')

  const { data: meeting, isLoading, error } = useQuery(
    ['meeting', id],
    () => meetingsApi.get(id!),
    { enabled: !!id, staleTime: 2 * 60 * 1000 }
  )

  if (isLoading) return <LoadingPage />
  if (error || !meeting)
    return (
      <div className="p-6">
        <ErrorBox message="Failed to load meeting details. Please try again." />
      </div>
    )

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      {/* ── Header ── */}
      <div className="flex flex-col gap-4">
        {/* Back + actions row */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <button
            onClick={() => navigate(-1)}
            className="btn-secondary flex items-center gap-1.5 text-sm"
          >
            <ArrowLeft size={15} />
            Back
          </button>

          <button
            onClick={() => window.open(meetingsApi.downloadMomPdf(meeting.id), '_blank')}
            className="btn-primary flex items-center gap-2 text-sm"
          >
            <Download size={14} />
            Download MOM PDF
          </button>
        </div>

        {/* Meeting info */}
        <div>
          <h1 className="text-2xl font-bold text-slate-100 leading-snug">{meeting.title}</h1>
          <div className="flex items-center gap-3 mt-2 flex-wrap">
            <span className="muted flex items-center gap-1 text-sm">
              <Calendar size={13} />
              {formatDate(meeting.meeting_date)}
            </span>
            <span className="muted flex items-center gap-1 text-sm">
              <Clock size={13} />
              {formatDuration(meeting.duration_seconds)}
            </span>
            {meeting.source_type && (
              <span className="badge badge-blue capitalize">{meeting.source_type}</span>
            )}
            {meeting.media_type && (
              <span className="badge badge-purple capitalize">{meeting.media_type}</span>
            )}
            <span className="badge badge-slate text-xs">{meeting.audio_file_name}</span>
          </div>
        </div>
      </div>

      {/* ── Tabs ── */}
      <div className="border-b border-slate-800 overflow-x-auto">
        <nav className="flex gap-0 min-w-max">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center gap-1.5 px-4 py-3 text-sm font-medium border-b-2 transition-all whitespace-nowrap
                ${
                  activeTab === tab.id
                    ? 'border-brand-500 text-brand-400'
                    : 'border-transparent text-slate-500 hover:text-slate-300 hover:border-slate-700'
                }
              `}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* ── Tab Content ── */}
      <div>
        {activeTab === 'overview' && <OverviewTab meeting={meeting} />}
        {activeTab === 'transcript' && <TranscriptTab meetingId={meeting.id} />}
        {activeTab === 'actions' && <ActionItemsTab meeting={meeting} />}
        {activeTab === 'topics' && <TopicsTab meeting={meeting} />}
        {activeTab === 'analytics' && <AnalyticsTab meetingId={meeting.id} />}
        {activeTab === 'reports' && <EmployeeReportsTab meetingId={meeting.id} />}
        {activeTab === 'manager' && (
          <ManagerSummaryTab meetingId={meeting.id} meetingTitle={meeting.title} />
        )}
      </div>
    </div>
  )
}

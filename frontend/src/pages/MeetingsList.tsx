import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { useNavigate } from 'react-router-dom'
import toast from 'react-hot-toast'
import {
  Mic2,
  Clock,
  Users,
  FileText,
  Download,
  Eye,
  Trash2,
  Calendar,
  Search,
  Upload,
  ChevronUp,
  ChevronDown,
  ChevronsUpDown,
  Video,
  Radio,
} from 'lucide-react'
import { meetingsApi, Meeting } from '../api'
import { formatDuration, formatDate, formatTimeAgo } from '../utils'
import { LoadingPage, EmptyState, Page } from '../components/UI'

// ─── Types ────────────────────────────────────────────────────────────────────

type SortKey = 'title' | 'meeting_date' | 'duration_seconds' | 'speakers'
type SortDir = 'asc' | 'desc'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function MediaBadge({ type }: { type: string }) {
  const isVideo = type?.toLowerCase() === 'video'
  return (
    <span
      className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full border ${
        isVideo
          ? 'bg-violet-900/40 border-violet-700/60 text-violet-300'
          : 'bg-blue-900/40 border-blue-700/60 text-blue-300'
      }`}
    >
      {isVideo ? <Video className="w-3 h-3" /> : <Mic2 className="w-3 h-3" />}
      {isVideo ? 'Video' : 'Audio'}
    </span>
  )
}

function SourceBadge({ type }: { type: string }) {
  const isLive = type?.toLowerCase() === 'live'
  return (
    <span
      className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full border ${
        isLive
          ? 'bg-emerald-900/40 border-emerald-700/60 text-emerald-300'
          : 'bg-slate-800 border-slate-700 text-slate-400'
      }`}
    >
      {isLive ? <Radio className="w-3 h-3" /> : <Upload className="w-3 h-3" />}
      {isLive ? 'Live' : 'Uploaded'}
    </span>
  )
}

function SortIcon({ active, dir }: { active: boolean; dir: SortDir }) {
  if (!active) return <ChevronsUpDown className="w-3.5 h-3.5 text-slate-600" />
  return dir === 'asc'
    ? <ChevronUp className="w-3.5 h-3.5 text-brand-400" />
    : <ChevronDown className="w-3.5 h-3.5 text-brand-400" />
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function MeetingsList() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('meeting_date')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [deletingId, setDeletingId] = useState<string | null>(null)

  // ── Data fetching ──────────────────────────────────────────────────────────

  const { data: meetings, isLoading, isError } = useQuery<Meeting[]>(
    ['meetings'],
    () => meetingsApi.list(100),
    { staleTime: 30_000 },
  )

  // ── Delete mutation ────────────────────────────────────────────────────────

  const deleteMutation = useMutation(
    (id: string) => meetingsApi.delete(id),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['meetings'])
        toast.success('Meeting deleted successfully')
        setDeletingId(null)
      },
      onError: () => {
        toast.error('Failed to delete meeting. Please try again.')
        setDeletingId(null)
      },
    },
  )

  // ── Handlers ───────────────────────────────────────────────────────────────

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  function handleDelete(meeting: Meeting) {
    const confirmed = window.confirm(
      `Are you sure you want to delete "${meeting.title}"? This action cannot be undone.`,
    )
    if (!confirmed) return
    setDeletingId(meeting.id)
    deleteMutation.mutate(meeting.id)
  }

  function handleDownload(id: string, e: React.MouseEvent) {
    e.stopPropagation()
    window.open(meetingsApi.downloadMomPdf(id), '_blank')
  }

  // ── Filtered + sorted data ─────────────────────────────────────────────────

  const filtered = useMemo(() => {
    if (!meetings) return []
    const q = search.trim().toLowerCase()
    const result = q
      ? meetings.filter(m => m.title.toLowerCase().includes(q))
      : [...meetings]

    result.sort((a, b) => {
      let cmp = 0
      if (sortKey === 'title') {
        cmp = a.title.localeCompare(b.title)
      } else if (sortKey === 'meeting_date') {
        cmp = (a.meeting_date ?? '').localeCompare(b.meeting_date ?? '')
      } else if (sortKey === 'duration_seconds') {
        cmp = (a.duration_seconds ?? 0) - (b.duration_seconds ?? 0)
      } else if (sortKey === 'speakers') {
        cmp = (a.speakers?.length ?? 0) - (b.speakers?.length ?? 0)
      }
      return sortDir === 'asc' ? cmp : -cmp
    })

    return result
  }, [meetings, search, sortKey, sortDir])

  // ── Column header helper ────────────────────────────────────────────────────

  function ColHeader({
    label,
    sortable,
    col,
    className = '',
  }: {
    label: string
    sortable?: SortKey
    col?: SortKey
    className?: string
  }) {
    const active = !!sortable && sortKey === sortable
    return (
      <th
        className={`px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider select-none ${
          sortable ? 'cursor-pointer hover:text-slate-200 transition-colors' : ''
        } ${className}`}
        onClick={sortable ? () => handleSort(sortable) : undefined}
      >
        <div className="flex items-center gap-1">
          {label}
          {sortable && <SortIcon active={active} dir={sortDir} />}
        </div>
      </th>
    )
  }

  // ── Render states ──────────────────────────────────────────────────────────

  if (isLoading) return <LoadingPage />

  if (isError) {
    return (
      <Page title="Meetings" subtitle="All processed meeting recordings">
        <div className="rounded-xl bg-red-900/20 border border-red-800/60 p-6 text-red-300 text-sm text-center">
          Failed to load meetings. Please refresh the page.
        </div>
      </Page>
    )
  }

  // ── Main render ─────────────────────────────────────────────────────────────

  return (
    <Page
      title="Meetings"
      subtitle="All processed meeting recordings"
      action={
        <button
          className="btn-primary flex items-center gap-2"
          onClick={() => navigate('/upload')}
        >
          <Upload className="w-4 h-4" />
          Upload Meeting
        </button>
      }
    >
      {/* ── Search & Stats bar ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="relative w-full sm:max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
          <input
            type="text"
            className="input pl-9 w-full"
            placeholder="Search meetings…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        <p className="muted text-sm shrink-0">
          {filtered.length} {filtered.length === 1 ? 'meeting' : 'meetings'}
          {search && ` matching "${search}"`}
        </p>
      </div>

      {/* ── Table ── */}
      {filtered.length === 0 ? (
        <EmptyState
          icon={<Mic2 className="w-6 h-6" />}
          title={search ? 'No meetings match your search' : 'No meetings yet'}
          description={
            search
              ? 'Try a different search term.'
              : 'Upload your first meeting recording to get started.'
          }
        />
      ) : (
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/60">
                  <ColHeader label="Title" sortable="title" className="min-w-[200px]" />
                  <ColHeader label="Date" sortable="meeting_date" className="min-w-[130px]" />
                  <ColHeader label="Duration" sortable="duration_seconds" className="min-w-[100px]" />
                  <ColHeader label="Speakers" sortable="speakers" className="min-w-[90px]" />
                  <ColHeader label="Media" className="min-w-[90px]" />
                  <ColHeader label="Source" className="min-w-[90px]" />
                  <th className="px-4 py-3 text-right text-xs font-semibold text-slate-400 uppercase tracking-wider min-w-[120px]">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filtered.map(meeting => (
                  <tr
                    key={meeting.id}
                    className="group hover:bg-slate-800/40 transition-colors cursor-pointer"
                    onClick={() => navigate(`/meetings/${meeting.id}`)}
                  >
                    {/* Title */}
                    <td className="px-4 py-3.5">
                      <div className="flex items-start gap-2.5">
                        <div className="mt-0.5 w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 group-hover:border-brand-600 transition-colors">
                          <FileText className="w-3.5 h-3.5 text-slate-400 group-hover:text-brand-400 transition-colors" />
                        </div>
                        <div className="min-w-0">
                          <p className="font-medium text-slate-200 truncate max-w-[260px] group-hover:text-brand-300 transition-colors">
                            {meeting.title}
                          </p>
                          <p className="text-xs text-slate-500 mt-0.5">
                            {formatTimeAgo(meeting.meeting_date)}
                          </p>
                        </div>
                      </div>
                    </td>

                    {/* Date */}
                    <td className="px-4 py-3.5">
                      <div className="flex items-center gap-1.5 text-slate-300">
                        <Calendar className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                        <span>{formatDate(meeting.meeting_date)}</span>
                      </div>
                    </td>

                    {/* Duration */}
                    <td className="px-4 py-3.5">
                      <div className="flex items-center gap-1.5 text-slate-300">
                        <Clock className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                        <span className="tabular-nums">
                          {formatDuration(meeting.duration_seconds)}
                        </span>
                      </div>
                    </td>

                    {/* Speakers */}
                    <td className="px-4 py-3.5">
                      <div className="flex items-center gap-1.5 text-slate-300">
                        <Users className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                        <span>{meeting.speakers?.length ?? 0}</span>
                      </div>
                    </td>

                    {/* Media type */}
                    <td className="px-4 py-3.5">
                      <MediaBadge type={meeting.media_type} />
                    </td>

                    {/* Source type */}
                    <td className="px-4 py-3.5">
                      <SourceBadge type={meeting.source_type} />
                    </td>

                    {/* Actions */}
                    <td className="px-4 py-3.5" onClick={e => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-1">
                        {/* View */}
                        <button
                          title="View meeting"
                          className="p-1.5 rounded-lg text-slate-400 hover:text-brand-300 hover:bg-slate-700 transition-colors"
                          onClick={e => {
                            e.stopPropagation()
                            navigate(`/meetings/${meeting.id}`)
                          }}
                        >
                          <Eye className="w-4 h-4" />
                        </button>

                        {/* Download PDF */}
                        <button
                          title="Download MoM PDF"
                          className="p-1.5 rounded-lg text-slate-400 hover:text-emerald-300 hover:bg-slate-700 transition-colors"
                          onClick={e => handleDownload(meeting.id, e)}
                        >
                          <Download className="w-4 h-4" />
                        </button>

                        {/* Delete */}
                        <button
                          title="Delete meeting"
                          disabled={deletingId === meeting.id}
                          className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-slate-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                          onClick={e => {
                            e.stopPropagation()
                            handleDelete(meeting)
                          }}
                        >
                          {deletingId === meeting.id ? (
                            <svg
                              className="w-4 h-4 animate-spin"
                              viewBox="0 0 24 24"
                              fill="none"
                            >
                              <circle
                                className="opacity-25"
                                cx="12" cy="12" r="10"
                                stroke="currentColor" strokeWidth="4"
                              />
                              <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8v8H4z"
                              />
                            </svg>
                          ) : (
                            <Trash2 className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ── Footer ── */}
          <div className="px-4 py-3 border-t border-slate-800 bg-slate-900/40 flex items-center justify-between gap-4">
            <p className="muted text-xs">
              Showing {filtered.length} of {meetings?.length ?? 0} meetings
            </p>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500">Sort:</span>
              <span className="text-xs text-slate-300 capitalize">
                {sortKey === 'meeting_date'
                  ? 'Date'
                  : sortKey === 'duration_seconds'
                  ? 'Duration'
                  : sortKey.charAt(0).toUpperCase() + sortKey.slice(1)}{' '}
                ({sortDir === 'asc' ? '↑' : '↓'})
              </span>
            </div>
          </div>
        </div>
      )}
    </Page>
  )
}

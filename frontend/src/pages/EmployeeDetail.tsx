import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from 'react-query'
import { employeesApi } from '../api'
import { formatDate, formatTimeAgo } from '../utils'
import { LoadingPage, ErrorBox, Page } from '../components/UI'
import {
  User,
  Building2,
  Mail,
  Briefcase,
  Mic2,
  CheckCircle,
  XCircle,
  Upload,
  ArrowLeft,
  Calendar,
  Shield,
} from 'lucide-react'
import { useState, useRef, useCallback } from 'react'
import toast from 'react-hot-toast'
import { useDropzone } from 'react-dropzone'

// ─── Voice prompts ────────────────────────────────────────────────────────────
const VOICE_PROMPTS = [
  'The quick brown fox jumps over the lazy dog',
  'She sells seashells by the seashore on sunny days',
  'How much wood would a woodchuck chuck if a woodchuck could chuck wood',
  'Peter Piper picked a peck of pickled peppers',
  'Around the rugged rock the ragged rascal ran',
]

// ─── Avatar helper ────────────────────────────────────────────────────────────
function getInitials(name: string): string {
  return name
    .split(' ')
    .slice(0, 2)
    .map((n) => n[0])
    .join('')
    .toUpperCase()
}

// ─── Profile Info Row ─────────────────────────────────────────────────────────
function InfoRow({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: string
}) {
  return (
    <div className="flex items-start gap-3 py-3 border-b border-slate-800 last:border-0">
      <div className="mt-0.5 text-slate-500 shrink-0">{icon}</div>
      <div className="min-w-0">
        <p className="text-xs text-slate-500 mb-0.5">{label}</p>
        <p className="text-slate-200 text-sm font-medium break-all">{value}</p>
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function EmployeeDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  // Pick a stable prompt index per employee id
  const promptIndex = id
    ? Math.abs(
        id.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0)
      ) % VOICE_PROMPTS.length
    : 0
  const promptText = VOICE_PROMPTS[promptIndex]

  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)

  // ── Fetch employee ──────────────────────────────────────────────────────────
  const {
    data: employee,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery(['employee', id], () => employeesApi.get(id!), {
    enabled: !!id,
    staleTime: 30_000,
  })

  // ── Dropzone ────────────────────────────────────────────────────────────────
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0])
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/wav': ['.wav'],
      'audio/mpeg': ['.mp3'],
      'audio/webm': ['.webm'],
      'audio/mp4': ['.m4a'],
    },
    maxFiles: 1,
    multiple: false,
  })

  // ── Upload handler ──────────────────────────────────────────────────────────
  async function handleUpload() {
    if (!selectedFile || !id) return
    setUploading(true)
    try {
      await employeesApi.uploadVoiceSample(id, selectedFile, promptText)
      toast.success('Voice sample uploaded successfully!')
      setSelectedFile(null)
      refetch()
    } catch (err: any) {
      toast.error(
        err?.response?.data?.detail ?? 'Failed to upload voice sample'
      )
    } finally {
      setUploading(false)
    }
  }

  // ── States ──────────────────────────────────────────────────────────────────
  if (isLoading) return <LoadingPage />
  if (isError || !employee) {
    return (
      <Page>
        <ErrorBox
          message={
            (error as any)?.response?.data?.detail ??
            'Failed to load employee profile.'
          }
        />
      </Page>
    )
  }

  const enrolled = employee.has_enrolled_voice

  return (
    <div className="p-6 max-w-screen-lg mx-auto space-y-6">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate(-1)}
          className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-100 transition-colors"
          aria-label="Go back"
        >
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold text-slate-100 truncate">
              {employee.name}
            </h1>
            <span className="badge badge-slate font-mono text-xs">
              {employee.employee_id}
            </span>
          </div>
          <p className="muted text-sm mt-0.5">
            {employee.designation} · {employee.department}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── Left column: Profile card ─────────────────────────────────────── */}
        <div className="lg:col-span-1 space-y-6">
          <div className="card p-6">
            {/* Avatar */}
            <div className="flex flex-col items-center text-center mb-6">
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-brand-600 to-brand-800 flex items-center justify-center text-white text-2xl font-bold mb-3 shadow-lg shadow-brand-900/40">
                {getInitials(employee.name)}
              </div>
              <h2 className="text-lg font-semibold text-slate-100">
                {employee.name}
              </h2>
              <p className="text-sm text-slate-400">{employee.designation}</p>
              {enrolled ? (
                <span className="badge badge-green mt-2 flex items-center gap-1">
                  <CheckCircle className="w-3 h-3" /> Voice Enrolled
                </span>
              ) : (
                <span className="badge badge-yellow mt-2 flex items-center gap-1">
                  <XCircle className="w-3 h-3" /> Voice Pending
                </span>
              )}
            </div>

            {/* Info rows */}
            <div>
              <InfoRow
                icon={<Building2 className="w-4 h-4" />}
                label="Department"
                value={employee.department}
              />
              <InfoRow
                icon={<Briefcase className="w-4 h-4" />}
                label="Team"
                value={employee.team}
              />
              <InfoRow
                icon={<Mail className="w-4 h-4" />}
                label="Email"
                value={employee.email}
              />
              <InfoRow
                icon={<Shield className="w-4 h-4" />}
                label="Employee ID"
                value={employee.employee_id}
              />
              <InfoRow
                icon={<Calendar className="w-4 h-4" />}
                label="Registered"
                value={`${formatDate(employee.created_at)} (${formatTimeAgo(employee.created_at)})`}
              />
            </div>
          </div>
        </div>

        {/* ── Right column ──────────────────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-6">

          {/* Voice Enrollment Status Card */}
          <div className="card p-6">
            <h3 className="section-title mb-4 flex items-center gap-2">
              <Mic2 className="w-4 h-4 text-brand-400" />
              Voice Enrollment Status
            </h3>

            <div
              className={`rounded-xl p-5 flex items-start gap-4 border ${
                enrolled
                  ? 'bg-emerald-950/40 border-emerald-800'
                  : 'bg-amber-950/40 border-amber-800'
              }`}
            >
              <div
                className={`mt-0.5 shrink-0 ${
                  enrolled ? 'text-emerald-400' : 'text-amber-400'
                }`}
              >
                {enrolled ? (
                  <CheckCircle className="w-7 h-7" />
                ) : (
                  <XCircle className="w-7 h-7" />
                )}
              </div>
              <div>
                <p
                  className={`font-semibold text-base ${
                    enrolled ? 'text-emerald-300' : 'text-amber-300'
                  }`}
                >
                  {enrolled ? 'Voice Enrolled' : 'Voice Pending'}
                </p>
                <p className="text-sm text-slate-400 mt-1">
                  {enrolled
                    ? 'Voice profile active — this employee will be recognized in meetings.'
                    : 'No voice samples yet — upload samples to enable speaker recognition.'}
                </p>
              </div>
              <div className="ml-auto shrink-0 text-right">
                <p className="text-2xl font-bold text-slate-100">
                  {employee.voice_samples_count}
                </p>
                <p className="text-xs text-slate-500 mt-0.5">
                  {employee.voice_samples_count === 1 ? 'sample' : 'samples'}
                </p>
              </div>
            </div>
          </div>

          {/* Upload Voice Sample */}
          <div className="card p-6">
            <h3 className="section-title mb-1 flex items-center gap-2">
              <Upload className="w-4 h-4 text-brand-400" />
              Upload Voice Sample
            </h3>
            <p className="muted text-sm mb-4">
              Read the following sentence clearly and upload the recording:
            </p>

            {/* Prompt text */}
            <div className="rounded-lg bg-slate-800/60 border border-slate-700 px-4 py-3 mb-5">
              <p className="text-slate-200 text-sm italic">
                &ldquo;{promptText}&rdquo;
              </p>
            </div>

            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`rounded-xl border-2 border-dashed px-6 py-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-brand-500 bg-brand-950/30'
                  : 'border-slate-700 bg-slate-900/40 hover:border-brand-600 hover:bg-brand-950/20'
              }`}
            >
              <input {...getInputProps()} />
              <Mic2
                className={`w-8 h-8 mx-auto mb-3 ${
                  isDragActive ? 'text-brand-400' : 'text-slate-500'
                }`}
              />
              {isDragActive ? (
                <p className="text-brand-300 text-sm font-medium">
                  Drop the audio file here…
                </p>
              ) : (
                <>
                  <p className="text-slate-300 text-sm font-medium">
                    Drag &amp; drop an audio file here
                  </p>
                  <p className="text-slate-500 text-xs mt-1">
                    or click to browse · .wav, .mp3, .webm, .m4a
                  </p>
                </>
              )}
            </div>

            {/* Selected file info */}
            {selectedFile && (
              <div className="mt-3 rounded-lg bg-slate-800 border border-slate-700 px-4 py-3 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0">
                  <Mic2 className="w-4 h-4 text-brand-400 shrink-0" />
                  <div className="min-w-0">
                    <p className="text-sm text-slate-200 truncate">
                      {selectedFile.name}
                    </p>
                    <p className="text-xs text-slate-500">
                      {(selectedFile.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="text-slate-500 hover:text-slate-300 transition-colors shrink-0 text-xs"
                >
                  Remove
                </button>
              </div>
            )}

            {/* Upload button */}
            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              className="btn-primary mt-4 w-full flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {uploading ? (
                <>
                  <svg
                    className="animate-spin w-4 h-4"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v8H4z"
                    />
                  </svg>
                  Uploading…
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  Upload Sample
                </>
              )}
            </button>
          </div>

          {/* Meetings info card */}
          <div className="card p-6 flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg bg-slate-800 flex items-center justify-center text-brand-400 shrink-0">
              <User className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-200 mb-1">
                Meetings Featuring This Employee
              </h3>
              <p className="text-sm text-slate-400">
                Search for meetings featuring{' '}
                <span className="text-slate-200 font-medium">
                  {employee.name}
                </span>{' '}
                in the{' '}
                <span className="text-brand-400 font-medium">
                  Knowledge Base
                </span>
                . Use the employee's name as a search query to find all
                meetings, summaries, and action items they appeared in.
              </p>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}

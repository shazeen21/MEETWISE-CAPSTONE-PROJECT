import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { meetingsApi } from '../api'
import { Page, Spinner } from '../components/UI'
import { Upload, FileAudio, FileVideo, CheckCircle, XCircle, Info } from 'lucide-react'
import toast from 'react-hot-toast'
import { formatDuration } from '../utils'

// ─── Processing Steps ─────────────────────────────────────────────────────────

// Mirrors PIPELINE_STAGES in backend/app/pipeline.py; replaced by the server's list once polling starts.
const PROCESSING_STEPS = [
  'Audio Cleaning & Noise Reduction',
  'Speech-to-Text (WhisperX)',
  'Speaker Diarization',
  'Transcript & Speaker Alignment',
  'Speaker Identification',
  'Meeting Intelligence (MOM, Action Items, Topics, Emotions)',
  'Saving Meeting Records',
  'ChromaDB Indexing',
  'MOM PDF Generation',
  'Participant Notifications',
]

const PROGRESS_POLL_MS = 2000

// ─── Accepted File Types ───────────────────────────────────────────────────────

const ACCEPTED_TYPES = {
  'audio/mpeg': ['.mp3'],
  'audio/wav': ['.wav'],
  'audio/x-wav': ['.wav'],
  'audio/mp4': ['.m4a'],
  'audio/x-m4a': ['.m4a'],
  'video/mp4': ['.mp4'],
  'video/webm': ['.webm'],
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function isVideoFile(file: File): boolean {
  return file.type.startsWith('video/')
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function UploadMeeting() {
  const navigate = useNavigate()

  // Form state
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [numSpeakers, setNumSpeakers] = useState('')
  const [sourceType, setSourceType] = useState<'uploaded' | 'live'>('uploaded')
  const [processImmediately, setProcessImmediately] = useState(true)

  // Upload state
  const [isUploading, setIsUploading] = useState(false)
  // Index of the stage the backend is running; every earlier stage is done.
  const [activeStep, setActiveStep] = useState<number>(-1)
  const [steps, setSteps] = useState<string[]>(PROCESSING_STEPS)
  // The backend runs one meeting at a time; later uploads wait their turn.
  const [isQueued, setIsQueued] = useState(false)
  const [result, setResult] = useState<{ id: string; title: string } | null>(null)
  const [error, setError] = useState<string | null>(null)

  // ─── Dropzone ──────────────────────────────────────────────────────────────

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length > 0) {
      setFile(accepted[0])
      setError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    disabled: isUploading,
  })

  // ─── Live Progress ────────────────────────────────────────────────────────

  // Processing runs in the background on the server; follow it until it ends.
  // Resolves on completion, rejects with the server's error on failure.
  function waitForProcessing(meetingId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      let started = false
      const interval = setInterval(async () => {
        try {
          const p = await meetingsApi.getProgress(meetingId)
          if (p.stages?.length) setSteps(p.stages)
          setIsQueued(p.state === 'queued')
          if (p.stage >= 0) setActiveStep(p.stage)
          if (p.state === 'queued' || p.state === 'running') started = true
          if (p.state === 'completed') {
            clearInterval(interval)
            resolve()
          } else if (p.state === 'failed') {
            clearInterval(interval)
            reject(new Error(p.error || 'Processing failed.'))
          } else if (p.state === 'pending' && started) {
            // Progress is kept in server memory, so a restart forgets the run.
            clearInterval(interval)
            reject(new Error('Processing was interrupted because the server restarted. Open the meeting from the Meetings page and process it again.'))
          }
        } catch {
          // Network blips or a server restart in progress: keep polling.
        }
      }, PROGRESS_POLL_MS)
    })
  }

  // ─── Submit ────────────────────────────────────────────────────────────────

  async function handleSubmit() {
    if (!file) {
      toast.error('Please select an audio or video file first.')
      return
    }

    setIsUploading(true)
    setError(null)
    setResult(null)
    setActiveStep(-1)
    setSteps(PROCESSING_STEPS)
    setIsQueued(false)

    const formData = new FormData()
    formData.append('file', file)
    if (title.trim()) formData.append('title', title.trim())
    formData.append('source_type', sourceType)
    // Upload first, then process separately so the page can poll the meeting's progress.
    formData.append('process_immediately', 'false')

    try {
      const uploaded = await meetingsApi.upload(formData)
      const meetingId: string = uploaded.meeting_id

      if (!processImmediately) {
        setResult({ id: meetingId, title: title || file.name })
        toast.success('Meeting uploaded.')
        return
      }

      setActiveStep(0)
      const speakers = numSpeakers.trim() ? Number(numSpeakers) : undefined
      await meetingsApi.process(meetingId, speakers)
      await waitForProcessing(meetingId)
      setActiveStep(steps.length)

      const meeting = await meetingsApi.get(meetingId).catch(() => null)
      setResult({ id: meetingId, title: meeting?.title || title || file.name })
      toast.success('Meeting processed successfully!')
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        (err instanceof Error ? err.message : 'Upload failed. Please try again.')
      setError(detail)
      setActiveStep(-1)
      toast.error(detail)
    } finally {
      setIsUploading(false)
    }
  }

  // ─── Render ────────────────────────────────────────────────────────────────

  return (
    <Page
      title="Upload Meeting"
      subtitle="Process audio or video recordings with AI"
    >
      <div className="max-w-2xl mx-auto space-y-6">

        {/* ── Dropzone ── */}
        <div
          {...getRootProps()}
          className={[
            'relative flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed px-8 py-14 text-center transition-colors cursor-pointer select-none',
            isDragReject
              ? 'border-red-500 bg-red-900/10'
              : isDragActive
              ? 'border-brand-400 bg-brand-400/10'
              : file
              ? 'border-emerald-500 bg-emerald-900/10'
              : 'border-slate-600 bg-slate-800/40 hover:border-slate-500 hover:bg-slate-800/60',
            isUploading ? 'pointer-events-none opacity-60' : '',
          ].join(' ')}
        >
          <input {...getInputProps()} />

          {/* Icon */}
          <div
            className={[
              'flex h-16 w-16 items-center justify-center rounded-full transition-colors',
              file
                ? 'bg-emerald-900/40 text-emerald-400'
                : isDragActive
                ? 'bg-brand-900/40 text-brand-400'
                : 'bg-slate-800 text-slate-400',
            ].join(' ')}
          >
            {file ? (
              isVideoFile(file) ? <FileVideo className="h-8 w-8" /> : <FileAudio className="h-8 w-8" />
            ) : (
              <Upload className="h-8 w-8" />
            )}
          </div>

          {/* Text */}
          {file ? (
            <div className="space-y-1">
              <p className="font-semibold text-emerald-300">{file.name}</p>
              <p className="text-sm text-slate-400">{formatFileSize(file.size)}</p>
              <p className="text-xs text-slate-500">Click or drag to replace</p>
            </div>
          ) : isDragReject ? (
            <div className="space-y-1">
              <p className="font-semibold text-red-400">Unsupported file type</p>
              <p className="text-sm text-slate-400">Please use .mp3, .wav, .m4a, .mp4 or .webm</p>
            </div>
          ) : (
            <div className="space-y-1">
              <p className="font-semibold text-slate-200">
                {isDragActive ? 'Drop your file here' : 'Drag & drop your recording'}
              </p>
              <p className="text-sm text-slate-400">or click to browse files</p>
              <p className="text-xs text-slate-500 mt-1">Supports .mp3 · .wav · .m4a · .mp4 · .webm</p>
            </div>
          )}
        </div>

        {/* ── Form Fields ── */}
        <div className="card p-6 space-y-6">

          {/* Meeting title */}
          <div>
            <label htmlFor="meeting-title" className="label">
              Meeting Title <span className="text-slate-500 font-normal">(optional)</span>
            </label>
            <input
              id="meeting-title"
              type="text"
              className="input w-full mt-1"
              placeholder="e.g. Q3 Sprint Planning"
              value={title}
              onChange={e => setTitle(e.target.value)}
              disabled={isUploading}
            />
          </div>

          {/* Speaker count: the strongest hint for accurate diarization */}
          <div>
            <label htmlFor="num-speakers" className="label">
              Number of Speakers <span className="text-slate-500 font-normal">(optional, improves accuracy)</span>
            </label>
            <input
              id="num-speakers"
              type="number"
              min={1}
              max={20}
              className="input w-full mt-1"
              placeholder="Leave empty to detect automatically"
              value={numSpeakers}
              onChange={e => setNumSpeakers(e.target.value)}
              disabled={isUploading}
            />
          </div>

          {/* Source type */}
          <div>
            <p className="label mb-2">Source Type</p>
            <div className="flex gap-6">
              {(['uploaded', 'live'] as const).map(type => (
                <label
                  key={type}
                  className="flex items-center gap-2.5 cursor-pointer group"
                >
                  <input
                    type="radio"
                    name="source-type"
                    value={type}
                    checked={sourceType === type}
                    onChange={() => setSourceType(type)}
                    disabled={isUploading}
                    className="accent-brand-400 w-4 h-4"
                  />
                  <span className="capitalize text-sm text-slate-300 group-hover:text-slate-100 transition-colors">
                    {type === 'uploaded' ? 'Uploaded Recording' : 'Live Recording'}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Process immediately */}
          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              checked={processImmediately}
              onChange={e => setProcessImmediately(e.target.checked)}
              disabled={isUploading}
              className="accent-brand-400 w-4 h-4 rounded"
            />
            <div>
              <span className="text-sm font-medium text-slate-200 group-hover:text-slate-100 transition-colors">
                Process Immediately
              </span>
              <p className="text-xs text-slate-500 mt-0.5">
                Run the full AI pipeline right after upload (transcription, diarization, MOM generation, etc.)
              </p>
            </div>
          </label>

          {/* Info banner when process immediately is off */}
          {!processImmediately && (
            <div className="flex items-start gap-3 rounded-lg bg-blue-900/20 border border-blue-800/50 px-4 py-3 text-blue-300 text-sm">
              <Info className="h-4 w-4 shrink-0 mt-0.5" />
              <span>
                The file will be uploaded but not processed. You can trigger processing later from the meeting detail page.
              </span>
            </div>
          )}
        </div>

        {/* ── Processing Progress ── */}
        {isUploading && (
          <div className="card p-6 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Spinner className="w-4 h-4" />
              <p className="text-sm font-semibold text-slate-200">
                {isQueued ? 'Waiting for another meeting to finish processing…' : 'Processing your meeting…'}
              </p>
            </div>
            <ol className="space-y-2.5">
              {steps.map((step, idx) => {
                const isDone = idx < activeStep
                const isActive = activeStep === idx

                return (
                  <li key={step} className="flex items-center gap-3 text-sm">
                    <span className="shrink-0 w-5 h-5 flex items-center justify-center">
                      {isDone ? (
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                      ) : isActive ? (
                        <Spinner className="w-4 h-4" />
                      ) : (
                        <span className="w-2 h-2 rounded-full bg-slate-600 mx-auto block" />
                      )}
                    </span>
                    <span
                      className={[
                        'transition-colors',
                        isDone
                          ? 'text-emerald-400 line-through decoration-emerald-700'
                          : isActive
                          ? 'text-slate-100 font-medium'
                          : 'text-slate-500',
                      ].join(' ')}
                    >
                      {step}
                    </span>
                  </li>
                )
              })}
            </ol>
          </div>
        )}

        {/* ── Success Result ── */}
        {result && !isUploading && (
          <div className="card p-6 flex items-start gap-4 border border-emerald-700/60 bg-emerald-900/10">
            <CheckCircle className="h-6 w-6 text-emerald-400 shrink-0 mt-0.5" />
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-emerald-300">Meeting processed successfully!</p>
              <p className="text-sm text-slate-400 mt-0.5 truncate">
                {result.title}
              </p>
            </div>
            <button
              className="btn-primary shrink-0"
              onClick={() => navigate(`/meetings/${result.id}`)}
            >
              View Meeting
            </button>
          </div>
        )}

        {/* ── Error ── */}
        {error && !isUploading && (
          <div className="card p-6 flex items-start gap-4 border border-red-700/60 bg-red-900/10">
            <XCircle className="h-6 w-6 text-red-400 shrink-0 mt-0.5" />
            <div className="min-w-0">
              <p className="font-semibold text-red-300">Upload failed</p>
              <p className="text-sm text-red-400/80 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* ── Submit ── */}
        {!result && (
          <div className="flex justify-end">
            <button
              className="btn-primary flex items-center gap-2 px-8 py-2.5 disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleSubmit}
              disabled={isUploading || !file}
            >
              {isUploading ? (
                <>
                  <Spinner className="w-4 h-4" />
                  Processing…
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4" />
                  Upload & Process
                </>
              )}
            </button>
          </div>
        )}

      </div>
    </Page>
  )
}

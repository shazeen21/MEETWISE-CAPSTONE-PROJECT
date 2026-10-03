import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import { meetingsApi } from '../api'
import { Page, Spinner } from '../components/UI'
import { Upload, FileAudio, FileVideo, CheckCircle, XCircle, Info } from 'lucide-react'
import toast from 'react-hot-toast'
import { formatDuration } from '../utils'

// ─── Processing Steps ─────────────────────────────────────────────────────────

const PROCESSING_STEPS = [
  'Audio Cleaning & Noise Reduction',
  'Voice Activity Detection',
  'Speaker Diarization',
  'Speaker Identification',
  'Speech-to-Text (WhisperX)',
  'Emotion & Accent Detection',
  'Topic & Keyword Extraction',
  'Action Item Detection',
  'MOM Generation',
  'ChromaDB Indexing',
]

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
  const [sourceType, setSourceType] = useState<'uploaded' | 'live'>('uploaded')
  const [processImmediately, setProcessImmediately] = useState(true)

  // Upload state
  const [isUploading, setIsUploading] = useState(false)
  const [activeStep, setActiveStep] = useState<number>(-1)
  const [completedSteps, setCompletedSteps] = useState<number[]>([])
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

  // ─── Processing Animation ─────────────────────────────────────────────────

  function runProcessingAnimation(): Promise<void> {
    return new Promise((resolve) => {
      let step = 0
      setActiveStep(0)

      const interval = setInterval(() => {
        setCompletedSteps(prev => [...prev, step])
        step += 1

        if (step >= PROCESSING_STEPS.length) {
          clearInterval(interval)
          setActiveStep(-1)
          resolve()
        } else {
          setActiveStep(step)
        }
      }, 2500)
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
    setCompletedSteps([])
    setActiveStep(-1)

    const formData = new FormData()
    formData.append('file', file)
    if (title.trim()) formData.append('title', title.trim())
    formData.append('source_type', sourceType)
    formData.append('process_immediately', String(processImmediately))

    try {
      // Start animation concurrently with the upload
      const animationPromise = runProcessingAnimation()

      const data = await meetingsApi.upload(formData)

      // Wait for animation to finish so UX feels smooth
      await animationPromise

      setResult({ id: data.id ?? data.meeting_id, title: (data.title ?? title) || file.name })
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
              <p className="text-sm font-semibold text-slate-200">Processing your meeting…</p>
            </div>
            <ol className="space-y-2.5">
              {PROCESSING_STEPS.map((step, idx) => {
                const isDone = completedSteps.includes(idx)
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

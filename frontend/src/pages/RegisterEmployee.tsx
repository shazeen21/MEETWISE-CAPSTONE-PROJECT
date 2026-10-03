import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from 'react-query'
import { employeesApi, Employee } from '../api'
import { Page, Spinner, ErrorBox } from '../components/UI'
import {
  User,
  Building,
  Briefcase,
  Mail,
  Mic,
  Upload,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  ArrowLeft,
  Mic2,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { useDropzone } from 'react-dropzone'

// ─── Constants ────────────────────────────────────────────────────────────────

const DEPARTMENTS = [
  'Engineering',
  'Product',
  'Design',
  'Marketing',
  'Sales',
  'HR',
  'Finance',
  'Operations',
  'Executive',
  'Legal',
]

const ACCENT_HINTS = [
  'American',
  'British',
  'Indian',
  'Australian',
  'International English',
]

const VOICE_PROMPTS = [
  'The quick brown fox jumps over the lazy dog.',
  'She sells seashells by the seashore every single day.',
  'How much wood would a woodchuck chuck if a woodchuck could chuck wood?',
  'In our team meeting today, we discussed the quarterly roadmap and key deliverables.',
  'The project deadline is next Friday and all action items must be completed before then.',
]

const ACCEPTED_AUDIO = {
  'audio/wav': ['.wav'],
  'audio/mpeg': ['.mp3'],
  'audio/webm': ['.webm'],
  'audio/mp4': ['.m4a'],
  'audio/x-m4a': ['.m4a'],
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface FormData {
  employee_id: string
  name: string
  email: string
  department: string
  team: string
  designation: string
  accent_hint: string
}

interface UploadedSample {
  fileName: string
  promptText: string
}

// ─── Step Indicator ───────────────────────────────────────────────────────────

function StepIndicator({ currentStep }: { currentStep: number }) {
  const steps = [
    { label: 'Profile Info', icon: User },
    { label: 'Create Account', icon: Briefcase },
    { label: 'Voice Enrollment', icon: Mic2 },
  ]

  return (
    <div className="flex items-center justify-center mb-10">
      {steps.map((step, idx) => {
        const stepNum = idx + 1
        const isComplete = currentStep > stepNum
        const isActive = currentStep === stepNum
        const Icon = step.icon

        return (
          <div key={stepNum} className="flex items-center">
            {/* Step circle */}
            <div className="flex flex-col items-center gap-1.5">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                  isComplete
                    ? 'bg-emerald-500 border-emerald-500 text-white'
                    : isActive
                    ? 'bg-brand-500 border-brand-500 text-white'
                    : 'bg-slate-800 border-slate-700 text-slate-500'
                }`}
              >
                {isComplete ? (
                  <CheckCircle className="w-5 h-5" />
                ) : (
                  <Icon className="w-4 h-4" />
                )}
              </div>
              <span
                className={`text-xs font-medium whitespace-nowrap ${
                  isActive
                    ? 'text-brand-400'
                    : isComplete
                    ? 'text-emerald-400'
                    : 'text-slate-500'
                }`}
              >
                {step.label}
              </span>
            </div>

            {/* Connector line */}
            {idx < steps.length - 1 && (
              <div
                className={`h-0.5 w-20 mx-2 mb-5 rounded transition-all duration-500 ${
                  currentStep > stepNum ? 'bg-emerald-500' : 'bg-slate-700'
                }`}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

// ─── Step 1: Profile Information ──────────────────────────────────────────────

function StepProfile({
  formData,
  onChange,
  onNext,
}: {
  formData: FormData
  onChange: (key: keyof FormData, value: string) => void
  onNext: () => void
}) {
  const [errors, setErrors] = useState<Partial<Record<keyof FormData, string>>>({})

  const validate = () => {
    const newErrors: Partial<Record<keyof FormData, string>> = {}
    if (!formData.employee_id.trim()) newErrors.employee_id = 'Employee ID is required'
    if (!formData.name.trim()) newErrors.name = 'Full name is required'
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required'
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Enter a valid email address'
    }
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleNext = () => {
    if (validate()) onNext()
  }

  const field = (
    key: keyof FormData,
    label: string,
    icon: React.ReactNode,
    inputProps: React.InputHTMLAttributes<HTMLInputElement> = {}
  ) => (
    <div className="space-y-1.5">
      <label className="label flex items-center gap-1.5">
        <span className="text-slate-400 w-4 h-4">{icon}</span>
        {label}
      </label>
      <input
        className={`input ${errors[key] ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''}`}
        value={formData[key]}
        onChange={e => onChange(key, e.target.value)}
        {...inputProps}
      />
      {errors[key] && (
        <p className="text-xs text-red-400 flex items-center gap-1">
          <AlertCircle className="w-3 h-3" />
          {errors[key]}
        </p>
      )}
    </div>
  )

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {field('employee_id', 'Employee ID', <User className="w-4 h-4" />, {
          placeholder: 'e.g. EMP-0042',
        })}
        {field('name', 'Full Name', <User className="w-4 h-4" />, {
          placeholder: 'e.g. Jane Doe',
        })}
      </div>

      {field('email', 'Corporate Email', <Mail className="w-4 h-4" />, {
        type: 'email',
        placeholder: 'jane.doe@company.com',
      })}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {/* Department */}
        <div className="space-y-1.5">
          <label className="label flex items-center gap-1.5">
            <Building className="w-4 h-4 text-slate-400" />
            Department
          </label>
          <select
            className="input"
            value={formData.department}
            onChange={e => onChange('department', e.target.value)}
          >
            <option value="">Select department…</option>
            {DEPARTMENTS.map(d => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        {field('team', 'Team', <Building className="w-4 h-4" />, {
          placeholder: 'e.g. Platform',
        })}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {field('designation', 'Designation', <Briefcase className="w-4 h-4" />, {
          placeholder: 'e.g. Senior Software Engineer',
        })}

        {/* Accent Hint */}
        <div className="space-y-1.5">
          <label className="label flex items-center gap-1.5">
            <Mic className="w-4 h-4 text-slate-400" />
            Accent Hint{' '}
            <span className="text-slate-600 font-normal">(optional)</span>
          </label>
          <select
            className="input"
            value={formData.accent_hint}
            onChange={e => onChange('accent_hint', e.target.value)}
          >
            <option value="">Select accent…</option>
            {ACCENT_HINTS.map(a => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex justify-end pt-2">
        <button className="btn-primary flex items-center gap-2" onClick={handleNext}>
          Next
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

// ─── Step 2: Create Account ───────────────────────────────────────────────────

function StepCreateAccount({
  formData,
  onBack,
  onSuccess,
}: {
  formData: FormData
  onBack: () => void
  onSuccess: (emp: Employee) => void
}) {
  const registerMutation = useMutation(
    () =>
      employeesApi.register({
        employee_id: formData.employee_id,
        name: formData.name,
        email: formData.email,
        department: formData.department,
        team: formData.team,
        designation: formData.designation,
        accent_hint: formData.accent_hint || undefined,
      }),
    {
      onSuccess: emp => {
        toast.success(`Employee profile created for ${emp.name}`)
        onSuccess(emp)
      },
      onError: (err: any) => {
        const msg =
          err?.response?.data?.detail || err?.message || 'Registration failed'
        toast.error(msg)
      },
    }
  )

  const summaryRows = [
    { label: 'Employee ID', value: formData.employee_id, icon: <User className="w-4 h-4" /> },
    { label: 'Full Name', value: formData.name, icon: <User className="w-4 h-4" /> },
    { label: 'Email', value: formData.email, icon: <Mail className="w-4 h-4" /> },
    { label: 'Department', value: formData.department || '—', icon: <Building className="w-4 h-4" /> },
    { label: 'Team', value: formData.team || '—', icon: <Building className="w-4 h-4" /> },
    { label: 'Designation', value: formData.designation || '—', icon: <Briefcase className="w-4 h-4" /> },
    { label: 'Accent Hint', value: formData.accent_hint || '—', icon: <Mic className="w-4 h-4" /> },
  ]

  return (
    <div className="space-y-6">
      {/* Summary card */}
      <div className="card p-5 space-y-3">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
          Profile Summary
        </h3>
        <div className="divide-y divide-slate-800">
          {summaryRows.map(row => (
            <div
              key={row.label}
              className="flex items-center justify-between py-2.5 gap-4"
            >
              <span className="flex items-center gap-2 text-slate-400 text-sm">
                <span className="text-slate-500">{row.icon}</span>
                {row.label}
              </span>
              <span className="text-slate-200 text-sm font-medium text-right break-all">
                {row.value}
              </span>
            </div>
          ))}
        </div>
      </div>

      {registerMutation.isError && (
        <ErrorBox
          message={
            (registerMutation.error as any)?.response?.data?.detail ||
            (registerMutation.error as any)?.message ||
            'Registration failed. Please try again.'
          }
        />
      )}

      <div className="flex items-center justify-between pt-2">
        <button
          className="btn-secondary flex items-center gap-2"
          onClick={onBack}
          disabled={registerMutation.isLoading}
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button
          className="btn-primary flex items-center gap-2"
          onClick={() => registerMutation.mutate()}
          disabled={registerMutation.isLoading}
        >
          {registerMutation.isLoading && <Spinner className="w-4 h-4" />}
          Create Employee Profile
          {!registerMutation.isLoading && <ArrowRight className="w-4 h-4" />}
        </button>
      </div>
    </div>
  )
}

// ─── Voice Sample Dropzone ────────────────────────────────────────────────────

function VoiceDropzone({
  employeeId,
  promptText,
  onUploaded,
}: {
  employeeId: string
  promptText: string
  onUploaded: (sample: UploadedSample) => void
}) {
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length > 0) {
      setPendingFile(accepted[0])
      setUploadError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_AUDIO,
    maxFiles: 1,
    multiple: false,
  })

  const handleUpload = async () => {
    if (!pendingFile) return
    setUploading(true)
    setUploadError(null)
    try {
      await employeesApi.uploadVoiceSample(employeeId, pendingFile, promptText)
      onUploaded({ fileName: pendingFile.name, promptText })
      setPendingFile(null)
      toast.success('Voice sample uploaded successfully')
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail || err?.message || 'Upload failed'
      setUploadError(msg)
      toast.error(msg)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 ${
          isDragActive
            ? 'border-brand-400 bg-brand-500/10'
            : 'border-slate-700 hover:border-slate-600 bg-slate-800/40'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="w-7 h-7 mx-auto mb-2 text-slate-400" />
        {isDragActive ? (
          <p className="text-brand-300 text-sm font-medium">Drop the file here…</p>
        ) : (
          <>
            <p className="text-slate-300 text-sm font-medium">
              Drag & drop a voice file, or click to browse
            </p>
            <p className="text-slate-500 text-xs mt-1">.wav, .mp3, .webm, .m4a</p>
          </>
        )}
      </div>

      {pendingFile && (
        <div className="flex items-center justify-between bg-slate-800 rounded-lg px-4 py-3 gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <Mic className="w-4 h-4 text-brand-400 shrink-0" />
            <span className="text-slate-300 text-sm truncate">{pendingFile.name}</span>
            <span className="text-slate-500 text-xs shrink-0">
              ({(pendingFile.size / 1024).toFixed(0)} KB)
            </span>
          </div>
          <button
            className="btn-primary flex items-center gap-1.5 text-sm py-1.5 px-3 shrink-0"
            onClick={handleUpload}
            disabled={uploading}
          >
            {uploading ? (
              <Spinner className="w-3.5 h-3.5" />
            ) : (
              <Upload className="w-3.5 h-3.5" />
            )}
            {uploading ? 'Uploading…' : 'Upload'}
          </button>
        </div>
      )}

      {uploadError && <ErrorBox message={uploadError} />}
    </div>
  )
}

// ─── Step 3: Voice Enrollment ─────────────────────────────────────────────────

function StepVoiceEnrollment({
  employee,
  onFinish,
}: {
  employee: Employee
  onFinish: () => void
}) {
  const [uploadedSamples, setUploadedSamples] = useState<UploadedSample[]>([])
  const [activePromptIdx, setActivePromptIdx] = useState(0)

  const handleUploaded = (sample: UploadedSample) => {
    setUploadedSamples(prev => [...prev, sample])
    // Advance to next prompt
    setActivePromptIdx(prev => Math.min(prev + 1, VOICE_PROMPTS.length - 1))
  }

  const totalRequired = 3
  const uploaded = uploadedSamples.length
  const canFinish = uploaded >= 1

  return (
    <div className="space-y-6">
      {/* Employee confirmed */}
      <div className="card p-4 flex items-center gap-3 border-emerald-800/40 bg-emerald-900/10">
        <CheckCircle className="w-5 h-5 text-emerald-400 shrink-0" />
        <div className="min-w-0">
          <p className="text-emerald-300 text-sm font-medium">
            Profile created — {employee.name}
          </p>
          <p className="text-slate-400 text-xs mt-0.5">ID: {employee.employee_id}</p>
        </div>
      </div>

      {/* Instructions */}
      <div className="space-y-2">
        <h3 className="text-slate-200 font-semibold flex items-center gap-2">
          <Mic2 className="w-5 h-5 text-brand-400" />
          Voice Enrollment
        </h3>
        <p className="muted text-sm">
          Record or upload <strong className="text-slate-300">3+ voice samples</strong> for
          accurate speaker identification. Read the prompt below aloud and upload the audio
          file.
        </p>
      </div>

      {/* Progress */}
      <div className="flex items-center gap-3">
        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-brand-500 rounded-full transition-all duration-500"
            style={{ width: `${Math.min((uploaded / totalRequired) * 100, 100)}%` }}
          />
        </div>
        <span
          className={`text-sm font-semibold ${
            uploaded >= totalRequired ? 'text-emerald-400' : 'text-slate-400'
          }`}
        >
          {uploaded}/{totalRequired} samples uploaded
        </span>
      </div>

      {/* Uploaded list */}
      {uploadedSamples.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Uploaded
          </p>
          <ul className="space-y-1.5">
            {uploadedSamples.map((s, i) => (
              <li
                key={i}
                className="flex items-start gap-2 bg-emerald-900/10 border border-emerald-800/40 rounded-lg px-3 py-2"
              >
                <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
                <div className="min-w-0">
                  <p className="text-emerald-300 text-sm font-medium truncate">
                    {s.fileName}
                  </p>
                  <p className="text-slate-500 text-xs truncate">{s.promptText}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Prompt + dropzone */}
      <div className="card p-5 space-y-4">
        <div className="space-y-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Prompt {activePromptIdx + 1} of {VOICE_PROMPTS.length}
          </p>
          <p className="text-slate-100 text-base leading-relaxed font-medium bg-slate-800/60 rounded-lg px-4 py-3 border border-slate-700">
            "{VOICE_PROMPTS[activePromptIdx]}"
          </p>
          <div className="flex flex-wrap gap-2">
            {VOICE_PROMPTS.map((_, i) => (
              <button
                key={i}
                onClick={() => setActivePromptIdx(i)}
                className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                  i === activePromptIdx
                    ? 'bg-brand-500 border-brand-400 text-white'
                    : 'border-slate-700 text-slate-400 hover:border-slate-500'
                }`}
              >
                {i + 1}
              </button>
            ))}
          </div>
        </div>

        <VoiceDropzone
          employeeId={employee.id}
          promptText={VOICE_PROMPTS[activePromptIdx]}
          onUploaded={handleUploaded}
        />
      </div>

      {/* Finish */}
      <div className="flex items-center justify-between pt-2">
        {!canFinish && (
          <p className="text-xs text-slate-500 flex items-center gap-1">
            <AlertCircle className="w-3.5 h-3.5" />
            Upload at least 1 sample to finish
          </p>
        )}
        <div className="ml-auto">
          <button
            className="btn-primary flex items-center gap-2"
            onClick={onFinish}
            disabled={!canFinish}
          >
            Finish Setup
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function RegisterEmployee() {
  const navigate = useNavigate()
  const [step, setStep] = useState(1)
  const [createdEmployee, setCreatedEmployee] = useState<Employee | null>(null)

  const [formData, setFormData] = useState<FormData>({
    employee_id: '',
    name: '',
    email: '',
    department: '',
    team: '',
    designation: '',
    accent_hint: '',
  })

  const handleChange = (key: keyof FormData, value: string) => {
    setFormData(prev => ({ ...prev, [key]: value }))
  }

  const handleRegistered = (emp: Employee) => {
    setCreatedEmployee(emp)
    setStep(3)
  }

  const handleFinish = () => {
    if (createdEmployee) {
      navigate(`/employees/${createdEmployee.id}`)
    }
  }

  return (
    <Page
      title="Register Employee"
      subtitle="Set up a new employee profile and enroll their voice for speaker identification"
    >
      <div className="max-w-2xl mx-auto">
        <StepIndicator currentStep={step} />

        <div className="card p-6">
          {step === 1 && (
            <StepProfile
              formData={formData}
              onChange={handleChange}
              onNext={() => setStep(2)}
            />
          )}

          {step === 2 && (
            <StepCreateAccount
              formData={formData}
              onBack={() => setStep(1)}
              onSuccess={handleRegistered}
            />
          )}

          {step === 3 && createdEmployee && (
            <StepVoiceEnrollment
              employee={createdEmployee}
              onFinish={handleFinish}
            />
          )}
        </div>
      </div>
    </Page>
  )
}

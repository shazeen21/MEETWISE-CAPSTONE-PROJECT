import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'X-User-Id': 'emp-admin',
    'X-User-Role': 'admin',
    'X-User-Email': 'admin@company.internal',
  },
})

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Meeting {
  id: string
  title: string
  audio_file_name: string
  duration_seconds: number
  media_type: string
  source_type: string
  summary: string | null
  meeting_date: string
  speakers: Speaker[]
  action_items: ActionItem[]
  decisions: Decision[]
  topics: Topic[]
  keywords: Keyword[]
}

export interface Speaker {
  id: string
  speaker_label: string
  speaker_name: string | null
  confidence_score: number
  total_speaking_time: number
  detected_accent: string | null
  dominant_emotion: string | null
}

export interface ActionItem {
  id: string
  task: string
  owner: string
  deadline: string
  priority: 'High' | 'Medium' | 'Low'
  status: 'pending' | 'in_progress' | 'completed'
  created_at?: string
}

export interface Decision {
  id?: string
  decision: string
  timestamp: string | null
}

export interface Topic {
  id: string
  topic: string
  start_time: number
  end_time: number
  duration_seconds: number
  participants: string[]
}

export interface Keyword {
  id: string
  keyword: string
  category: string
  relevance_score: number
}

export interface TranscriptSegment {
  id: string
  speaker: string
  start_time: number
  end_time: number
  text: string
  emotion: string
  speaking_style: string
  segment_index: number
}

export interface Employee {
  id: string
  employee_id: string
  name: string
  email: string
  department: string
  team: string
  designation: string
  voice_samples_count: number
  has_enrolled_voice: boolean
  created_at: string
}

export interface EmployeeReport {
  id: string
  employee_id: string | null
  employee_name: string
  topics_discussed: string[]
  decisions_affecting: string[]
  assigned_tasks: string[]
  mentioned_deadlines: string[]
  follow_ups: string[]
  created_at: string
}

export interface RAGSearchResult {
  answer: string
  sources: { text: string; meeting_id: string; start_time?: number }[]
  model: string
  query: string
}

export interface Analytics {
  meeting_id: string
  title: string
  duration_seconds: number
  overall_sentiment: {
    overall_sentiment: string
    positive_percentage: number
    neutral_percentage: number
    negative_percentage: number
    emotions_breakdown: Record<string, number>
    key_emotional_moments: string[]
  } | null
  speakers: {
    name: string
    speaking_time: number
    confidence: number
    accent: string
    dominant_emotion: string
  }[]
  topics_count: number
  keywords_count: number
  decisions_count: number
  action_items_count: number
}

export interface Notification {
  id: string
  meeting_id: string | null
  recipient: string
  channel: string
  status: string
  subject: string | null
  payload: string | null
  created_at: string
}

export interface AuditLog {
  id: string
  action: string
  user_id: string
  details: string | null
  ip_address: string | null
  created_at: string
}

export interface ConfiguredKeyword {
  id: string
  keyword: string
  category: string
  is_active: boolean
  created_at: string
}

export interface ManagerSummary {
  executive_summary: string
  key_decisions: string[]
  risks: string[]
  delays: string[]
  team_blockers: string[]
  assigned_tasks: string[]
  follow_up_actions: string[]
}

// ─── Meetings ─────────────────────────────────────────────────────────────────

export const meetingsApi = {
  list: (limit = 50, offset = 0) =>
    api.get<Meeting[]>(`/meetings?limit=${limit}&offset=${offset}`).then(r => r.data),

  get: (id: string) =>
    api.get<Meeting>(`/meetings/${id}`).then(r => r.data),

  upload: (formData: FormData) =>
    api.post('/meetings/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data),

  process: (id: string) =>
    api.post(`/meetings/${id}/process`).then(r => r.data),

  delete: (id: string) =>
    api.delete(`/meetings/${id}`).then(r => r.data),

  getTranscript: (id: string) =>
    api.get<{ meeting_id: string; title: string; segments: TranscriptSegment[] }>(`/meetings/${id}/transcript`).then(r => r.data),

  downloadMomPdf: (id: string) =>
    `${api.defaults.baseURL}/meetings/${id}/mom/pdf`,

  getManagerSummary: (id: string) =>
    api.get<ManagerSummary>(`/meetings/${id}/manager-summary`).then(r => r.data),

  getEmployeeReports: (id: string) =>
    api.get<EmployeeReport[]>(`/meetings/${id}/employee-reports`).then(r => r.data),

  getAnalytics: (id: string) =>
    api.get<Analytics>(`/meetings/${id}/analytics`).then(r => r.data),
}

// ─── Employees ────────────────────────────────────────────────────────────────

export const employeesApi = {
  list: (department?: string, limit = 100) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (department) params.append('department', department)
    return api.get<Employee[]>(`/employees?${params}`).then(r => r.data)
  },

  get: (id: string) =>
    api.get<Employee>(`/employees/${id}`).then(r => r.data),

  register: (data: {
    employee_id: string
    name: string
    email: string
    department: string
    team: string
    designation: string
    accent_hint?: string
  }) => api.post<Employee>('/employees/register', data).then(r => r.data),

  uploadVoiceSample: (idOrEmpId: string, file: File, promptText?: string) => {
    const fd = new FormData()
    fd.append('file', file)
    if (promptText) fd.append('prompt_text', promptText)
    return api.post(`/employees/${idOrEmpId}/voice-samples`, fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data)
  },
}

// ─── Search ───────────────────────────────────────────────────────────────────

export const searchApi = {
  query: (query: string, topK = 5, meetingId?: string) =>
    api.post<RAGSearchResult>('/search', { query, top_k: topK, meeting_id: meetingId }).then(r => r.data),
}

// ─── Admin ────────────────────────────────────────────────────────────────────

export const adminApi = {
  getKeywords: () =>
    api.get<ConfiguredKeyword[]>('/admin/keywords').then(r => r.data),

  addKeyword: (keyword: string, category = 'custom') =>
    api.post<ConfiguredKeyword>('/admin/keywords', { keyword, category }).then(r => r.data),

  deleteKeyword: (id: string) =>
    api.delete(`/admin/keywords/${id}`).then(r => r.data),

  getAuditLogs: (limit = 100) =>
    api.get<AuditLog[]>(`/admin/audit-logs?limit=${limit}`).then(r => r.data),

  getNotifications: (recipient?: string, limit = 50) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (recipient) params.append('recipient', recipient)
    return api.get<Notification[]>(`/notifications?${params}`).then(r => r.data)
  },

  healthCheck: () =>
    api.get('/health').then(r => r.data),
}

export default api

import { format, formatDistanceToNow, parseISO } from 'date-fns'

export function formatDuration(seconds: number): string {
  if (!seconds) return '0:00'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${m}:${String(s).padStart(2, '0')}`
}

export function formatTimeAgo(dateStr: string): string {
  try {
    return formatDistanceToNow(parseISO(dateStr), { addSuffix: true })
  } catch {
    return dateStr
  }
}

export function formatDate(dateStr: string): string {
  try {
    return format(parseISO(dateStr), 'MMM d, yyyy')
  } catch {
    return dateStr
  }
}

export function formatTimestamp(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function priorityColor(priority: string): string {
  switch (priority?.toLowerCase()) {
    case 'high': return 'badge-red'
    case 'medium': return 'badge-yellow'
    case 'low': return 'badge-green'
    default: return 'badge-slate'
  }
}

export function statusColor(status: string): string {
  switch (status?.toLowerCase()) {
    case 'completed': return 'badge-green'
    case 'in_progress': return 'badge-blue'
    case 'pending': return 'badge-yellow'
    default: return 'badge-slate'
  }
}

export function emotionColor(emotion: string): string {
  // Keep in sync with EMOTION_TAXONOMY in backend/app/meeting_intelligence/schemas.py
  const map: Record<string, string> = {
    joy: 'text-yellow-400',
    excitement: 'text-orange-400',
    satisfaction: 'text-emerald-400',
    neutral: 'text-slate-400',
    confusion: 'text-sky-400',
    anxiety: 'text-violet-400',
    frustration: 'text-red-400',
    anger: 'text-red-500',
    sadness: 'text-indigo-400',
    disappointment: 'text-amber-400',
  }
  return map[emotion?.toLowerCase()] ?? 'text-slate-400'
}

export function accentBadge(accent: string): string {
  if (!accent) return 'badge-slate'
  const map: Record<string, string> = {
    indian: 'badge-yellow',
    british: 'badge-purple',
    american: 'badge-blue',
    australian: 'badge-green',
  }
  return map[accent.toLowerCase()] ?? 'badge-slate'
}

export function categoryColor(category: string): string {
  const map: Record<string, string> = {
    product: 'badge-blue',
    client: 'badge-purple',
    risk: 'badge-red',
    budget: 'badge-yellow',
    hiring: 'badge-green',
    revenue: 'badge-green',
    deadline: 'badge-red',
    deliverable: 'badge-blue',
    custom: 'badge-slate',
    general: 'badge-slate',
  }
  return map[category?.toLowerCase()] ?? 'badge-slate'
}

export function clsx(...classes: (string | undefined | false | null)[]): string {
  return classes.filter(Boolean).join(' ')
}

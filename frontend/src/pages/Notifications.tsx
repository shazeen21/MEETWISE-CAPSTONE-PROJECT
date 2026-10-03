import { useQuery } from 'react-query'
import { Bell, Mail, MessageSquare, CheckCircle } from 'lucide-react'
import { adminApi, Notification } from '../api'
import { Page, LoadingPage, EmptyState, ErrorBox } from '../components/UI'

export default function NotificationsPage() {
  const { data, isLoading, isError } = useQuery<Notification[]>(['notifications'], () => adminApi.getNotifications())
  if (isLoading) return <LoadingPage />
  return (
    <Page title="Notifications" subtitle="Meeting summaries and follow-up delivery status">
      {isError ? <ErrorBox message="Unable to load notifications." /> : !data?.length ? <EmptyState icon={<Bell className="w-6 h-6" />} title="No notifications yet" description="Processed meeting summaries will appear here." /> : (
        <div className="space-y-3">{data.map(item => <article key={item.id} className="card p-4 flex gap-4"><div className="w-9 h-9 rounded-lg bg-slate-800 flex items-center justify-center text-brand-400">{item.channel === 'email' ? <Mail className="w-4 h-4" /> : <MessageSquare className="w-4 h-4" />}</div><div className="min-w-0 flex-1"><div className="flex justify-between gap-3"><h2 className="font-semibold text-slate-200 truncate">{item.subject || 'Meeting notification'}</h2><span className="text-xs text-emerald-400 flex items-center gap-1"><CheckCircle className="w-3 h-3" />{item.status}</span></div><p className="text-sm text-slate-400 mt-1">{item.recipient} via {item.channel}</p><p className="text-xs text-slate-500 mt-2">{new Date(item.created_at).toLocaleString()}</p></div></article>)}</div>
      )}
    </Page>
  )
}

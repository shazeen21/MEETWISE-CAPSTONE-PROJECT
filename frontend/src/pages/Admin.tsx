import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { Plus, Shield, Trash2 } from 'lucide-react'
import { adminApi, ConfiguredKeyword, AuditLog } from '../api'
import { Page, LoadingPage, ErrorBox, EmptyState } from '../components/UI'

export default function AdminPage() {
  const client = useQueryClient()
  const [keyword, setKeyword] = useState('')
  const [category, setCategory] = useState('custom')
  const keywords = useQuery<ConfiguredKeyword[]>(['admin-keywords'], adminApi.getKeywords)
  const logs = useQuery<AuditLog[]>(['audit-logs'], () => adminApi.getAuditLogs(20))
  const add = useMutation(() => adminApi.addKeyword(keyword.trim(), category), { onSuccess: () => { setKeyword(''); client.invalidateQueries(['admin-keywords']) } })
  const remove = useMutation((id: string) => adminApi.deleteKeyword(id), { onSuccess: () => client.invalidateQueries(['admin-keywords']) })
  if (keywords.isLoading || logs.isLoading) return <LoadingPage />
  return (
    <Page title="Administration" subtitle="Configure enterprise monitoring and review audit activity">
      {(keywords.isError || logs.isError) && <ErrorBox message="Some administration data could not be loaded." />}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="card p-5"><h2 className="section-title flex items-center gap-2 mb-4"><Shield className="w-4 h-4 text-brand-400" /> Monitored Keywords</h2><form className="flex gap-2 mb-4" onSubmit={e => { e.preventDefault(); if (keyword.trim()) add.mutate() }}><input className="input flex-1" value={keyword} onChange={e => setKeyword(e.target.value)} placeholder="Client ABC" /><input className="input w-28" value={category} onChange={e => setCategory(e.target.value)} placeholder="category" /><button className="btn-primary" aria-label="Add keyword"><Plus className="w-4 h-4" /></button></form>{!keywords.data?.length ? <EmptyState icon={<Shield className="w-5 h-5" />} title="No monitored keywords" /> : <div className="space-y-2">{keywords.data.map(item => <div key={item.id} className="flex items-center justify-between border-b border-slate-800 py-2"><span className="text-sm text-slate-200">{item.keyword}<small className="block text-xs text-slate-500">{item.category}</small></span><button className="btn-ghost text-red-400" onClick={() => remove.mutate(item.id)} aria-label={`Delete ${item.keyword}`}><Trash2 className="w-4 h-4" /></button></div>)}</div>}</section>
        <section className="card p-5"><h2 className="section-title mb-4">Recent Audit Activity</h2>{!logs.data?.length ? <EmptyState icon={<Shield className="w-5 h-5" />} title="No audit events" /> : <div className="space-y-3">{logs.data.map(log => <div key={log.id} className="border-b border-slate-800 pb-3"><p className="text-sm text-slate-200">{log.action}</p><p className="text-xs text-slate-500 mt-1">{log.user_id} · {new Date(log.created_at).toLocaleString()}</p>{log.details && <p className="text-xs text-slate-400 mt-1">{log.details}</p>}</div>)}</div>}</section>
      </div>
    </Page>
  )
}

import { useState } from 'react'
import { useMutation } from 'react-query'
import { Search, BookOpen, Clock, Users } from 'lucide-react'
import { searchApi, RAGSearchResult } from '../api'
import { Page, Spinner, EmptyState, ErrorBox } from '../components/UI'

export default function KnowledgeBase() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<RAGSearchResult | null>(null)
  const mutation = useMutation(() => searchApi.query(query), {
    onSuccess: setResult,
  })

  function submit(event: React.FormEvent) {
    event.preventDefault()
    if (query.trim()) mutation.mutate()
  }

  return (
    <Page title="Knowledge Base" subtitle="Search transcripts and meeting decisions with natural language">
      <form onSubmit={submit} className="card p-4 flex gap-3 mb-6">
        <Search className="w-5 h-5 text-slate-500 mt-2 shrink-0" />
        <input className="input flex-1" value={query} onChange={e => setQuery(e.target.value)} placeholder="What did we discuss about onboarding?" />
        <button className="btn-primary" disabled={mutation.isLoading}>{mutation.isLoading ? <Spinner className="w-4 h-4" /> : 'Search'}</button>
      </form>
      {mutation.isError && <ErrorBox message="Search failed. Check that the knowledge services are available." />}
      {result ? (
        <div className="space-y-4">
          <section className="card p-5">
            <h2 className="section-title mb-3">Answer</h2>
            <p className="text-slate-200 whitespace-pre-wrap">{result.answer}</p>
          </section>
          <section>
            <h2 className="section-title mb-3 flex items-center gap-2"><BookOpen className="w-4 h-4 text-brand-400" /> Sources</h2>
            <div className="space-y-3">
              {result.sources.map((source, index) => (
                <article key={`${source.meeting_id}-${index}`} className="card p-4">
                  <p className="text-slate-200">{source.text}</p>
                  <div className="flex flex-wrap gap-4 mt-3 text-xs text-slate-500"><span>{source.meeting_id}</span>{source.start_time !== undefined && <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {Math.round(source.start_time)}s</span>}<span className="flex items-center gap-1"><Users className="w-3 h-3" /> Meeting source</span></div>
                </article>
              ))}
            </div>
          </section>
        </div>
      ) : <EmptyState icon={<Search className="w-6 h-6" />} title="Search your meeting memory" description="Ask a question to find grounded answers across processed meetings." />}
    </Page>
  )
}

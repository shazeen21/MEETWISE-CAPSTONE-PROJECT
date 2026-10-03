import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import MeetingsList from './pages/MeetingsList'
import MeetingDetail from './pages/MeetingDetail'
import UploadMeeting from './pages/UploadMeeting'
import Employees from './pages/Employees'
import EmployeeDetail from './pages/EmployeeDetail'
import RegisterEmployee from './pages/RegisterEmployee'
import KnowledgeBase from './pages/KnowledgeBase'
import NotificationsPage from './pages/Notifications'
import AdminPage from './pages/Admin'

export default function App() {
  return (
    <div className="flex min-h-screen bg-slate-950">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/meetings" element={<MeetingsList />} />
          <Route path="/meetings/:id" element={<MeetingDetail />} />
          <Route path="/upload" element={<UploadMeeting />} />
          <Route path="/employees" element={<Employees />} />
          <Route path="/employees/register" element={<RegisterEmployee />} />
          <Route path="/employees/:id" element={<EmployeeDetail />} />
          <Route path="/search" element={<KnowledgeBase />} />
          <Route path="/notifications" element={<NotificationsPage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </main>
    </div>
  )
}

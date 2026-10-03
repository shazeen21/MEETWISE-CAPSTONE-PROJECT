import { create } from 'zustand'

interface AppState {
  sidebarOpen: boolean
  toggleSidebar: () => void

  activeMeetingId: string | null
  setActiveMeetingId: (id: string | null) => void
}

export const useAppStore = create<AppState>((set) => ({
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  activeMeetingId: null,
  setActiveMeetingId: (id) => set({ activeMeetingId: id }),
}))

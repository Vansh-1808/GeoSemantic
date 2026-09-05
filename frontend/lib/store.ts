/**
 * Global Zustand store.
 * Holds UI state that spans multiple pages.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SystemState {
  isOffline: boolean;
  dbConnected: boolean;
  modelsReady: boolean;
  setSystemStatus: (status: Partial<Pick<SystemState, "isOffline" | "dbConnected" | "modelsReady">>) => void;
}

interface IngestState {
  activeJobId: string | null;
  setActiveJobId: (id: string | null) => void;
}

interface SearchState {
  lastQuery: string;
  setLastQuery: (q: string) => void;
}

type AppStore = SystemState & IngestState & SearchState;

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      // System
      isOffline: false,
      dbConnected: false,
      modelsReady: false,
      setSystemStatus: (status) => set(status),

      // Ingestion
      activeJobId: null,
      setActiveJobId: (id) => set({ activeJobId: id }),

      // Search
      lastQuery: "",
      setLastQuery: (q) => set({ lastQuery: q }),
    }),
    {
      name: "geosemantic-store",
      partialize: (state) => ({ lastQuery: state.lastQuery }),
    }
  )
);

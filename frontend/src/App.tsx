import { useEffect, type ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";

import { StoreProvider, useStore } from "./app/store";
import { PrototypeBanner } from "./components/shared";
import {
  ConsentPage,
  ContributionChoicePage,
  DataTypesPage,
  ParticipantDashboardPage,
  ParticipantLoginPage,
  ProjectContributionPage,
} from "./pages/participant";
import { LandingPage, NotFoundPage, ProjectDetailPage, ProjectsPage } from "./pages/public";
import { ScientistDashboardPage, ScientistLoginPage, ScientistTermsPage } from "./pages/scientist";
import { TreePage } from "./pages/tree";

function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [pathname]);

  return null;
}

function RequireParticipant({ children }: { children: ReactNode }) {
  const { state, authReady } = useStore();
  const location = useLocation();
  if (!authReady) {
    return <AuthLoading />;
  }
  return state.participantAuthenticated
    ? children
    : <Navigate to="/participant/login" replace state={{ returnTo: `${location.pathname}${location.search}` }} />;
}

function RequireScientist({ terms = false, children }: { terms?: boolean; children: ReactNode }) {
  const { state, authReady } = useStore();
  if (!authReady) {
    return <AuthLoading />;
  }
  if (!state.scientistAuthenticated) {
    return <Navigate to="/scientist/login" replace />;
  }
  if (terms && !state.scientistTermsAccepted) {
    return <Navigate to="/scientist/terms" replace />;
  }
  return children;
}

function AuthLoading() {
  return <main className="grid min-h-[calc(100dvh-33px)] place-items-center bg-[var(--warm-white)]"><p className="text-sm font-semibold">Checking your session…</p></main>;
}

function AppRoutes() {
  return (
    <>
      <PrototypeBanner />
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/tree" element={<TreePage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:projectId" element={<ProjectDetailPage />} />
        <Route path="/participant/login" element={<ParticipantLoginPage />} />
        <Route path="/participant/contribution-choice" element={<RequireParticipant><ContributionChoicePage /></RequireParticipant>} />
        <Route path="/participant/data-types" element={<RequireParticipant><DataTypesPage /></RequireParticipant>} />
        <Route path="/participant/project/:projectId/contribute" element={<RequireParticipant><ProjectContributionPage /></RequireParticipant>} />
        <Route path="/participant/consent/:dataType" element={<RequireParticipant><ConsentPage /></RequireParticipant>} />
        <Route path="/participant/dashboard" element={<RequireParticipant><ParticipantDashboardPage /></RequireParticipant>} />
        <Route path="/scientist/login" element={<ScientistLoginPage />} />
        <Route path="/scientist/terms" element={<RequireScientist><ScientistTermsPage /></RequireScientist>} />
        <Route path="/scientist/dashboard" element={<RequireScientist terms><ScientistDashboardPage /></RequireScientist>} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <StoreProvider>
        <AppRoutes />
      </StoreProvider>
    </BrowserRouter>
  );
}

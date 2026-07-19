import { ArrowRight, CircleUserRound, FlaskConical, Search, ShieldCheck } from "lucide-react";
import { useDeferredValue, useId, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { useStore } from "../app/store";
import { categories, faqs } from "../data/catalog";
import type { AvailabilityState, Project } from "../lib/types";
import { cn, formatNumber } from "../lib/utils";
import { buttonVariants } from "./ui/button";

export function BrandMark({ className }: { className?: string }) {
  return <img className={className} src="/assets/diana-logo-transparent.svg" alt="" aria-hidden="true" />;
}

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <Link to="/" className="group inline-flex min-h-11 items-center rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black" aria-label="DIANA home">
      <BrandMark className={cn("h-auto transition-transform group-hover:translate-x-0.5", compact ? "w-[88px]" : "w-[112px]")} />
    </Link>
  );
}

export function PrototypeBanner() {
  return (
    <div className="border-b border-black/10 bg-black px-4 py-2 text-center text-xs font-semibold tracking-wide text-white">
      Prototype environment — do not enter real health information.
    </div>
  );
}

export function QuestionBar() {
  const { publicProjects } = useStore();
  const searchId = useId();
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const deferredQuery = useDeferredValue(query.trim().toLowerCase());
  const projectResults = publicProjects.filter((project) => `${project.title} ${project.summary}`.toLowerCase().includes(deferredQuery));
  const categoryResults = categories.filter((category) => `${category.title} ${category.description}`.toLowerCase().includes(deferredQuery));
  const faqResults = faqs.filter((faq) => `${faq.title} ${faq.text}`.toLowerCase().includes(deferredQuery));
  const hasResults = projectResults.length + categoryResults.length + faqResults.length > 0;

  return (
    <div
      className="relative w-full max-w-2xl"
      onFocus={() => setFocused(true)}
      onBlur={(event) => {
        if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) {
          setFocused(false);
        }
      }}
    >
      <label className="sr-only" htmlFor={searchId}>Ask or search DIANA</label>
      <Search className="pointer-events-none absolute left-5 top-1/2 size-4 -translate-y-1/2" aria-hidden="true" />
      <input
        id={searchId}
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        className="h-12 w-full rounded-full border border-black bg-white px-12 text-sm outline-none transition-shadow placeholder:text-neutral-500 focus:ring-2 focus:ring-[var(--purple)]"
        placeholder="Ask questions about female health…"
      />
      {focused && deferredQuery.length > 1 && (
        <div className="absolute left-0 right-0 top-14 z-40 max-h-80 overflow-y-auto rounded-3xl border border-black bg-white p-3 shadow-xl" role="status" aria-live="polite">
          {!hasResults && <p className="p-4 text-sm text-[var(--muted)]">No project or data-topic matches. DIANA does not provide medical diagnosis.</p>}
          {projectResults.slice(0, 3).map((project) => (
            <Link key={project.id} to={`/projects/${project.id}`} className="flex min-h-12 items-center justify-between rounded-2xl px-4 py-2 text-sm hover:bg-[var(--green-soft)]">
              <span><strong>{project.title}</strong><span className="block text-xs text-[var(--muted)]">Research project</span></span>
              <ArrowRight className="size-4" aria-hidden="true" />
            </Link>
          ))}
          {categoryResults.slice(0, 3).map((category) => (
            <Link key={category.id} to="/participant/data-types" className="block min-h-12 rounded-2xl px-4 py-2 text-sm hover:bg-[var(--purple-soft)]">
              <strong>{category.title}</strong><span className="block text-xs text-[var(--muted)]">Data category</span>
            </Link>
          ))}
          {faqResults.slice(0, 2).map((faq) => (
            <div key={faq.title} className="rounded-2xl px-4 py-2 text-sm">
              <strong>{faq.title}</strong><span className="block text-xs leading-5 text-[var(--muted)]">{faq.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function StandardHeader({ participant = false }: { participant?: boolean }) {
  const { state } = useStore();

  return (
    <header className="sticky top-0 z-30 border-b border-black/10 bg-[color:var(--warm-white)]/95 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1440px] items-center gap-4 px-5 py-4 lg:px-10">
        <Logo compact />
        <div className="hidden flex-1 justify-center sm:flex"><QuestionBar /></div>
        <nav className="ml-auto flex items-center gap-1 sm:gap-3" aria-label="Primary navigation">
          <Link to="/projects" className="grid min-h-11 place-items-center rounded-full px-3 text-sm font-semibold hover:bg-black/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black">Projects</Link>
          {participant && state.participantAuthenticated && (
            <Link to="/participant/dashboard" className="grid size-11 place-items-center rounded-full border border-black hover:bg-[var(--purple-soft)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black" aria-label="Participant profile and dashboard">
              <CircleUserRound className="size-5" aria-hidden="true" />
            </Link>
          )}
        </nav>
      </div>
      <div className="px-4 pb-3 sm:hidden"><QuestionBar /></div>
    </header>
  );
}

export function AuthShell({ title, step, children }: { title: string; step: number; children: ReactNode }) {
  const longTitle = title.length > 20;

  return (
    <main className="min-h-[calc(100dvh-33px)] bg-[var(--warm-white)] px-5 py-6 sm:px-10">
      <Logo />
      <div className="mx-auto flex min-h-[75dvh] max-w-4xl flex-col justify-center py-12">
        <h1 className={cn("mb-10 text-center font-bold", longTitle ? "text-3xl tracking-[-0.04em] sm:text-5xl" : "text-4xl tracking-[0.16em] sm:text-5xl")}>{title}</h1>
        {children}
      </div>
      <div className="mx-auto flex max-w-52 gap-2" aria-label={`Step ${step} of 3`}>
        {[1, 2, 3].map((item) => <span key={item} className={cn("h-1.5 flex-1 rounded-full", item <= step ? "bg-black" : "bg-black/15")} />)}
      </div>
    </main>
  );
}

export function PageTitle({ children }: { children: ReactNode }) {
  return <h1 className="rounded-full border border-black bg-white px-7 py-5 text-2xl font-semibold tracking-[-0.03em] sm:text-3xl">{children}</h1>;
}

export function StatusPill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "purple" | "green" | "warning" }) {
  return <span className={cn("inline-flex min-h-7 items-center rounded-full border border-black/25 px-3 text-xs font-semibold", tone === "purple" && "bg-[var(--purple-soft)]", tone === "green" && "bg-[var(--green-soft)] text-[var(--deep-green)]", tone === "warning" && "bg-amber-50 text-amber-900")}>{children}</span>;
}

export function ProjectRow({ project }: { project: Project }) {
  return (
    <Link to={`/projects/${project.id}`} className="group grid gap-5 rounded-[28px] border border-black bg-white p-6 transition-transform hover:-translate-y-0.5 hover:bg-[var(--green-soft)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black md:grid-cols-[1fr_auto] md:items-center">
      <div>
        <p className="mb-2 text-sm text-[var(--muted)]">{project.institution}</p>
        <h2 className="text-xl font-semibold tracking-[-0.03em] sm:text-2xl">{project.title}</h2>
        <p className="mt-2 max-w-3xl leading-7 text-[var(--muted)]">{project.summary}</p>
        <div className="mt-4 flex flex-wrap gap-2">
          {project.categories.slice(0, 5).map((id) => <StatusPill key={id} tone="green">{categories.find((category) => category.id === id)?.shortTitle}</StatusPill>)}
        </div>
      </div>
      <span className="inline-flex items-center gap-2 font-semibold">View <ArrowRight className="size-5 transition-transform group-hover:translate-x-1" aria-hidden="true" /></span>
    </Link>
  );
}

export function AvailabilityBar({ label, percent, participants, state }: { label: string; percent: number; participants: number; state: AvailabilityState }) {
  return (
    <div className="grid gap-3 border-b border-black/10 py-4 lg:grid-cols-[180px_1fr_120px] lg:items-center">
      <div><p className="font-semibold">{label}</p><p className="text-xs text-[var(--muted)]">{state}</p></div>
      <div className="h-3 overflow-hidden rounded-full border border-black bg-white" role="progressbar" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100} aria-label={`${label} availability ${percent}%`}>
        <div className="h-full rounded-full bg-gradient-to-r from-[var(--purple)] to-[var(--green)]" style={{ width: `${percent}%` }} />
      </div>
      <p className="text-sm font-semibold lg:text-right">{percent}% · {formatNumber(participants)}</p>
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="rounded-[28px] border border-dashed border-black p-10 text-center text-[var(--muted)]"><FlaskConical className="mx-auto mb-4 size-8" aria-hidden="true" />{children}</div>;
}

export function PrivacyNote() {
  return <p className="flex items-start gap-2 text-sm leading-6 text-[var(--muted)]"><ShieldCheck className="mt-0.5 size-4 shrink-0" aria-hidden="true" />This prototype uses synthetic values and does not provide diagnosis, treatment, or scientific suitability guarantees.</p>;
}

export function ResearcherLink() {
  return <Link to="/scientist/login" className={buttonVariants({ variant: "outline" })}>Researcher sign in <ArrowRight className="size-4" aria-hidden="true" /></Link>;
}

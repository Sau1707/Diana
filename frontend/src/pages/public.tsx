import { ArrowRight, Check, Database, Shield, UploadCloud } from "lucide-react";
import { useDeferredValue, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { useStore } from "../app/store";
import { categories } from "../data/catalog";
import { BrandMark, PageTitle, PrivacyNote, ProjectRow, ResearcherLink, StandardHeader, StatusPill } from "../components/shared";
import { Button, buttonVariants } from "../components/ui/button";
import { cn } from "../lib/utils";

const purposeCopy = [
  "DIANA is a consent-based health-data infrastructure platform for women and researchers.",
  "Participants can contribute selected health information, including cycle data, symptoms, sleep, activity, temperature, glucose, hormones, wearable measurements, medical history, and demographics. Participants control what they share and which research projects may use it.",
  "Researchers define the population, variables, follow-up period, completeness requirements, and study type they need. DIANA then shows how many potentially eligible participants exist, which variables are available, how complete the dataset is, and whether additional prospective data collection may be required.",
  "Once the necessary approvals and participant permissions are in place, DIANA can provide a de-identified, structured, analysis-ready dataset.",
  "DIANA does not conduct the scientific study, test medical hypotheses, create diagnostic models, or replace ethics approval.",
];

export function LandingPage() {
  const navigate = useNavigate();
  const { state, setIntent } = useStore();

  function startGeneralContribution(): void {
    setIntent({ kind: "general" });
    navigate(state.participantAuthenticated ? "/participant/contribution-choice" : "/participant/login");
  }

  return (
    <div>
      <StandardHeader />
      <main>
        <section className="soft-field flex min-h-[calc(100dvh-130px)] items-center border-b border-black px-5 py-20 sm:px-10">
          <div className="mx-auto max-w-5xl text-center">
            <h1 className="sr-only">DIANA</h1>
            <BrandMark className="mx-auto mb-12 h-auto w-[min(620px,90vw)]" />
            <p className="mx-auto max-w-3xl text-lg leading-8 text-[var(--muted)] sm:text-xl sm:leading-9">
              Our vision is to help women better understand how hormonal changes interact with symptoms, sleep, stress, nutrition, and menstrual-cycle patterns—while making responsibly consented data available for research.
            </p>
            <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Button variant="purple" size="large" onClick={startGeneralContribution}>Donate data <ArrowRight className="size-5" aria-hidden="true" /></Button>
              <Link to="/scientist/login" className={buttonVariants({ variant: "green", size: "large" })}>Research Access <ArrowRight className="size-5" aria-hidden="true" /></Link>
            </div>
            <p className="mt-4 text-xs text-[var(--muted)]">This means contributing selected data, not money.</p>
          </div>
        </section>

        <section className="border-b border-black bg-white px-5 py-24 sm:px-10">
          <div className="mx-auto max-w-6xl space-y-8">
            {[
              { icon: UploadCloud, title: "Share only what you choose", text: "Participants can contribute selected information such as cycle data, symptoms, sleep, wearable measurements, activity, temperature, glucose, laboratory results, and medical history.", tone: "bg-[var(--purple-soft)]" },
              { icon: Shield, title: "Consent and structure come first", text: "DIANA records permission, organises contributed information into consistent formats, and prepares de-identified data for approved research use.", tone: "bg-gradient-to-br from-[var(--purple-soft)] to-[var(--green-soft)]" },
              { icon: Database, title: "Researchers see what is available", text: "Researchers can evaluate population fit, variable availability, follow-up length, and data completeness before requesting access.", tone: "bg-[var(--green-soft)]" },
            ].map((item, index) => (
              <article key={item.title} className="grid items-center gap-6 border-b border-black/15 pb-8 last:border-b-0 md:grid-cols-[220px_1fr] md:gap-16">
                <div className={cn("flex h-36 items-end justify-between rounded-[28px] border border-black p-6", item.tone)}>
                  <span className="text-sm font-semibold">0{index + 1}</span>
                  <item.icon className="size-12 stroke-[1.2]" aria-hidden="true" />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold tracking-[-0.04em] sm:text-3xl">{item.title}</h2>
                  <p className="mt-3 max-w-3xl text-lg leading-8 text-[var(--muted)]">{item.text}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="overflow-hidden border-b border-black px-5 py-24 sm:px-10">
          <div className="mx-auto max-w-6xl">
            <div className="mx-auto max-w-2xl text-center">
              <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em]">How DIANA connects both sides</p>
              <h2 className="text-5xl font-semibold tracking-[-0.06em] sm:text-6xl">Data <span className="serif-accent">flow</span></h2>
            </div>
            <div className="relative mt-14 grid items-center gap-10 md:grid-cols-[0.9fr_1.1fr] md:gap-14">
              <div className="space-y-6 text-lg leading-8 text-[var(--muted)]">
                <p>Women contribute selected health data and decide how it may be used. DIANA securely structures, de-identifies, and checks the data against each participant’s consent preferences.</p>
                <p>Approved researchers submit their study requirements, and DIANA identifies matching data that can be shared for that specific project.</p>
              </div>
              <div className="flex justify-center md:justify-end">
                <Link to="/projects" className="group block w-full max-w-2xl rounded-[32px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black focus-visible:ring-offset-4" aria-label="View research projects">
                  <img src="/assets/diana-data-flow.png" alt="DIANA data flow: female health data in, scientist requests out" className="h-auto w-full transition-transform duration-200 group-hover:-translate-y-1" />
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className="bg-white px-5 py-24 sm:px-10">
          <div className="mx-auto max-w-4xl">
            <div className="text-center"><Button size="large" onClick={startGeneralContribution}>Donate data</Button></div>
            <h2 className="mt-20 text-4xl font-semibold tracking-[-0.04em] underline decoration-1 underline-offset-8">Purpose</h2>
            <div className="mt-10 space-y-6 text-lg leading-8 text-[var(--muted)]">
              {purposeCopy.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
            </div>
            <div className="mt-10 flex flex-wrap items-center gap-4"><ResearcherLink /><PrivacyNote /></div>
          </div>
        </section>
      </main>
    </div>
  );
}

export function ProjectsPage() {
  const { publicProjects } = useStore();
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const deferredQuery = useDeferredValue(query.toLowerCase());
  const filtered = publicProjects.filter((project) => {
    const matchesText = `${project.title} ${project.institution} ${project.summary}`.toLowerCase().includes(deferredQuery);
    return matchesText && (category === "" || project.categories.some((id) => id === category));
  });

  return (
    <div>
      <StandardHeader />
      <main className="mx-auto min-h-[80dvh] max-w-6xl px-5 py-14 sm:px-10 sm:py-20">
        <PageTitle>Research projects</PageTitle>
        <p className="mt-5 max-w-2xl text-lg leading-8 text-[var(--muted)]">Explore synthetic demo studies seeking longitudinal women’s health data.</p>
        <div className="my-10 grid gap-4 rounded-[28px] border border-black bg-[var(--neutral)] p-4 sm:grid-cols-[1fr_280px]">
          <div><label className="mb-2 block text-sm font-semibold" htmlFor="project-search">Search projects</label><input id="project-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Project, institution, or topic" /></div>
          <div><label className="mb-2 block text-sm font-semibold" htmlFor="category-filter">Data type</label><select id="category-filter" value={category} onChange={(event) => setCategory(event.target.value)}><option value="">All data types</option>{categories.map((item) => <option key={item.id} value={item.id}>{item.shortTitle}</option>)}</select></div>
        </div>
        <p className="mb-5 text-sm text-[var(--muted)]" aria-live="polite">{filtered.length} demo project{filtered.length === 1 ? "" : "s"}</p>
        <div className="space-y-5">
          {filtered.map((project) => <ProjectRow key={project.id} project={project} />)}
          {filtered.length === 0 && <div className="rounded-[28px] border border-dashed border-black p-12 text-center"><h2 className="text-xl font-semibold">No matching projects</h2><p className="mt-2 text-[var(--muted)]">Try a broader search or choose all data types.</p></div>}
        </div>
      </main>
    </div>
  );
}

export function ProjectDetailPage() {
  const { projectId } = useParams();
  const navigate = useNavigate();
  const { publicProjects, state, setIntent } = useStore();
  const project = publicProjects.find((item) => item.id === projectId);

  if (project === undefined) {
    return <NotFoundPage />;
  }

  const contributionProjectId = project.id;

  function contribute(): void {
    setIntent({ kind: "project", projectId: contributionProjectId });
    navigate(state.participantAuthenticated ? `/participant/project/${contributionProjectId}/contribute` : "/participant/login");
  }

  return (
    <div>
      <StandardHeader participant />
      <main className="mx-auto max-w-7xl px-5 py-12 sm:px-10 sm:py-16">
        <Link to="/projects" className="mb-6 inline-flex min-h-11 items-center gap-2 rounded-full font-semibold hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black">← All projects</Link>
        <div className="rounded-full border border-black bg-[var(--green-soft)] px-7 py-6">
          <div className="flex flex-wrap items-center gap-3"><StatusPill tone="green">Demo project</StatusPill><span className="text-sm text-[var(--muted)]">{project.institution}</span></div>
          <h1 className="mt-3 text-3xl font-semibold tracking-[-0.05em] sm:text-5xl">{project.title}</h1>
        </div>
        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <section className="rounded-[32px] border border-black bg-white p-7 sm:p-9">
            <h2 className="text-3xl font-semibold tracking-[-0.04em]">Data requested</h2>
            <ul className="mt-7 divide-y divide-black/10">
              {project.requestedVariables.map((item) => <li key={item} className="flex items-center gap-3 py-3"><Check className="size-4 text-[var(--deep-green)]" aria-hidden="true" />{item}</li>)}
            </ul>
            <dl className="mt-8 grid gap-5 rounded-3xl bg-[var(--neutral)] p-5 sm:grid-cols-2">
              <div><dt className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">Minimum follow-up</dt><dd className="mt-1 font-semibold">{project.minimumFollowUp}</dd></div>
              <div><dt className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">Collection</dt><dd className="mt-1 font-semibold">{project.collectionType}</dd></div>
              <div className="sm:col-span-2"><dt className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">Target population</dt><dd className="mt-1 font-semibold">{project.targetPopulation}</dd></div>
            </dl>
          </section>
          <section className="rounded-[32px] border border-black bg-[var(--green-soft)] p-7 sm:p-9">
            <h2 className="text-3xl font-semibold tracking-[-0.04em]">Abstract</h2>
            <p className="mt-7 leading-7">{project.abstract}</p>
            <dl className="mt-8 space-y-5">
              <div><dt className="font-semibold">Research question</dt><dd className="mt-1 leading-7 text-[var(--muted)]">{project.researchQuestion}</dd></div>
              <div><dt className="font-semibold">Intended use</dt><dd className="mt-1 leading-7 text-[var(--muted)]">{project.intendedUse}</dd></div>
              <div className="grid gap-5 sm:grid-cols-2"><div><dt className="font-semibold">Expected study period</dt><dd className="mt-1 text-[var(--muted)]">{project.studyPeriod}</dd></div><div><dt className="font-semibold">Ethics status</dt><dd className="mt-1 text-[var(--muted)]">{project.ethicsStatus}</dd></div></div>
            </dl>
          </section>
        </div>
        <div className="mt-8 flex flex-wrap items-center gap-5"><Button size="large" onClick={contribute}>Donate data to this project <ArrowRight className="size-5" aria-hidden="true" /></Button><PrivacyNote /></div>
      </main>
    </div>
  );
}

export function NotFoundPage() {
  return (
    <main className="grid min-h-dvh place-items-center px-5 text-center">
      <div><p className="text-sm font-semibold uppercase tracking-[0.2em]">404</p><h1 className="mt-3 text-5xl font-semibold tracking-[-0.06em]">Page not found</h1><p className="mt-4 text-[var(--muted)]">The requested DIANA prototype page does not exist.</p><Link to="/" className={cn(buttonVariants({ variant: "primary" }), "mt-8")}>Return home</Link></div>
    </main>
  );
}

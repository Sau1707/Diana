import { ArrowRight, FileText, ShieldCheck } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate, useParams, useSearchParams } from "react-router-dom";

import { useStore } from "../app/store";
import { AuthShell, PageTitle, PrivacyNote, StandardHeader, StatusPill } from "../components/shared";
import { Accordion, AccordionItem } from "../components/ui/accordion";
import { Button, buttonVariants } from "../components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "../components/ui/dialog";
import { categories, categoryById } from "../data/catalog";
import { matchingCategories } from "../lib/matching";
import type { ConsentRecord, DataCategoryId } from "../lib/types";
import { cn } from "../lib/utils";

function isCategoryId(value: string): value is DataCategoryId {
  return categories.some((category) => category.id === value);
}

export function ParticipantLoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { state, authReady, signInParticipant } = useStore();
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (authReady && state.participantAuthenticated && !submitting) {
      const projectId = state.intent?.kind === "project" ? state.intent.projectId : undefined;
      navigate(projectId === undefined ? "/participant/contribution-choice" : `/participant/project/${projectId}/contribute`, { replace: true });
    }
  }, [authReady, navigate, state.intent, state.participantAuthenticated, submitting]);

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const username = String(form.get("username") ?? "").trim();
    const password = String(form.get("password") ?? "");
    setSubmitting(true);
    const authError = await signInParticipant(username, password);
    if (authError !== null) {
      setSubmitting(false);
      setError(authError);
      return;
    }

    setError("");
    const navigationState: unknown = location.state;
    const returnTo = typeof navigationState === "object"
      && navigationState !== null
      && "returnTo" in navigationState
      && typeof navigationState.returnTo === "string"
      && navigationState.returnTo.startsWith("/participant/")
      ? navigationState.returnTo
      : null;
    const projectId = state.intent?.kind === "project" ? state.intent.projectId : undefined;
    navigate(returnTo ?? (projectId === undefined ? "/participant/contribution-choice" : `/participant/project/${projectId}/contribute`));
  }

  return (
    <AuthShell title="LOG IN" step={1}>
      <form onSubmit={submit} className="mx-auto grid w-full max-w-2xl gap-5">
        <div><label className="mb-2 block font-semibold" htmlFor="participant-username">Username</label><input id="participant-username" name="username" autoComplete="username" required placeholder="Enter your username" /></div>
        <div><label className="mb-2 block font-semibold" htmlFor="participant-password">Password</label><input id="participant-password" name="password" type="password" minLength={1} autoComplete="current-password" required /></div>
        {error !== "" && <p className="rounded-2xl bg-red-50 p-4 text-sm font-semibold text-[var(--error)]" role="alert">{error}</p>}
        <div className="mt-3 flex justify-end"><Button type="submit" size="large" disabled={submitting}>{submitting ? "Signing in…" : "Sign in"}<ArrowRight className="size-6" aria-hidden="true" /></Button></div>
        <PrivacyNote />
      </form>
    </AuthShell>
  );
}

export function ContributionChoicePage() {
  const navigate = useNavigate();
  const { state } = useStore();
  const selectedProject = state.intent?.kind === "project" ? state.intent.projectId : undefined;

  return (
    <AuthShell title="What data is collected?" step={2}>
      <div className="mx-auto grid w-full max-w-4xl gap-5 md:grid-cols-2">
        <button onClick={() => navigate("/participant/data-types")} className="group min-h-64 rounded-[32px] border border-black bg-[var(--purple-soft)] p-8 text-left transition-transform hover:-translate-y-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black">
          <span className="flex h-full flex-col justify-between"><FileText className="size-10 stroke-[1.3]" aria-hidden="true" /><span className="flex items-center justify-between gap-4 text-2xl font-semibold">Data types <ArrowRight className="transition-transform group-hover:translate-x-1" aria-hidden="true" /></span></span>
        </button>
        <button onClick={() => navigate(selectedProject === undefined ? "/projects" : `/participant/project/${selectedProject}/contribute`)} className="group min-h-64 rounded-[32px] border border-black bg-[var(--green-soft)] p-8 text-left transition-transform hover:-translate-y-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black">
          <span className="flex h-full flex-col justify-between"><ShieldCheck className="size-10 stroke-[1.3]" aria-hidden="true" /><span className="flex items-center justify-between gap-4 text-2xl font-semibold">{selectedProject === undefined ? "Data for specific projects" : "Continue with selected project"}<ArrowRight className="shrink-0 transition-transform group-hover:translate-x-1" aria-hidden="true" /></span></span>
        </button>
      </div>
    </AuthShell>
  );
}

export function DataTypesPage() {
  const navigate = useNavigate();
  const { state, setIntent } = useStore();

  function openConsent(id: DataCategoryId): void {
    setIntent({ kind: "general" });
    navigate(`/participant/consent/${id}`);
  }

  return (
    <div>
      <StandardHeader participant />
      <main className="mx-auto max-w-6xl px-5 py-12 sm:px-10 sm:py-16">
        <PageTitle>Data types</PageTitle>
        <p className="mt-5 max-w-3xl leading-7 text-[var(--muted)]">Choose one category at a time. “Donate data” means sharing selected data after reviewing consent; it never refers to money.</p>
        <div className="mt-9 space-y-4">
          {categories.map((category) => {
            const broadConsent = state.consents.find((consent) => consent.status === "active" && consent.scope === "approved-projects" && consent.categoryIds.includes(category.id));
            const projectConsent = state.consents.find((consent) => consent.status === "active" && consent.scope === "project" && consent.categoryIds.includes(category.id));
            const contributed = broadConsent !== undefined;
            const timingPermitted = broadConsent?.options.retrospectiveData === true || broadConsent?.options.prospectiveCollection === true;
            return (
              <article key={category.id} className="grid gap-5 rounded-[26px] border border-black bg-white p-6 md:grid-cols-[1fr_auto] md:items-center">
                <div><div className="flex flex-wrap items-center gap-3"><h2 className="text-xl font-semibold tracking-[-0.03em]">{category.title}</h2><StatusPill tone={contributed && !timingPermitted ? "warning" : contributed ? "purple" : projectConsent !== undefined ? "green" : "neutral"}>{contributed && !timingPermitted ? "Recorded · project use not permitted" : contributed ? "Broad consent active" : projectConsent !== undefined ? "Project consent active" : "Not contributed"}</StatusPill></div><p className="mt-2 max-w-3xl leading-7 text-[var(--muted)]">{category.description}</p><p className="mt-2 text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">Mock method: {category.method}</p></div>
                {contributed ? <Link className={buttonVariants({ variant: "outline" })} to="/participant/dashboard">View consent</Link> : <Button variant="purple" onClick={() => openConsent(category.id)}>Donate data</Button>}
              </article>
            );
          })}
        </div>
      </main>
    </div>
  );
}

export function ProjectContributionPage() {
  const navigate = useNavigate();
  const { projectId } = useParams();
  const { allProjects, state } = useStore();
  const project = allProjects.find((item) => item.id === projectId);
  const [selected, setSelected] = useState<DataCategoryId[]>([]);

  if (project === undefined) {
    return <main className="grid min-h-dvh place-items-center"><Link to="/projects">Project not found. Return to projects.</Link></main>;
  }

  const contributionProjectId = project.id;

  const alreadyContributed = new Set(matchingCategories(project, state.consents, allProjects));

  function toggle(id: DataCategoryId): void {
    setSelected((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]);
  }

  function continueToConsent(): void {
    if (selected.length === 0) {
      return;
    }
    const params = new URLSearchParams({ project: contributionProjectId, categories: selected.join(",") });
    navigate(`/participant/consent/project-request?${params.toString()}`);
  }

  return (
    <div>
      <StandardHeader participant />
      <main className="mx-auto max-w-7xl px-5 py-12 sm:px-10 sm:py-16">
        <PageTitle>{project.title}</PageTitle>
        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <section className="rounded-[32px] border border-black bg-white p-7 sm:p-9">
            <h2 className="text-3xl font-semibold tracking-[-0.04em]">Data requested</h2>
            <p className="mt-3 leading-7 text-[var(--muted)]">Select only the requested categories you wish to contribute.</p>
            <div className="mt-6 space-y-3">
              {project.categories.map((id) => {
                const category = categoryById(id);
                if (category === undefined) return null;
                const completed = alreadyContributed.has(id);
                return <label key={id} className={cn("flex min-h-16 items-center gap-4 rounded-2xl border border-black p-4", completed ? "cursor-default bg-[var(--purple-soft)]" : "cursor-pointer bg-white hover:bg-[var(--neutral)]")}><input type="checkbox" checked={completed || selected.includes(id)} disabled={completed} onChange={() => toggle(id)} /><span><strong>{category.shortTitle}</strong><span className="block text-xs text-[var(--muted)]">{completed ? "Project consent active" : category.description}</span></span></label>;
              })}
            </div>
          </section>
          <section className="rounded-[32px] border border-black bg-[var(--green-soft)] p-7 sm:p-9">
            <h2 className="text-3xl font-semibold tracking-[-0.04em]">About this project</h2>
            <p className="mt-6 leading-7">{project.abstract}</p>
            <div className="mt-8 space-y-4 border-t border-black/15 pt-6"><p><strong>Intended use</strong><span className="mt-1 block text-[var(--muted)]">{project.intendedUse}</span></p><p><strong>Minimum follow-up</strong><span className="mt-1 block text-[var(--muted)]">{project.minimumFollowUp}</span></p><p><strong>Ethics status</strong><span className="mt-1 block text-[var(--muted)]">{project.ethicsStatus}</span></p></div>
          </section>
        </div>
        <div className="mt-8 flex flex-wrap items-center gap-4"><Button size="large" disabled={selected.length === 0} onClick={continueToConsent}>Donate data <ArrowRight className="size-5" aria-hidden="true" /></Button><span className="text-sm text-[var(--muted)]">Consent is required before any category is marked contributed.</span></div>
      </main>
    </div>
  );
}

export function ConsentPage() {
  const navigate = useNavigate();
  const { dataType } = useParams();
  const [searchParams] = useSearchParams();
  const { allProjects, state, recordConsent } = useStore();
  const [error, setError] = useState("");
  const [complete, setComplete] = useState(false);
  const projectId = searchParams.get("project");
  const project = projectId === null ? undefined : allProjects.find((item) => item.id === projectId);
  const requestedIds = dataType === "project-request"
    ? (searchParams.get("categories") ?? "").split(",").filter(isCategoryId)
    : dataType !== undefined && isCategoryId(dataType) ? [dataType] : [];
  const requestedCategories = requestedIds.map(categoryById).filter((category) => category !== undefined);
  const today = new Date().toISOString().slice(0, 10);
  const supportedCount = allProjects.filter((item) => matchingCategories(item, state.consents, allProjects).length > 0).length;

  if (requestedCategories.length === 0) {
    return <main className="grid min-h-dvh place-items-center"><Link to="/participant/data-types">Choose a valid data category.</Link></main>;
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const readSummary = form.get("readSummary") === "on";
    const voluntary = form.get("voluntary") === "on";
    const signedName = String(form.get("signedName") ?? "").trim();
    const signedOn = String(form.get("signedOn") ?? "");
    const generalScope = form.get("scope");
    const selectedProjectId = String(form.get("selectedProject") ?? "");
    const selectedProject = allProjects.find((item) => item.id === selectedProjectId);
    if (!readSummary || !voluntary || signedName.length < 2 || signedOn === "") {
      setError("Confirm both required statements, enter your name, and provide the consent date.");
      return;
    }
    if (project === undefined && generalScope !== "approved-projects" && generalScope !== "project") {
      setError("Choose whether this contribution is for one project or eligible approved projects.");
      return;
    }
    if (project === undefined && generalScope === "project" && selectedProject === undefined) {
      setError("Select the project that may use this contribution.");
      return;
    }
    if (project === undefined && generalScope === "project" && selectedProject !== undefined && !selectedProject.categories.some((id) => requestedIds.includes(id))) {
      setError("Select a project that requests this data category.");
      return;
    }

    recordConsent({
      categoryIds: requestedIds,
      projectId: project?.id ?? selectedProject?.id ?? null,
      scope: project !== undefined || generalScope === "project" ? "project" : "approved-projects",
      signedName,
      signedOn,
      options: {
        similarProjects: form.get("similarProjects") === "on",
        futureNotifications: form.get("futureNotifications") === "on",
        retrospectiveData: form.get("retrospectiveData") === "on",
        prospectiveCollection: form.get("prospectiveCollection") === "on",
      },
    });
    setError("");
    setComplete(true);
  }

  return (
    <AuthShell title="Consent form" step={3}>
      <form onSubmit={submit} className="mx-auto w-full max-w-4xl rounded-[32px] border border-black bg-white p-6 sm:p-9">
        <div className="rounded-3xl bg-gradient-to-r from-[var(--purple-soft)] to-[var(--green-soft)] p-6">
          <p className="text-sm font-semibold uppercase tracking-[0.16em]">Plain-language summary</p>
          <h2 className="mt-3 text-2xl font-semibold tracking-[-0.03em]">Sharing: {requestedCategories.map((category) => category.shortTitle).join(", ")}</h2>
          <p className="mt-3 leading-7 text-[var(--muted)]">{project === undefined ? "This permission can support eligible approved research projects that request the selected category." : `This permission is for ${project.title}.`}</p>
          <dl className="mt-5 grid gap-4 text-sm sm:grid-cols-2"><div><dt className="font-semibold">Why requested</dt><dd className="mt-1 text-[var(--muted)]">To prepare structured data for the described, approved research purpose.</dd></div><div><dt className="font-semibold">Who may access it</dt><dd className="mt-1 text-[var(--muted)]">Approved research teams receiving de-identified data through governance checks.</dd></div><div><dt className="font-semibold">Permission period</dt><dd className="mt-1 text-[var(--muted)]">Until withdrawal or the end of the approved project retention period.</dd></div><div><dt className="font-semibold">Withdrawal</dt><dd className="mt-1 text-[var(--muted)]">Future access can stop; completed research may not be reversible.</dd></div></dl>
        </div>

        <div className="mt-7"><Accordion><AccordionItem value="included" title="Data included">Only the selected category and the examples shown on the contribution page are included in this prototype permission.</AccordionItem><AccordionItem value="purpose" title="Purpose">Data may be used only for the approved purpose described by the selected project or eligible approved projects.</AccordionItem><AccordionItem value="access" title="Access">Researchers receive de-identified, structured data only after approval and permission checks. DIANA does not expose participant identities.</AccordionItem><AccordionItem value="retention" title="Retention">The approved project must follow its stated data-use and retention period.</AccordionItem><AccordionItem value="withdrawal" title="Withdrawal">You may stop future access. Analyses or completed research using previously permitted data may not be reversible.</AccordionItem><AccordionItem value="protection" title="Data protection">This prototype simulates data handling and must not be used for real health information. DIANA is not providing diagnosis or treatment.</AccordionItem></Accordion></div>

        <fieldset className="mt-7 space-y-3"><legend className="mb-3 text-lg font-semibold">Permission scope</legend>{project === undefined ? <><label className="flex items-start gap-3 rounded-2xl border border-black p-4"><input className="mt-1" type="radio" name="scope" value="approved-projects" /><span><strong>Eligible approved projects</strong><span className="block text-sm text-[var(--muted)]">Allow this category to match approved research projects with compatible requirements.</span></span></label><label className="flex items-start gap-3 rounded-2xl border border-black p-4"><input className="mt-1" type="radio" name="scope" value="project" /><span className="w-full"><strong>One project only</strong><span className="mb-3 block text-sm text-[var(--muted)]">Choose a single approved project for this category.</span><select name="selectedProject" aria-label="Project-specific permission"><option value="">Select a project</option>{allProjects.filter((item) => item.isPublic).map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select></span></label></> : <div className="rounded-2xl border border-black bg-[var(--green-soft)] p-4"><strong>This project only</strong><p className="mt-1 text-sm text-[var(--muted)]">{project.title}</p></div>}</fieldset>

        <fieldset className="mt-7 space-y-3"><legend className="mb-1 text-lg font-semibold">Optional permissions</legend><p className="mb-4 text-sm text-[var(--muted)]">These are unselected by default.</p><ConsentCheckbox name="similarProjects" label="Share with similar approved research projects" />
          <ConsentCheckbox name="futureNotifications" label="Allow DIANA to notify me about future matching studies" />
          <ConsentCheckbox name="retrospectiveData" label="Permit retrospective data" />
          <ConsentCheckbox name="prospectiveCollection" label="Permit new prospective collection" />
        </fieldset>

        <fieldset className="mt-7 space-y-3"><legend className="mb-3 text-lg font-semibold">Required confirmations</legend><ConsentCheckbox name="readSummary" label="I have read the plain-language summary." required /><ConsentCheckbox name="voluntary" label="I voluntarily consent to the selected use and understand I can stop future access." required /></fieldset>

        <div className="mt-7 grid gap-5 sm:grid-cols-2"><div><label className="mb-2 block font-semibold" htmlFor="signed-name">Typed name</label><input id="signed-name" name="signedName" required autoComplete="name" /></div><div><label className="mb-2 block font-semibold" htmlFor="signed-on">Date</label><input id="signed-on" name="signedOn" type="date" required defaultValue={today} /></div></div>
        {error !== "" && <p className="mt-5 rounded-2xl bg-red-50 p-4 text-sm font-semibold text-[var(--error)]" role="alert">{error}</p>}
        <div className="mt-8 flex flex-wrap-reverse items-center justify-between gap-4"><Button variant="outline" onClick={() => navigate(-1)}>Go back</Button><Button type="submit" size="large">Sign and continue <ArrowRight className="size-5" aria-hidden="true" /></Button></div>
      </form>

      <Dialog open={complete} onOpenChange={() => undefined}>
        <DialogContent hideClose>
          <DialogHeader><DialogTitle>Enough data?</DialogTitle><DialogDescription>Your permission has been recorded. Your data currently contributes to {supportedCount} research project{supportedCount === 1 ? "" : "s"}.</DialogDescription></DialogHeader>
          <p className="leading-7">Would you like to increase this number by contributing another data type? You can stop at any time.</p>
          <div className="mt-8 flex flex-wrap justify-end gap-3"><Button variant="outline" onClick={() => navigate("/participant/dashboard")}>Finish</Button><Button variant="purple" onClick={() => navigate("/participant/data-types")}>Donate more</Button></div>
        </DialogContent>
      </Dialog>
    </AuthShell>
  );
}

function ConsentCheckbox({ name, label, required = false }: { name: string; label: string; required?: boolean }) {
  return <label className="flex min-h-12 items-start gap-3 rounded-2xl border border-black/20 p-3 hover:bg-[var(--neutral)]"><input className="mt-1" type="checkbox" name={name} required={required} /><span className="text-sm leading-6">{label}</span></label>;
}

export function ParticipantDashboardPage() {
  const navigate = useNavigate();
  const { state, authenticatedUsername, allProjects, updateConsent, withdrawConsent, signOut } = useStore();
  const [selectedConsent, setSelectedConsent] = useState<ConsentRecord | null>(null);
  const [mode, setMode] = useState<"review" | "modify" | "withdraw" | null>(null);
  const [manageError, setManageError] = useState("");
  const [logoutError, setLogoutError] = useState("");
  const activeConsents = state.consents.filter((consent) => consent.status === "active");
  const supportedProjects = allProjects.map((project) => ({ project, matches: matchingCategories(project, state.consents, allProjects) })).filter((item) => item.matches.length > 0);

  function openConsent(consent: ConsentRecord, nextMode: "review" | "modify" | "withdraw"): void {
    setSelectedConsent(consent);
    setManageError("");
    setMode(nextMode);
  }

  function submitModification(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    if (selectedConsent === null) return;
    const form = new FormData(event.currentTarget);
    const scope = form.get("scope") === "approved-projects" ? "approved-projects" : "project";
    const projectId = scope === "project" ? String(form.get("projectId") ?? "") : null;
    if (scope === "project" && !allProjects.some((project) => project.id === projectId)) {
      setManageError("Select the project that should receive future permission.");
      return;
    }
    const selectedProject = allProjects.find((project) => project.id === projectId);
    if (scope === "project" && selectedProject !== undefined && !selectedProject.categories.some((id) => selectedConsent.categoryIds.includes(id))) {
      setManageError("Select a project that requests at least one consented category.");
      return;
    }
    updateConsent(selectedConsent.id, scope, projectId, {
      similarProjects: form.get("similarProjects") === "on",
      futureNotifications: form.get("futureNotifications") === "on",
      retrospectiveData: form.get("retrospectiveData") === "on",
      prospectiveCollection: form.get("prospectiveCollection") === "on",
    });
    setManageError("");
    setMode(null);
  }

  function confirmWithdrawal(): void {
    if (selectedConsent === null) return;
    withdrawConsent(selectedConsent.id);
    setMode(null);
  }

  async function logout(): Promise<void> {
    const authError = await signOut();
    if (authError !== null) {
      setLogoutError(authError);
      return;
    }
    setLogoutError("");
    navigate("/participant/login", { replace: true });
  }

  return (
    <div>
      <StandardHeader participant />
      <main className="mx-auto max-w-7xl px-5 py-12 sm:px-10 sm:py-16">
        <div className="grid gap-4 sm:grid-cols-[1fr_auto] sm:items-center"><PageTitle>Your contribution to research</PageTitle><div className="flex items-center justify-between gap-4 rounded-full border border-black bg-white px-5 py-3 sm:justify-start"><span className="text-sm font-semibold">{authenticatedUsername ?? "Participant"}</span><Button variant="ghost" size="small" onClick={() => void logout()}>Sign out</Button></div></div>
        {logoutError !== "" && <p className="mt-4 rounded-2xl bg-red-50 p-4 text-sm font-semibold text-[var(--error)]" role="alert">{logoutError}</p>}
        <div className="mt-8 grid gap-4 sm:grid-cols-3"><Summary value={state.contributedCategoryIds.length} label="Data categories contributed" tone="purple" /><Summary value={activeConsents.length} label="Active permissions" tone="neutral" /><Summary value={supportedProjects.length} label="Projects your data may support" tone="green" /></div>

        <section className="mt-16"><h2 className="text-3xl font-semibold tracking-[-0.04em]">Projects your data may support</h2><div className="mt-6 grid gap-5 lg:grid-cols-2">{supportedProjects.map(({ project, matches }) => <article key={project.id} className="rounded-[28px] border border-black bg-white p-6"><p className="text-sm text-[var(--muted)]">{project.institution}</p><h3 className="mt-2 text-xl font-semibold">{project.title}</h3><p className="mt-3 text-sm text-[var(--muted)]">Your contribution matches {matches.length} requested data type{matches.length === 1 ? "" : "s"} for this project.</p><div className="mt-4 flex flex-wrap gap-2">{matches.map((id) => <StatusPill key={id} tone="purple">{categoryById(id)?.shortTitle}</StatusPill>)}</div><div className="mt-6 flex items-center justify-between gap-4"><StatusPill tone="green">Permission active</StatusPill><Link to={`/projects/${project.id}`} className="font-semibold underline underline-offset-4">View project</Link></div></article>)}{supportedProjects.length === 0 && <div className="rounded-[28px] border border-dashed border-black p-8 text-[var(--muted)]">No active contribution currently matches a project.</div>}</div></section>

        <section className="mt-16"><h2 className="text-3xl font-semibold tracking-[-0.04em]">Your contributed data</h2><div className="mt-6 overflow-hidden rounded-[28px] border border-black bg-white">{categories.map((category) => { const records = state.consents.filter((consent) => consent.categoryIds.includes(category.id)); const active = records.some((consent) => consent.status === "active"); const withdrawn = records.length > 0 && !active; return <div key={category.id} className="flex min-h-16 flex-wrap items-center justify-between gap-3 border-b border-black/10 px-5 py-3 last:border-0"><span className="font-semibold">{category.title}</span><StatusPill tone={active ? "purple" : withdrawn ? "warning" : "neutral"}>{active ? "Contributed · consent active" : withdrawn ? "Consent withdrawn" : "Not contributed"}</StatusPill></div>; })}</div></section>

        <section className="mt-16 rounded-[32px] border border-black bg-gradient-to-r from-[var(--purple-soft)] to-[var(--green-soft)] p-7 sm:p-10"><h2 className="text-3xl font-semibold tracking-[-0.04em]">Where your data goes</h2><div className="mt-8 grid gap-3 md:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr] md:items-center">{["Your contribution", "Consent recorded", "Data de-identified", "Approved project dataset"].map((label, index) => <div key={label} className="contents"><div className="rounded-2xl border border-black bg-white p-5 text-center font-semibold">{label}</div>{index < 3 && <ArrowRight className="mx-auto hidden size-5 md:block" aria-hidden="true" />}</div>)}</div></section>

        <section className="mt-16"><h2 className="text-3xl font-semibold tracking-[-0.04em]">Manage permissions</h2><div className="mt-6 space-y-4">{state.consents.map((consent) => <article key={consent.id} className="grid gap-4 rounded-[24px] border border-black bg-white p-5 lg:grid-cols-[1fr_auto] lg:items-center"><div><div className="flex flex-wrap items-center gap-3"><strong>{consent.categoryIds.map((id) => categoryById(id)?.shortTitle).join(", ")}</strong><StatusPill tone={consent.status === "active" ? "green" : "warning"}>{consent.status === "active" ? "Consent active" : "Consent withdrawn"}</StatusPill></div><p className="mt-2 text-sm text-[var(--muted)]">Signed {consent.signedOn} · {consent.scope === "project" ? "Project-specific" : "Eligible approved projects"}</p></div><div className="flex flex-wrap gap-2"><Button variant="ghost" size="small" onClick={() => openConsent(consent, "review")}>View consent</Button>{consent.status === "active" && <><Button variant="outline" size="small" onClick={() => openConsent(consent, "modify")}>Change future permission</Button><Button variant="danger" size="small" onClick={() => openConsent(consent, "withdraw")}>Withdraw future access</Button></>}</div></article>)}</div></section>
      </main>

      <Dialog open={mode !== null} onOpenChange={(open) => { if (!open) setMode(null); }}>
        <DialogContent>
          {selectedConsent !== null && mode === "review" && <><DialogHeader><DialogTitle>Consent record</DialogTitle><DialogDescription>Plain-language summary of the permission recorded on {selectedConsent.signedOn}.</DialogDescription></DialogHeader><div className="space-y-4 rounded-3xl bg-[var(--purple-soft)] p-6"><p><strong>Data</strong><span className="block text-[var(--muted)]">{selectedConsent.categoryIds.map((id) => categoryById(id)?.title).join(", ")}</span></p><p><strong>Scope</strong><span className="block text-[var(--muted)]">{selectedConsent.scope === "project" ? "One selected project" : "Eligible approved projects"}</span></p><p><strong>Status</strong><span className="block capitalize text-[var(--muted)]">{selectedConsent.status}</span></p></div><div className="mt-6 text-right"><DialogClose className={buttonVariants({ variant: "primary" })}>Close</DialogClose></div></>}
          {selectedConsent !== null && mode === "modify" && <><DialogHeader><DialogTitle>Change future permission</DialogTitle><DialogDescription>Changes apply to future access. Already-completed research may not be reversible.</DialogDescription></DialogHeader><form onSubmit={submitModification} className="space-y-4"><label className="flex gap-3 rounded-2xl border border-black p-4"><input type="radio" name="scope" value="project" defaultChecked={selectedConsent.scope === "project"} /><span>Selected project only</span></label><div><label className="mb-2 block text-sm font-semibold" htmlFor="permission-project">Project for future access</label><select id="permission-project" name="projectId" defaultValue={selectedConsent.projectId ?? ""}><option value="">Select a project</option>{allProjects.filter((project) => project.isPublic).map((project) => <option key={project.id} value={project.id}>{project.title}</option>)}</select></div><label className="flex gap-3 rounded-2xl border border-black p-4"><input type="radio" name="scope" value="approved-projects" defaultChecked={selectedConsent.scope === "approved-projects"} /><span>Eligible approved projects</span></label><p className="pt-2 text-sm font-semibold">Optional permissions</p><label className="flex gap-3 rounded-2xl border border-black/20 p-4"><input type="checkbox" name="similarProjects" defaultChecked={selectedConsent.options.similarProjects} /><span>Allow similar approved projects with the same collection type and overlapping requirements</span></label><label className="flex gap-3 rounded-2xl border border-black/20 p-4"><input type="checkbox" name="futureNotifications" defaultChecked={selectedConsent.options.futureNotifications} /><span>Notify me about future matching studies</span></label><label className="flex gap-3 rounded-2xl border border-black/20 p-4"><input type="checkbox" name="retrospectiveData" defaultChecked={selectedConsent.options.retrospectiveData} /><span>Permit retrospective data</span></label><label className="flex gap-3 rounded-2xl border border-black/20 p-4"><input type="checkbox" name="prospectiveCollection" defaultChecked={selectedConsent.options.prospectiveCollection} /><span>Permit new prospective collection</span></label>{manageError !== "" && <p className="rounded-2xl bg-red-50 p-4 text-sm font-semibold text-[var(--error)]" role="alert">{manageError}</p>}<div className="flex justify-end gap-3 pt-4"><Button variant="outline" onClick={() => setMode(null)}>Cancel</Button><Button type="submit">Save change</Button></div></form></>}
          {selectedConsent !== null && mode === "withdraw" && <><DialogHeader><DialogTitle>Withdraw future access?</DialogTitle><DialogDescription>Future matching and access for this permission will stop. Research already completed may not be reversible.</DialogDescription></DialogHeader><div className="flex justify-end gap-3"><Button variant="outline" onClick={() => setMode(null)}>Keep permission</Button><Button variant="danger" onClick={confirmWithdrawal}>Withdraw future access</Button></div></>}
        </DialogContent>
      </Dialog>
    </div>
  );
}

function Summary({ value, label, tone }: { value: number; label: string; tone: "purple" | "green" | "neutral" }) {
  return <div className={cn("rounded-[28px] border border-black p-6", tone === "purple" && "bg-[var(--purple-soft)]", tone === "green" && "bg-[var(--green-soft)]", tone === "neutral" && "bg-white")}><strong className="text-4xl tracking-[-0.05em]">{value}</strong><p className="mt-2 text-sm text-[var(--muted)]">{label}</p></div>;
}

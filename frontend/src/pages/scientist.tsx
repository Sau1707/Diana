import { ArrowRight, CircleUserRound, Download, FileCheck2, FlaskConical, Menu, Plus, ShieldAlert, XCircle } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useStore } from "../app/store";
import { AuthShell, AvailabilityBar, Logo, PrivacyNote, StatusPill } from "../components/shared";
import { Button } from "../components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "../components/ui/dialog";
import { categories, categoryById } from "../data/catalog";
import type { CollectionType, NewProjectInput, Project } from "../lib/types";
import { cn, formatNumber } from "../lib/utils";

export function ScientistLoginPage() {
  const navigate = useNavigate();
  const { state, authReady, signInScientist } = useStore();
  const [requestSent, setRequestSent] = useState(false);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (authReady && state.scientistAuthenticated && !submitting) {
      navigate(state.scientistTermsAccepted ? "/scientist/dashboard" : "/scientist/terms", { replace: true });
    }
  }, [authReady, navigate, state.scientistAuthenticated, state.scientistTermsAccepted, submitting]);

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const username = String(form.get("username") ?? "").trim();
    const password = String(form.get("password") ?? "");
    setSubmitting(true);
    const authError = await signInScientist(username, password);
    if (authError !== null) {
      setSubmitting(false);
      setError(authError);
      return;
    }

    setError("");
    navigate("/scientist/terms");
  }

  return (
    <AuthShell title="LOG IN" step={1}>
      <form onSubmit={submit} className="mx-auto grid w-full max-w-2xl gap-5">
        <div><label className="mb-2 block font-semibold" htmlFor="scientist-username">Username</label><input id="scientist-username" name="username" required autoComplete="username" placeholder="scientist" /></div>
        <div><label className="mb-2 block font-semibold" htmlFor="scientist-password">Password</label><input id="scientist-password" name="password" type="password" required minLength={1} autoComplete="current-password" /></div>
        <div><label className="mb-2 block font-semibold" htmlFor="scientist-institution">Institution</label><input id="scientist-institution" required placeholder="Demo research institution" /></div>
        <p className="text-sm leading-6 text-[var(--muted)]">Researcher access is subject to institutional verification and governance review.</p>
        {import.meta.env.DEV && <div className="rounded-2xl bg-[var(--green-soft)] p-4 text-sm"><strong>Local demo account</strong><p className="mt-1 text-[var(--muted)]">Username: scientist · Password: diana-scientist</p></div>}
        {requestSent && <p className="rounded-2xl bg-[var(--green-soft)] p-4 text-sm" role="status">Demo access request recorded. No message was sent outside this prototype.</p>}
        {error !== "" && <p className="rounded-2xl bg-red-50 p-4 text-sm font-semibold text-[var(--error)]" role="alert">{error}</p>}
        <div className="flex flex-wrap items-center justify-between gap-4"><button type="button" onClick={() => setRequestSent(true)} className="min-h-11 rounded-full px-2 text-sm font-semibold underline underline-offset-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black">Request researcher access</button><Button type="submit" size="large" disabled={submitting}>{submitting ? "Signing in…" : "Continue"} <ArrowRight className="size-6" aria-hidden="true" /></Button></div>
      </form>
    </AuthShell>
  );
}

export function ScientistTermsPage() {
  const navigate = useNavigate();
  const { acceptScientistTerms } = useStore();

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    acceptScientistTerms();
    navigate("/scientist/dashboard");
  }

  return (
    <AuthShell title="Researcher terms and data-use responsibilities" step={2}>
      <form onSubmit={submit} className="mx-auto w-full max-w-3xl rounded-[32px] border border-black bg-white p-6 sm:p-9">
        <div className="rounded-3xl bg-[var(--green-soft)] p-6"><h2 className="text-2xl font-semibold">Responsible access</h2><ul className="mt-5 space-y-3 text-sm leading-6 text-[var(--muted)]"><li>DIANA provides de-identified research data only after approval.</li><li>Data may be used only for the approved purpose and retention period.</li><li>Participant consent does not replace ethics approval.</li><li>Researchers must not attempt re-identification.</li><li>Project requirements must be accurately described.</li><li>Dataset access depends on governance and permission checks.</li></ul></div>
        <fieldset className="mt-7 space-y-3"><legend className="mb-3 font-semibold">Required confirmations</legend><TermsCheckbox name="responsibility" label="I understand my data-use and non-re-identification responsibilities." /><TermsCheckbox name="purpose" label="I will use data only for the approved project purpose." /><TermsCheckbox name="governance" label="I understand that participant permission does not replace ethics or governance approval." /></fieldset>
        <div className="mt-8 flex justify-end"><Button type="submit" size="large">Agree and continue <ArrowRight className="size-5" aria-hidden="true" /></Button></div>
      </form>
    </AuthShell>
  );
}

function TermsCheckbox({ name, label }: { name: string; label: string }) {
  return <label className="flex min-h-14 items-start gap-3 rounded-2xl border border-black/20 p-4"><input className="mt-1" type="checkbox" name={name} required /><span className="text-sm leading-6">{label}</span></label>;
}

export function ScientistDashboardPage() {
  const navigate = useNavigate();
  const { state, authenticatedUsername, allProjects, selectScientistProject, createProject, signOut } = useStore();
  const selected = allProjects.find((project) => project.id === state.selectedScientistProjectId) ?? allProjects[0];
  const [projectModal, setProjectModal] = useState(false);
  const [mobileProjects, setMobileProjects] = useState(false);
  const [downloadInfo, setDownloadInfo] = useState(false);
  const [profileInfo, setProfileInfo] = useState(false);
  const [profileError, setProfileError] = useState("");

  if (selected === undefined) {
    return <main className="grid min-h-dvh place-items-center">No demo projects are available.</main>;
  }

  function selectProject(id: string): void {
    selectScientistProject(id);
    setMobileProjects(false);
  }

  function downloadDataset(): void {
    if (!selected.downloadReady) return;
    const rows = ["variable,participants,completeness_percent,content", ...selected.availability.map((item) => `${item.categoryId},${item.participants},${item.percent},synthetic_demo_aggregate`)];
    const url = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${selected.id}-synthetic-availability.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function logout(): Promise<void> {
    const authError = await signOut();
    if (authError !== null) {
      setProfileError(authError);
      return;
    }
    setProfileError("");
    navigate("/scientist/login", { replace: true });
  }

  return (
    <div className="min-h-dvh bg-[var(--warm-white)] lg:grid lg:grid-cols-[310px_1fr]">
      <aside className="hidden border-r border-black bg-white p-7 lg:flex lg:h-dvh lg:sticky lg:top-0 lg:flex-col">
        <ScientistSidebar projects={allProjects} selectedId={selected.id} onSelect={selectProject} onNew={() => setProjectModal(true)} />
      </aside>
      <main className="min-w-0 px-5 py-6 sm:px-8 lg:px-10">
        <div className="flex items-center justify-between gap-4 lg:justify-end"><Button variant="outline" className="lg:hidden" onClick={() => setMobileProjects(true)}><Menu className="size-5" aria-hidden="true" /> Projects</Button><button onClick={() => setProfileInfo(true)} className="grid size-11 place-items-center rounded-full border border-black bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black" aria-label="Researcher profile"><CircleUserRound className="size-5" aria-hidden="true" /></button></div>
        <section className="mt-7 rounded-full border border-black bg-white px-6 py-5 sm:px-8"><div className="flex flex-wrap items-center gap-3"><StatusPill tone={selected.downloadReady ? "green" : "warning"}>{selected.governanceStatus}</StatusPill>{selected.isCreated && <StatusPill tone="purple">Preliminary availability estimate</StatusPill>}</div><h1 className="mt-3 text-2xl font-semibold tracking-[-0.04em] sm:text-4xl">{selected.title}</h1></section>

        <div className="mt-8 grid gap-5 xl:grid-cols-[1fr_310px]">
          <section className="rounded-[30px] border border-black bg-white p-6 sm:p-8"><div className="flex flex-wrap items-end justify-between gap-4"><div><p className="text-sm font-semibold uppercase tracking-[0.15em] text-[var(--muted)]">Prototype estimate</p><h2 className="mt-2 text-3xl font-semibold tracking-[-0.04em]">Data availability</h2></div><span className="text-sm text-[var(--muted)]">Aggregate, non-identifying values</span></div><div className="mt-5">{selected.availability.map((item) => <AvailabilityBar key={item.categoryId} label={categoryById(item.categoryId)?.shortTitle ?? item.categoryId} percent={item.percent} participants={item.participants} state={item.state} />)}</div></section>
          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-1"><Metric label="Data points collected" value={formatNumber(selected.dataPoints)} tone="purple" /><Metric label="Population participated" value={formatNumber(selected.matchingParticipants)} tone="green" /><section className="rounded-[28px] border border-black bg-white p-6 sm:col-span-2 xl:col-span-1"><h2 className="text-lg font-semibold">Missing variables</h2>{selected.missingVariables.length === 0 ? <p className="mt-3 text-sm text-[var(--muted)]">No missing variables in the current request.</p> : <ul className="mt-3 space-y-2 text-sm text-[var(--muted)]">{selected.missingVariables.map((item) => <li key={item} className="flex gap-2"><XCircle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />{item}</li>)}</ul>}</section></div>
        </div>

        <section className="mt-5 grid gap-5 rounded-[30px] border border-black bg-[var(--neutral)] p-6 sm:p-8 xl:grid-cols-[1fr_auto] xl:items-end"><div><h2 className="text-2xl font-semibold tracking-[-0.03em]">Governance and download status</h2><p className="mt-3 max-w-3xl leading-7 text-[var(--muted)]">{selected.downloadReady ? "Approval, project permission, and synthetic dataset preparation checks are complete for this demo project." : `Dataset access is unavailable while this project has the status: ${selected.governanceStatus}.`}</p><PrivacyNote /></div><div className="flex flex-wrap gap-3">{selected.downloadReady ? <Button size="large" onClick={downloadDataset}><Download className="size-5" aria-hidden="true" /> Download data</Button> : <><Button size="large" disabled><ShieldAlert className="size-5" aria-hidden="true" /> Download data</Button><Button variant="outline" onClick={() => setDownloadInfo(true)}>Why unavailable?</Button></>}</div></section>
      </main>

      <Dialog open={mobileProjects} onOpenChange={setMobileProjects}><DialogContent className="max-w-md"><DialogHeader><DialogTitle>Current projects</DialogTitle><DialogDescription>Select a project or create a new feasibility request.</DialogDescription></DialogHeader><ScientistSidebar projects={allProjects} selectedId={selected.id} onSelect={selectProject} onNew={() => { setMobileProjects(false); setProjectModal(true); }} hideLogo /></DialogContent></Dialog>
      <NewProjectDialog open={projectModal} onOpenChange={setProjectModal} onCreate={createProject} />
      <Dialog open={downloadInfo} onOpenChange={setDownloadInfo}><DialogContent><DialogHeader><DialogTitle>Dataset access is not available</DialogTitle><DialogDescription>Download is enabled only after all approval and participant-permission checks are complete.</DialogDescription></DialogHeader><div className="rounded-3xl bg-[var(--green-soft)] p-6"><h3 className="font-semibold">Current status</h3><p className="mt-2 text-[var(--muted)]">{selected.governanceStatus}</p><p className="mt-4 text-sm leading-6 text-[var(--muted)]">This prototype distinguishes preliminary availability from actual approved dataset access. It never exposes identifiable participant records.</p></div></DialogContent></Dialog>
      <Dialog open={profileInfo} onOpenChange={setProfileInfo}><DialogContent><DialogHeader><DialogTitle>Researcher profile</DialogTitle><DialogDescription>Authenticated prototype researcher account.</DialogDescription></DialogHeader><div className="rounded-3xl bg-[var(--neutral)] p-6"><p className="font-semibold">{authenticatedUsername ?? "Researcher"}</p><p className="mt-1 text-sm text-[var(--muted)]">Your demo institution · Researcher session active</p></div>{profileError !== "" && <p className="mt-4 rounded-2xl bg-red-50 p-4 text-sm font-semibold text-[var(--error)]" role="alert">{profileError}</p>}<div className="mt-6 flex flex-wrap items-center justify-between gap-4"><Link to="/" className="font-semibold underline underline-offset-4">Return to public site</Link><Button variant="outline" onClick={() => void logout()}>Sign out</Button></div></DialogContent></Dialog>
    </div>
  );
}

function ScientistSidebar({ projects, selectedId, onSelect, onNew, hideLogo = false }: { projects: Project[]; selectedId: string; onSelect: (id: string) => void; onNew: () => void; hideLogo?: boolean }) {
  return <>{!hideLogo && <Logo />}<h2 className={cn("text-sm font-semibold uppercase tracking-[0.16em] text-[var(--muted)]", hideLogo ? "mb-4" : "mb-4 mt-12")}>Current projects</h2><nav className="space-y-2 overflow-y-auto" aria-label="Scientist projects">{projects.map((project) => <button key={project.id} onClick={() => onSelect(project.id)} className={cn("w-full rounded-2xl border p-4 text-left text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black", selectedId === project.id ? "border-black bg-[var(--green-soft)]" : "border-transparent hover:border-black/20 hover:bg-[var(--neutral)]")}><span className="line-clamp-2">{project.title}</span>{project.isCreated && <span className="mt-1 block text-xs font-normal text-[var(--muted)]">Preliminary estimate</span>}</button>)}</nav><Button variant="purple" className="mt-5 w-full shrink-0" onClick={onNew}><Plus className="size-5" aria-hidden="true" /> New project</Button><Link to="/" className={cn("pt-6 text-sm font-semibold underline underline-offset-4", !hideLogo && "mt-auto")}>View public site</Link></>;
}

function Metric({ label, value, tone }: { label: string; value: string; tone: "purple" | "green" }) {
  return <section className={cn("rounded-[28px] border border-black p-6", tone === "purple" ? "bg-[var(--purple-soft)]" : "bg-[var(--green-soft)]")}><p className="text-sm font-semibold">{label}</p><strong className="mt-6 block text-4xl tracking-[-0.05em]">{value}</strong><span className="mt-2 block text-xs text-[var(--muted)]">Prototype aggregate</span></section>;
}

function NewProjectDialog({ open, onOpenChange, onCreate }: { open: boolean; onOpenChange: (open: boolean) => void; onCreate: (input: NewProjectInput) => Project }) {
  const formRef = useRef<HTMLFormElement>(null);
  const [error, setError] = useState("");

  function save(status: "draft" | "pending"): void {
    const form = formRef.current;
    if (form === null) return;
    if (status === "pending" && !form.reportValidity()) return;
    const data = new FormData(form);
    const selectedCategories = categories.filter((category) => data.get(`category-${category.id}`) === "on").map((category) => category.id);
    const targetParticipants = Number(data.get("targetParticipants"));
    const collectionTypeValue = String(data.get("collectionType"));
    const collectionType: CollectionType = collectionTypeValue === "Retrospective" || collectionTypeValue === "Prospective" ? collectionTypeValue : "Retrospective and prospective";
    if (status === "pending" && (selectedCategories.length === 0 || !Number.isFinite(targetParticipants) || targetParticipants < 1)) {
      setError("Choose at least one desired data category and provide a valid target participant number.");
      return;
    }
    onCreate({
      title: String(data.get("title")).trim() || "Untitled project",
      abstract: String(data.get("abstract")).trim() || "Draft project requirements.",
      population: String(data.get("population")).trim() || "Not specified",
      followUp: String(data.get("followUp")).trim() || "Not specified",
      targetParticipants: Number.isFinite(targetParticipants) && targetParticipants > 0 ? targetParticipants : 0,
      collectionType,
      ethicsStatus: String(data.get("ethicsStatus")),
      categories: selectedCategories,
      status,
    });
    form.reset();
    setError("");
    onOpenChange(false);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl">
        <DialogHeader><DialogTitle>New project</DialogTitle><DialogDescription>Define the essential requirements for a preliminary, synthetic availability estimate.</DialogDescription></DialogHeader>
        <form ref={formRef} className="grid gap-5 sm:grid-cols-2" onSubmit={(event) => { event.preventDefault(); save("pending"); }}>
          <div className="sm:col-span-2"><label className="mb-2 block font-semibold" htmlFor="project-name">Project name</label><input id="project-name" name="title" required /></div>
          <div className="sm:col-span-2"><label className="mb-2 block font-semibold" htmlFor="project-abstract">Abstract</label><textarea id="project-abstract" name="abstract" rows={4} required /></div>
          <fieldset className="sm:col-span-2"><legend className="mb-2 font-semibold">Desired data</legend><div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">{categories.map((category) => <label key={category.id} className="flex min-h-14 items-center gap-3 rounded-2xl border border-black/20 p-3 text-sm"><input type="checkbox" name={`category-${category.id}`} />{category.shortTitle}</label>)}</div></fieldset>
          <div className="sm:col-span-2"><label className="mb-2 block font-semibold" htmlFor="desired-population">Desired population</label><input id="desired-population" name="population" required placeholder="Broad, non-identifying inclusion description" /></div>
          <div><label className="mb-2 block font-semibold" htmlFor="follow-up">Minimum follow-up period</label><input id="follow-up" name="followUp" required placeholder="For example, 90 days" /></div>
          <div><label className="mb-2 block font-semibold" htmlFor="collection-type">Data timing</label><select id="collection-type" name="collectionType" defaultValue="Retrospective and prospective"><option>Retrospective</option><option>Prospective</option><option>Retrospective and prospective</option></select></div>
          <div><label className="mb-2 block font-semibold" htmlFor="target-participants">Target participant number</label><input id="target-participants" name="targetParticipants" type="number" min="1" required defaultValue="250" /></div>
          <div><label className="mb-2 block font-semibold" htmlFor="ethics-status">Ethics approval status</label><select id="ethics-status" name="ethicsStatus" defaultValue="Not yet submitted"><option>Not yet submitted</option><option>Submitted for review</option><option>Approved</option></select></div>
        </form>
        {error !== "" && <p className="mt-5 rounded-2xl bg-red-50 p-4 text-sm font-semibold text-[var(--error)]" role="alert">{error}</p>}
        <div className="mt-7 flex flex-wrap justify-end gap-3"><Button variant="outline" onClick={() => save("draft")}><FileCheck2 className="size-4" aria-hidden="true" /> Save draft</Button><Button onClick={() => save("pending")}><FlaskConical className="size-4" aria-hidden="true" /> Create project</Button></div>
      </DialogContent>
    </Dialog>
  );
}

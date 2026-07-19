import { ArrowDown, ArrowRight, BarChart3, Database, FileText, GitBranch, Layers, LockKeyhole, ShieldCheck, UploadCloud, type LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";

import { StandardHeader, StatusPill } from "../components/shared";
import { buttonVariants } from "../components/ui/button";
import { cn } from "../lib/utils";

type FlowTone = "purple" | "green" | "neutral" | "gradient";

type FlowNode = {
  id: string;
  icon: LucideIcon;
  tone: FlowTone;
  title: string;
  eyebrow: string;
  description: string;
};

const toneClasses: Record<FlowTone, string> = {
  purple: "bg-[var(--purple-soft)]",
  green: "bg-[var(--green-soft)]",
  neutral: "bg-white",
  gradient: "bg-gradient-to-br from-[var(--purple-soft)] to-[var(--green-soft)]",
};

const coreNodes: FlowNode[] = [
  {
    id: "licensed-data",
    icon: Database,
    tone: "purple",
    title: "Licensed mcPHASES data",
    eyebrow: "Governed input",
    description: "Restricted source rows are obtained under the data-use agreement and are never published from this repository.",
  },
  {
    id: "hormonbench-adapter",
    icon: GitBranch,
    tone: "green",
    title: "Hormonbench v1 adapter",
    eyebrow: "Frozen task",
    description: "The adapter fixes the causal window: approved wearable summaries from t-13 through t predict observed urinary LH, E3G, and PdG at t+1.",
  },
  {
    id: "private-bundle",
    icon: LockKeyhole,
    tone: "gradient",
    title: "Private prepared bundle",
    eyebrow: "Protected artifacts",
    description: "Rows, truth, participant IDs, fold mappings, and calibration views stay under artifacts/private/v1/.",
  },
];

const modelNodes: FlowNode[] = [
  {
    id: "feature-views",
    icon: Layers,
    tone: "neutral",
    title: "Feature-only views",
    eyebrow: "Model boundary",
    description: "Models receive approved features and authorized K-label calibration views, not participant IDs or evaluation truth.",
  },
  {
    id: "classical-baselines",
    icon: UploadCloud,
    tone: "purple",
    title: "Classical baselines",
    eyebrow: "Reference experts",
    description: "Population median, wearable Ridge, and CatBoost produce fold-specific prediction CSV files.",
  },
  {
    id: "h3p-layer-one",
    icon: GitBranch,
    tone: "green",
    title: "Diana-H3P Layer 1",
    eyebrow: "Wearable prior",
    description: "A participant-balanced convex stack combines the three baseline experts using development out-of-fold predictions.",
  },
  {
    id: "h3p-layer-two",
    icon: ShieldCheck,
    tone: "gradient",
    title: "Diana-H3P Layer 2",
    eyebrow: "K=0/3/7 personalization",
    description: "Authorized early hormone readings update a joint three-hormone residual model and produce research intervals.",
  },
];

const benchmarkNodes: FlowNode[] = [
  {
    id: "private-truth",
    icon: LockKeyhole,
    tone: "neutral",
    title: "Private truth and folds",
    eyebrow: "Evaluator-only",
    description: "The benchmark keeps held-out truth, folds, sample IDs, and participant metrics separate from model code.",
  },
  {
    id: "manifest",
    icon: FileText,
    tone: "purple",
    title: "Prediction manifest",
    eyebrow: "Explicit files",
    description: "Every prediction file is listed with task identity, fold, track, budget, model name, and SHA-256 hash.",
  },
  {
    id: "evaluator",
    icon: BarChart3,
    tone: "green",
    title: "Model-independent evaluator",
    eyebrow: "Private scoring",
    description: "benchmark/v1_evaluator.py validates schemas, joins private truth internally, and computes participant-macro metrics.",
  },
  {
    id: "privacy-checks",
    icon: ShieldCheck,
    tone: "gradient",
    title: "Aggregate public release",
    eyebrow: "Privacy gate",
    description: "Only aggregate results and release-safe documentation leave the private boundary.",
  },
];

function FlowCard({ item, compact = false }: { item: FlowNode; compact?: boolean }) {
  const Icon = item.icon;

  return (
    <article className={cn("rounded-[28px] border border-black p-5", toneClasses[item.tone], compact ? "min-h-40" : "min-h-44")}>
      <div className="flex items-start justify-between gap-4">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">{item.eyebrow}</p>
        <span className="grid size-11 shrink-0 place-items-center rounded-full border border-black bg-white">
          <Icon className="size-5 stroke-[1.6]" aria-hidden="true" />
        </span>
      </div>
      <h2 className="mt-5 text-2xl font-semibold tracking-[-0.04em]">{item.title}</h2>
      <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{item.description}</p>
    </article>
  );
}

function DownConnector() {
  return (
    <div className="flex justify-center py-3" aria-hidden="true">
      <span className="grid size-9 place-items-center rounded-full border border-black bg-white">
        <ArrowDown className="size-4" />
      </span>
    </div>
  );
}

function BranchPanel({ title, nodes }: { title: string; nodes: FlowNode[] }) {
  return (
    <section className="rounded-[32px] border border-black bg-white/80 p-4 sm:p-5">
      <h2 className="mb-4 rounded-full border border-black bg-[var(--neutral)] px-4 py-2 text-sm font-semibold">{title}</h2>
      <div>
        {nodes.map((item, index) => (
          <div key={item.id}>
            <FlowCard item={item} compact />
            {index < nodes.length - 1 && <DownConnector />}
          </div>
        ))}
      </div>
    </section>
  );
}

export function TreePage() {
  return (
    <div>
      <StandardHeader />
      <main>
        <section className="soft-field border-b border-black px-5 py-16 sm:px-10 sm:py-24">
          <div className="mx-auto grid max-w-7xl gap-12 lg:grid-cols-[0.8fr_1.2fr] lg:items-center">
            <div>
              <StatusPill tone="purple">Technical tree</StatusPill>
              <h1 className="mt-6 text-5xl font-semibold tracking-[-0.07em] sm:text-7xl">
                How the benchmark and model <span className="serif-accent">work together</span>
              </h1>
              <p className="mt-7 max-w-2xl text-lg leading-8 text-[var(--muted)]">
                Hormonbench owns the task, folds, schemas, private-truth scoring, and privacy checks. Models consume only approved views and return prediction files that the evaluator can score without importing model code.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link to="/projects" className={buttonVariants({ variant: "green", size: "large" })}>View projects <ArrowRight className="size-5" aria-hidden="true" /></Link>
                <Link to="/" className={buttonVariants({ variant: "outline", size: "large" })}>Back home</Link>
              </div>
            </div>

            <div className="rounded-[42px] border border-black bg-white p-4 shadow-[10px_10px_0_0_rgba(10,10,10,0.12)] sm:p-6">
              <div className="rounded-[34px] border border-black/10 bg-gradient-to-br from-white via-[var(--purple-soft)] to-[var(--green-soft)] p-4 sm:p-6">
                <div className="grid gap-4">
                  {coreNodes.map((item, index) => (
                    <div key={item.id}>
                      <FlowCard item={item} />
                      {index < coreNodes.length - 1 && <DownConnector />}
                    </div>
                  ))}
                </div>
                <DownConnector />
                <div className="grid gap-5 lg:grid-cols-2">
                  <BranchPanel title="Model path: model/" nodes={modelNodes} />
                  <BranchPanel title="Benchmark path: benchmark/" nodes={benchmarkNodes} />
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="border-b border-black bg-white px-5 py-18 sm:px-10 sm:py-24">
          <div className="mx-auto grid max-w-6xl gap-6 md:grid-cols-3">
            {[
              { title: "No leakage", text: "Participant IDs, future hormone values, stale calendar features, and evaluation truth are denied as predictors." },
              { title: "Participant macro", text: "Metrics average within each participant first so longer records do not dominate the reported score." },
              { title: "Aggregate only", text: "Private rows, predictions, calibration mappings, and participant-level metrics stay out of public release artifacts." },
            ].map((item) => (
              <article key={item.title} className="rounded-[30px] border border-black bg-[var(--neutral)] p-7">
                <h2 className="text-2xl font-semibold tracking-[-0.04em]">{item.title}</h2>
                <p className="mt-4 leading-7 text-[var(--muted)]">{item.text}</p>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

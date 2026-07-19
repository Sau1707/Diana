import { ArrowDown, BarChart3, Database, FileText, GitBranch, Layers, LockKeyhole, ShieldCheck, type LucideIcon } from "lucide-react";

import { StandardHeader, StatusPill } from "../components/shared";
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
    title: "Licensed data",
    eyebrow: "Input",
    description: "mcPHASES rows stay governed and private.",
  },
  {
    id: "hormonbench-adapter",
    icon: GitBranch,
    tone: "green",
    title: "Hormonbench task",
    eyebrow: "Adapter",
    description: "t-13...t wearable summaries predict urinary hormones at t+1.",
  },
  {
    id: "private-bundle",
    icon: LockKeyhole,
    tone: "gradient",
    title: "Private prepared bundle",
    eyebrow: "Protected",
    description: "Rows, truth, folds, IDs, and calibration views stay under artifacts/private/v1/.",
  },
];

const modelNodes: FlowNode[] = [
  {
    id: "feature-views",
    icon: Layers,
    tone: "neutral",
    title: "Feature-only views",
    eyebrow: "Boundary",
    description: "Models see approved features, not private truth.",
  },
  {
    id: "classical-baselines",
    icon: BarChart3,
    tone: "purple",
    title: "Classical baselines",
    eyebrow: "Experts",
    description: "Population median, Ridge, and CatBoost make predictions.",
  },
  {
    id: "h3p-layer-one",
    icon: GitBranch,
    tone: "green",
    title: "Diana-H3P Layer 1",
    eyebrow: "Stack",
    description: "The baseline predictions become one wearable prior.",
  },
  {
    id: "h3p-layer-two",
    icon: ShieldCheck,
    tone: "gradient",
    title: "Diana-H3P Layer 2",
    eyebrow: "Personalize",
    description: "K=0/3/7 readings adjust predictions and intervals.",
  },
];

const benchmarkNodes: FlowNode[] = [
  {
    id: "private-truth",
    icon: LockKeyhole,
    tone: "neutral",
    title: "Private truth and folds",
    eyebrow: "Evaluator-only",
    description: "Held-out truth stays separate from model code.",
  },
  {
    id: "manifest",
    icon: FileText,
    tone: "purple",
    title: "Prediction manifest",
    eyebrow: "Files",
    description: "Prediction CSVs are listed with SHA-256 hashes.",
  },
  {
    id: "evaluator",
    icon: BarChart3,
    tone: "green",
    title: "Model-independent evaluator",
    eyebrow: "Scoring",
    description: "The evaluator joins truth privately and computes metrics.",
  },
  {
    id: "privacy-checks",
    icon: ShieldCheck,
    tone: "gradient",
    title: "Aggregate results",
    eyebrow: "Public",
    description: "Only release-safe summaries are published.",
  },
];

function FlowCard({ item, compact = false }: { item: FlowNode; compact?: boolean }) {
  const Icon = item.icon;

  return (
    <article className={cn("rounded-[24px] border border-black p-5", toneClasses[item.tone], compact ? "min-h-32" : "min-h-36")}>
      <div className="flex items-start justify-between gap-4">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">{item.eyebrow}</p>
        <span className="grid size-11 shrink-0 place-items-center rounded-full border border-black bg-white">
          <Icon className="size-5 stroke-[1.6]" aria-hidden="true" />
        </span>
      </div>
      <h2 className="mt-4 text-xl font-semibold tracking-[-0.04em] sm:text-2xl">{item.title}</h2>
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
          <div className="mx-auto max-w-6xl">
            <div className="mx-auto mb-10 max-w-3xl text-center">
              <StatusPill tone="purple">Technical tree</StatusPill>
              <h1 className="mt-5 text-4xl font-semibold tracking-[-0.06em] sm:text-6xl">
                Benchmark <span className="serif-accent">to</span> model flow
              </h1>
            </div>

            <div className="rounded-[38px] border border-black bg-white p-4 shadow-[8px_8px_0_0_rgba(10,10,10,0.12)] sm:p-6">
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
      </main>
    </div>
  );
}

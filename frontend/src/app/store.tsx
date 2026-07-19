import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { z } from "zod";

import { projects } from "../data/catalog";
import { projectWithContribution } from "../lib/matching";
import type {
  ConsentRecord,
  ConsentOptions,
  ContributionIntent,
  NewProjectInput,
  Project,
  PrototypeState,
} from "../lib/types";

const STORAGE_KEY = "diana.prototype.v1";
const categorySchema = z.enum(["cycle", "symptoms", "sleep", "activity", "temperature", "hormones", "medical", "demographics"]);
const collectionTypeSchema = z.enum(["Retrospective", "Prospective", "Retrospective and prospective"]);
const availabilitySchema = z.object({
  categoryId: categorySchema,
  percent: z.number().min(0).max(100),
  participants: z.number().int().nonnegative(),
  state: z.enum(["Strong coverage", "Partial coverage", "Limited coverage"]),
});
const projectSchema = z.object({
  id: z.string(),
  title: z.string(),
  institution: z.string(),
  summary: z.string(),
  abstract: z.string(),
  researchQuestion: z.string(),
  intendedUse: z.string(),
  studyPeriod: z.string(),
  ethicsStatus: z.string(),
  targetPopulation: z.string(),
  minimumFollowUp: z.string(),
  collectionType: collectionTypeSchema,
  categories: z.array(categorySchema),
  requestedVariables: z.array(z.string()),
  availability: z.array(availabilitySchema),
  matchingParticipants: z.number().int().nonnegative(),
  dataPoints: z.number().int().nonnegative(),
  missingVariables: z.array(z.string()),
  governanceStatus: z.string(),
  status: z.enum(["approved", "pending", "draft"]),
  isPublic: z.boolean(),
  downloadReady: z.boolean(),
  isCreated: z.boolean().optional(),
});

const stateSchema = z.object({
  participantAuthenticated: z.boolean(),
  scientistAuthenticated: z.boolean(),
  scientistTermsAccepted: z.boolean(),
  intent: z
    .object({
      kind: z.enum(["general", "project"]),
      projectId: z.string().optional(),
    })
    .nullable(),
  consents: z.array(z.object({
    id: z.string(),
    categoryIds: z.array(categorySchema),
    projectId: z.string().nullable(),
    scope: z.enum(["project", "approved-projects"]),
    signedName: z.string(),
    signedOn: z.string(),
    status: z.enum(["active", "withdrawn"]),
    options: z.object({
      similarProjects: z.boolean(),
      futureNotifications: z.boolean(),
      retrospectiveData: z.boolean(),
      prospectiveCollection: z.boolean(),
    }),
  })),
  contributedCategoryIds: z.array(categorySchema),
  createdProjects: z.array(projectSchema),
  selectedScientistProjectId: z.string(),
});

const initialState: PrototypeState = {
  participantAuthenticated: false,
  scientistAuthenticated: false,
  scientistTermsAccepted: false,
  intent: null,
  consents: [],
  contributedCategoryIds: [],
  createdProjects: [],
  selectedScientistProjectId: projects[0].id,
};

interface StoreValue {
  state: PrototypeState;
  allProjects: Project[];
  publicProjects: Project[];
  signInParticipant: () => void;
  signInScientist: () => void;
  acceptScientistTerms: () => void;
  setIntent: (intent: ContributionIntent | null) => void;
  recordConsent: (consent: Omit<ConsentRecord, "id" | "status">) => ConsentRecord;
  updateConsent: (id: string, scope: ConsentRecord["scope"], projectId: string | null, options: ConsentOptions) => void;
  withdrawConsent: (id: string) => void;
  selectScientistProject: (id: string) => void;
  createProject: (input: NewProjectInput) => Project;
}

const StoreContext = createContext<StoreValue | null>(null);

function restoreState(): PrototypeState {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === null) {
      return initialState;
    }
    const parsed: unknown = JSON.parse(saved);
    const result = stateSchema.safeParse(parsed);
    return result.success ? result.data : initialState;
  } catch {
    return initialState;
  }
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PrototypeState>(restoreState);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // The prototype remains usable when private browsing blocks persistence.
    }
  }, [state]);

  const baseProjects = [...projects, ...state.createdProjects];
  const allProjects = baseProjects.map((project) => projectWithContribution(project, state.consents, baseProjects));
  const publicProjects = allProjects.filter((project) => project.isPublic && project.status === "approved");

  function signInParticipant(): void {
    setState((current) => ({ ...current, participantAuthenticated: true }));
  }

  function signInScientist(): void {
    setState((current) => ({ ...current, scientistAuthenticated: true }));
  }

  function acceptScientistTerms(): void {
    setState((current) => ({ ...current, scientistTermsAccepted: true }));
  }

  function setIntent(intent: ContributionIntent | null): void {
    setState((current) => ({ ...current, intent }));
  }

  function recordConsent(input: Omit<ConsentRecord, "id" | "status">): ConsentRecord {
    const consent: ConsentRecord = {
      ...input,
      id: crypto.randomUUID(),
      status: "active",
    };

    setState((current) => ({
      ...current,
      consents: [...current.consents, consent],
      contributedCategoryIds: [...new Set([...current.contributedCategoryIds, ...consent.categoryIds])],
    }));
    return consent;
  }

  function updateConsent(id: string, scope: ConsentRecord["scope"], projectId: string | null, options: ConsentOptions): void {
    setState((current) => ({
      ...current,
      consents: current.consents.map((consent) =>
        consent.id === id
          ? { ...consent, scope, projectId, options }
          : consent,
      ),
    }));
  }

  function withdrawConsent(id: string): void {
    setState((current) => ({
      ...current,
      consents: current.consents.map((consent) =>
        consent.id === id ? { ...consent, status: "withdrawn" } : consent,
      ),
    }));
  }

  function selectScientistProject(id: string): void {
    setState((current) => ({ ...current, selectedScientistProjectId: id }));
  }

  function createProject(input: NewProjectInput): Project {
    const followUpAmount = Number.parseInt(input.followUp, 10);
    const normalizedFollowUp = input.followUp.toLowerCase();
    const followUpDays = Number.isFinite(followUpAmount)
      ? normalizedFollowUp.includes("year")
        ? followUpAmount * 365
        : normalizedFollowUp.includes("month")
          ? followUpAmount * 30
          : followUpAmount
      : 120;
    const followUpFactor = Math.min(1, 90 / Math.max(followUpDays, 1));
    const collectionFactor = input.collectionType === "Retrospective" ? 1 : input.collectionType === "Prospective" ? 0.65 : 0.82;
    const availability = input.categories.map((categoryId) => {
      const observations = projects.flatMap((project) => project.availability).filter((item) => item.categoryId === categoryId);
      const baselinePercent = observations.length === 0 ? 0 : observations.reduce((total, item) => total + item.percent, 0) / observations.length;
      const baselineParticipants = observations.length === 0 ? 0 : observations.reduce((total, item) => total + item.participants, 0) / observations.length;
      const percent = Math.round(baselinePercent * followUpFactor);
      const participants = Math.round(baselineParticipants * followUpFactor * collectionFactor);
      return {
        categoryId,
        percent,
        participants,
        state: percent >= 70 ? "Strong coverage" as const : percent >= 45 ? "Partial coverage" as const : "Limited coverage" as const,
      };
    });
    const matchingParticipants = availability.length === 0 ? 0 : Math.min(...availability.map((item) => item.participants));
    const missingVariables: string[] = availability.filter((item) => item.percent < 50).map((item) => item.categoryId);
    missingVariables.push("Population criteria require structured governance review");
    if (input.targetParticipants > matchingParticipants) {
      missingVariables.push(`Target shortfall: ${input.targetParticipants - matchingParticipants} participants`);
    }
    const project: Project = {
      id: `project-${crypto.randomUUID()}`,
      title: input.title,
      institution: "Your demo institution",
      summary: input.abstract.slice(0, 120),
      abstract: input.abstract,
      researchQuestion: "Defined in the submitted project abstract.",
      intendedUse: "Feasibility assessment for the described approved research purpose.",
      studyPeriod: "To be confirmed after governance review",
      ethicsStatus: input.ethicsStatus,
      targetPopulation: input.population,
      minimumFollowUp: input.followUp,
      collectionType: input.collectionType,
      categories: input.categories,
      requestedVariables: input.categories,
      availability,
      matchingParticipants,
      dataPoints: availability.reduce((total, item) => total + item.participants * 24, 0),
      missingVariables,
      governanceStatus: input.status === "draft" ? "Draft requirements" : "Preliminary governance review",
      status: input.status,
      isPublic: false,
      downloadReady: false,
      isCreated: true,
    };

    setState((current) => ({
      ...current,
      createdProjects: [...current.createdProjects, project],
      selectedScientistProjectId: project.id,
    }));
    return project;
  }

  return (
    <StoreContext.Provider
      value={{
        state,
        allProjects,
        publicProjects,
        signInParticipant,
        signInScientist,
        acceptScientistTerms,
        setIntent,
        recordConsent,
        updateConsent,
        withdrawConsent,
        selectScientistProject,
        createProject,
      }}
    >
      {children}
    </StoreContext.Provider>
  );
}

// The hook intentionally shares the provider module so its context type remains local.
// eslint-disable-next-line react-refresh/only-export-components
export function useStore(): StoreValue {
  const context = useContext(StoreContext);
  if (context === null) {
    throw new Error("useStore must be used within StoreProvider");
  }
  return context;
}

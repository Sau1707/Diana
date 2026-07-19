export type DataCategoryId =
  | "cycle"
  | "symptoms"
  | "sleep"
  | "activity"
  | "temperature"
  | "hormones"
  | "medical"
  | "demographics";

export type ProjectStatus = "approved" | "pending" | "draft";
export type CollectionType = "Retrospective" | "Prospective" | "Retrospective and prospective";
export type AvailabilityState = "Strong coverage" | "Partial coverage" | "Limited coverage";

export interface DataCategory {
  id: DataCategoryId;
  title: string;
  shortTitle: string;
  description: string;
  examples: string[];
  method: string;
}

export interface Availability {
  categoryId: DataCategoryId;
  percent: number;
  participants: number;
  state: AvailabilityState;
}

export interface Project {
  id: string;
  title: string;
  institution: string;
  summary: string;
  abstract: string;
  researchQuestion: string;
  intendedUse: string;
  studyPeriod: string;
  ethicsStatus: string;
  targetPopulation: string;
  minimumFollowUp: string;
  collectionType: CollectionType;
  categories: DataCategoryId[];
  requestedVariables: string[];
  availability: Availability[];
  matchingParticipants: number;
  dataPoints: number;
  missingVariables: string[];
  governanceStatus: string;
  status: ProjectStatus;
  isPublic: boolean;
  downloadReady: boolean;
  isCreated?: boolean;
}

export interface ConsentOptions {
  similarProjects: boolean;
  futureNotifications: boolean;
  retrospectiveData: boolean;
  prospectiveCollection: boolean;
}

export interface ConsentRecord {
  id: string;
  categoryIds: DataCategoryId[];
  projectId: string | null;
  scope: "project" | "approved-projects";
  signedName: string;
  signedOn: string;
  status: "active" | "withdrawn";
  options: ConsentOptions;
}

export interface ContributionIntent {
  kind: "general" | "project";
  projectId?: string;
}

export interface PrototypeState {
  participantAuthenticated: boolean;
  scientistAuthenticated: boolean;
  scientistTermsAccepted: boolean;
  intent: ContributionIntent | null;
  consents: ConsentRecord[];
  contributedCategoryIds: DataCategoryId[];
  createdProjects: Project[];
  selectedScientistProjectId: string;
}

export interface NewProjectInput {
  title: string;
  abstract: string;
  population: string;
  followUp: string;
  targetParticipants: number;
  collectionType: CollectionType;
  ethicsStatus: string;
  categories: DataCategoryId[];
  status: "draft" | "pending";
}

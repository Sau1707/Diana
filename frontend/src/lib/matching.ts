import type { ConsentRecord, DataCategoryId, Project } from "./types";

export function consentSupportsProject(consent: ConsentRecord, project: Project, projects: Project[]): boolean {
  if (consent.status !== "active") {
    return false;
  }

  const timingPermitted = project.collectionType === "Retrospective"
    ? consent.options.retrospectiveData
    : project.collectionType === "Prospective"
      ? consent.options.prospectiveCollection
      : consent.options.retrospectiveData || consent.options.prospectiveCollection;
  if (!timingPermitted) {
    return false;
  }

  if (consent.scope === "project") {
    if (consent.projectId === project.id) {
      return true;
    }

    const source = projects.find((item) => item.id === consent.projectId);
    const sharedRequirements = source?.categories.filter((id) => project.categories.includes(id)).length ?? 0;
    return consent.options.similarProjects
      && project.status === "approved"
      && source?.collectionType === project.collectionType
      && sharedRequirements >= 2;
  }

  return project.status === "approved";
}

export function matchingCategories(project: Project, consents: ConsentRecord[], projects: Project[]): DataCategoryId[] {
  const matches = consents
    .filter((consent) => consentSupportsProject(consent, project, projects))
    .flatMap((consent) => consent.categoryIds)
    .filter((categoryId) => project.categories.includes(categoryId));

  return [...new Set(matches)];
}

export function projectWithContribution(project: Project, consents: ConsentRecord[], projects: Project[]): Project {
  const matches = matchingCategories(project, consents, projects);
  if (matches.length === 0) {
    return project;
  }

  return {
    ...project,
    dataPoints: project.dataPoints + matches.length * 420,
    availability: project.availability.map((item) =>
      matches.includes(item.categoryId)
        ? { ...item, participants: item.participants + 1, percent: Math.min(item.percent + 1, 100) }
        : item,
    ),
  };
}

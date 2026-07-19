# Foundation Models for Women's Hormonal Health

## Challenge

Build an open, reusable AI building block for women's hormonal health. Focus on one specific hormone-related problem and contribute a dataset, benchmark, model, or application that future researchers can immediately use and extend.

The goal is not to build one large foundation model during the hackathon. The goal is to create one rigorous, reproducible layer of the infrastructure needed for future women's health research.

## Motivation

Female physiology remains underrepresented in AI and biomedical research despite women representing more than half of the global population. Hormones interact continuously with sleep, stress, nutrition, age, brain function, cardiovascular health, and metabolism, but clinical care often relies on occasional snapshots.

This contributes to several persistent problems:

- **No shared benchmark:** Researchers lack standardized multimodal datasets that combine hormone measurements, wearables, laboratory data, symptoms, and longitudinal physiological signals.
- **Fragmented infrastructure:** Data, models, and benchmarks are scattered across institutions, making methods difficult to compare and slowing cumulative research.
- **Dynamic biology, static care:** Clinical decisions rely on occasional laboratory tests rather than a continuous understanding of hormonal physiology.

Conditions such as PCOS, endometriosis, and menopause-related disease can take years to diagnose. Better shared infrastructure could improve health outcomes, access, understanding, and daily well-being for women worldwide.

## Build One Reusable Layer

Choose the layer where the team can make the strongest contribution.

### 1. Data and Benchmark Infrastructure

Curate, integrate, and publish data foundations that allow researchers to train, evaluate, and compare AI models for women's hormonal health.

Possible contributions include:

- A standardized multimodal dataset combining public wearables, laboratory measurements, imaging, symptoms, and longitudinal signals.
- Benchmark datasets with documented training, validation, and test splits and a transparent evaluation methodology.
- Focused prediction tasks such as hormone-level, ovulation, menopause-stage, or disease-risk prediction.

**Goal:** Leave behind an open dataset or benchmark that researchers can use immediately.

### 2. AI Model Infrastructure

Train a focused model that can contribute to future foundation models. Prioritize reproducibility, scientific rigor, and explainability over model size.

Possible contributions include:

- Hormone-level or hormonal-state prediction, including biological signatures of early or late menopause onset.
- Explainable AI that integrates multiple data sources and interprets findings in the context of hormonal variability.
- Prediction of clinical trials in which women's hormonal variability may affect efficacy or safety.

**Goal:** Build a reusable model that researchers can reproduce, extend, and combine into larger systems.

### 3. Application Infrastructure

Demonstrate how stronger AI infrastructure can improve women's lives. Build on reusable datasets and models to solve one clearly defined problem.

Possible contributions include:

- A regulator-conscious way for volunteers to contribute de-identified EHR or longitudinal health data to research.
- AI symptom tracking, digital hormone journals, or voice-based health logging.
- Personalized hormone insights or digital twins of female physiology that address gaps in access and representation.

**Goal:** Translate foundational AI into meaningful, measurable impact for women.

## Core Requirements

- Choose one clearly defined hormone-related problem.
- Demonstrate how the solution advances research and improves women's health.
- Use open datasets whenever possible.
- Comply with applicable privacy, ethical, consent, and licensing requirements.
- Document assumptions, preprocessing, data splits, and evaluation choices.
- Avoid unsupported medical or diagnostic claims.
- Optimize for reproducibility, scientific rigor, explainability, and responsible design.
- Publish datasets, benchmarks, source code, model checkpoints, documentation, and evaluation pipelines under an open license whenever possible.
- Produce a reusable scientific asset, not only a prototype or isolated interface.

## Suggested Data Sources

| Dataset | Description | Link |
| --- | --- | --- |
| mcPHASES (PhysioNet) | Fitbit, continuous glucose monitoring, hormone measurements, menstrual-cycle data, sleep, and symptoms | <https://physionet.org/content/mcphases> |
| NHANES (CDC) | Reproductive health, thyroid hormones, laboratory data, nutrition, and demographics | <https://wwwn.cdc.gov/nchs/nhanes> |

Teams may use other datasets, integrate multiple datasets, or responsibly collect new longitudinal data.

## OpenAI Resources

Hack-Nation provides each team with $50 in OpenAI API credits on a first-come, first-served basis.

Teams are encouraged to explore multimodal approaches that combine data across domains, including OpenAI multimodal models and `gpt-image-2` alongside modality-specific models. Strong solutions should show clear user value, technical quality, creativity, responsible design, and a convincing demonstration.

## Deliverables

- [ ] A working prototype and source code.
- [ ] Technical documentation and a dataset description.
- [ ] A documented benchmark methodology.
- [ ] A short demonstration video.
- [ ] Reproducible setup and usage instructions.
- [ ] Clearly stated data sources, licenses, assumptions, preprocessing, and evaluation choices.
- [ ] Open datasets, benchmarks, model checkpoints, and evaluation pipelines whenever possible.

## What Makes a Strong Submission

| Strong submissions | Weak submissions |
| --- | --- |
| Publish reusable datasets, benchmarks, model checkpoints, and evaluation pipelines under an open license. | Build an isolated application with no reusable dataset, benchmark, or model contribution. |
| Solve one clearly defined prediction or infrastructure problem exceptionally well using transparent methods and evidence. | Depend on undocumented proprietary data or hide assumptions, preprocessing, or evaluation choices. |
| Share reproducible code and documentation that the research community can extend. | Make unsupported medical or diagnostic claims or present a polished interface without scientific validation. |

## Success Criteria

1. **Women's health impact:** How significantly could the work improve women's health, and how many women could ultimately benefit?
2. **Technical excellence:** How innovative, rigorous, reproducible, and scalable is the dataset, benchmark, model, or application?
3. **Foundation value:** Does the project leave behind reusable infrastructure that accelerates future research?

Impact should be measured across two dimensions:

- **Reach:** How many women could ultimately benefit from the dataset, benchmark, model, or application?
- **Quality of life:** How meaningfully could the solution improve health outcomes, access, understanding, or daily well-being?

## Definition of Done

The project is complete when it solves one specific hormonal-health problem, includes a working and documented implementation, reports a transparent benchmark or evaluation, demonstrates measurable value for women, and leaves behind a reusable scientific asset that future researchers can build upon.

## Source

Hack-Nation Challenge 05, "Foundation Models for Women's Hormonal Health," in collaboration with the MIT Club of Northern California and the MIT Club of Germany, 6th Global AI Hackathon.

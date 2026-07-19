# Diana

DIANA is a consent-based women's hormonal health data infrastructure platform connecting participants who choose to contribute longitudinal, multimodal health data with researchers seeking structured, analysis-ready datasets. Participants control which data categories and research projects they support, while DIANA records consent, de-identifies and standardizes contributions, and helps researchers assess population fit, variable availability, completeness, and follow-up coverage before requesting approved access.

The project implements the challenge's reusable application-infrastructure layer through a responsive prototype covering public project discovery, participant contributions and consent management, and researcher project creation, feasibility assessment, and governance. It must remain open, reproducible, privacy-conscious, and scientifically responsible; clearly document data sources and assumptions; expose no identifiable participant data; avoid diagnosis or unsupported medical claims; and deliver reusable infrastructure rather than an isolated interface.

## Frontend Guidelines

Build the DIANA prototype by following the supplied black-and-white wireframes as the source of truth for page structure, navigation, and user journeys.

Keep the visual design system already defined:

- DIANA black, purple, and green palette;
- spacious editorial layout;
- soft purple-to-green gradients;
- thin black outlines;
- rounded cards and pill-shaped controls;
- premium but minimal dashboard design;
- simple transitions only;
- no complex animations;
- no stereotypical pink women’s-health design;
- no hospital-style interface.


## Python Guidelines

- Use safe, practical defaults that minimize required configuration.
- Validate inputs as early as possible, preferably at system boundaries.
- Use exceptions for genuine error conditions while avoiding unnecessary `try` and `except` blocks.
- Represent application state explicitly with typed models, enums, or structured objects.
- Use `Protocol` for behavioral interfaces and dependency contracts.
- Avoid `Any` and prefer precise type annotations.
- Keep logic in one function unless extraction clearly improves reuse, readability, or separation of concerns, and avoid single-use helpers unless they hide a genuinely complex boundary.
- Do not introduce private `_...` helper functions just to wrap a short local sequence, even when that sequence appears in two nearby call sites. Keep simple route/service flows inline unless extraction isolates a complex external boundary or creates reusable domain behavior.
- Simplify control flow, remove dead or duplicated code, and review the final implementation for further simplifications.
- Prefer concise local names when the surrounding scope already provides context; avoid repeating the domain in every variable name.
- Avoid redundant validation or normalization calls for persisted or already-derived values; validate once at the boundary unless the transformed value is used.
- Follow existing project conventions for naming, structure, formatting, and architecture.
- Prefer established, well-maintained libraries over handwritten implementations when they reduce complexity.
- Target Python 3+ and do not use the `__future__` module.
- Add a docstring to every Python function.
- Add a descriptive `# ...` comment before each logic block and leave one blank line before the comment.
- Keep a lookup and its immediate existence check in the same logic block; place the block comment before the lookup, not between the lookup and `if ... is None` check.
- Use two blank lines between function definitions and keep function signatures on one line when they fit within the configured line length.
- Do not start Python files with module-level triple-quoted docstrings unless the file is an Alembic revision.
- Do not add `__all__` unless the module has a concrete public star-import contract.
- Use clear domain names, prefer single-word Python filenames, and keep related model module names plural and consistent across API and database layers.
- Avoid renaming imports unless it materially improves clarity or consistency.
- Prefer namespaced module APIs, such as `adapters.database(...)`, over directly importing many related factory functions.
- Group Pydantic model fields into clearly commented sections and order fields from shortest name to longest name within each section.
- Declare `response_model` on FastAPI routes and return raw ORM objects, dictionaries, lists, or primitive values without manually instantiating or validating response models.
- Omit FastAPI route handler return annotations when the decorator already defines the response contract, such as routes with `response_model` or no-body `status_code=204` responses.
- Test the actual implementation rather than duplicating production logic, and do not add new test cases unless explicitly requested.
- Avoid mocks and global runtime-state modifications where practical, preferring real implementations and explicit dependency boundaries.
- Prefer simple, maintainable, conventional solutions over clever hacks.

## JavaScript / TypeScript Guidelines

- Validate inputs at system boundaries.
- Avoid any; prefer precise types, generics, unknown with narrowing, discriminated unions, and established validation libraries.
- Avoid unsafe assertions and truthiness checks when 0, false, or empty strings are valid.
- Structure and simplicity: Keep logic inline unless extraction improves reuse, readability, or separation of concerns.
- Avoid single-use helpers, unnecessary abstractions, duplicated state, dead code, and clever hacks.
- Keep changes small and follow existing project conventions.
- Functions and documentation: Keep function signatures on one line when they fit.
- Add JSDoc to JavaScript functions and to TypeScript functions when behavior is not clear from the types.
- Add a descriptive `// ...` comment before logic blocks, with one blank line before each comment.
- Keep a lookup and its immediate existence check in the same logic block; place the block comment before the lookup, not between the lookup and the `if` check.
- Use clear domain terminology, concise filenames, consistent plural model names, and namespaced APIs for related factories or facades.
- Avoid renaming imports unless it improves clarity.
- Inline simple single-use prop types and className expressions. Keep named prop types when shared or complex.
- Extract components only for meaningful UI boundaries.
- Avoid unnecessary cards, duplicated derived state, index-based keys, and effects that do not synchronize with external systems.
- Async and state: Prefer explicit async/await, handle every promise, use concurrency only when operations are independent, and clean up timers, listeners, subscriptions, and observers. Avoid global runtime-state changes unless unavoidable.
- Prefer established libraries for validation, routing, forms, dates, URLs, parsing, and internationalization when they simplify the implementation.
- Declare route response schemas and return raw domain objects or primitive values without reconstructing response models solely for validation.
- Do not add tests unless explicitly requested. Test the real implementation, avoid mocks where practical, and never duplicate production logic in tests.
- Run formatting, linting, type checking, and relevant existing tests, then review the implementation for further simplification.
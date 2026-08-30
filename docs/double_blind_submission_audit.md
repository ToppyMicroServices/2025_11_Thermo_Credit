# Double-Blind Submission Audit

Date: 2026-07-28

## Confirmed target and policy

First target: *Research Papers in Economics and Finance* (REF), using its
regular OJS submission route.

The journal's official instructions require a fully anonymized manuscript and
double-blind review by at least two independent reviewers after editorial
screening. Author names, affiliations, ORCID iDs, and e-mail addresses are
entered separately in OJS. The manuscript must be supplied as Microsoft Word,
OpenOffice, or RTF; REF does not list PDF as a main-manuscript format.

The same official page reports a 2025 acceptance rate of 37%, a desk-rejection
rate of 16%, and rejection after peer review of 47%. These are historical
journal-wide outcomes, not a paper-specific acceptance probability.

Official policy and statistics:
<https://journals.ue.poznan.pl/REF/about/submissions>

Fit-adjusted backup: regular submission to the *Review of Political Economy*,
which reports double-anonymized review and a 27% acceptance rate:
<https://www.tandfonline.com/journals/crpe20/about-this-journal>

The Japanese Political Economy is excluded despite its higher published rate
because it uses single-anonymized review and therefore fails the mandatory
blind-review condition.

## Blinded review package

Use these files for REF:

- `submission/ref_anonymous_manuscript.docx`
- `submission/highlights.txt`, if the system accepts optional highlights
- `submission/anonymous_replication_archive.zip`
- `submission/anonymous_replication_archive.zip.sha256`

Use `submission/ref_cover_letter.md` to prepare the editor-facing letter.
Enter author identities in OJS. Do not upload `output/pdf/title_page.pdf`
unless OJS explicitly requests a separate identifying file; the official
instructions instead direct authors to enter identifying information in the
submission system.

The following are local QA or backup artifacts, not REF's main manuscript:

- `output/pdf/anonymous_manuscript.pdf`
- `output/pdf/title_page.pdf`
- `tex/theory.tex`
- `submission/jfs_anonymous_manuscript_source.zip`

## Excluded from blinded upload

Do not upload:

- `README.md` or `README_JP.md`;
- `.git/`, `.github/`, `.vscode/`, `.beads/`, caches, or local build folders;
- `site/` or other public dashboard outputs;
- `tex/theory.html` or extracted-text files;
- a raw checkout of the repository;
- a title page, acknowledgements, or other file containing identifying author
  information; or
- venue-specific correspondence prepared for JFS or ROPE.

These files may expose project ownership, author identity, repository history,
or local-machine context even when the manuscript itself is anonymous.

## Required content and format audit

Record completion in `submission/ref_submission_checklist.md`. The final gate
must establish:

- no more than 8,000 words including tables, figures, footnotes, and
  references;
- an abstract of no more than 200 words, no more than five keywords, and
  mandatory JEL codes;
- 12-point type and 1.5 line spacing;
- numbered sections and APA 7th edition references;
- tables and figures placed in the text;
- editable equations and tables, with figures embedded in the Word file;
- a factually correct AI-use declaration;
- no identity in document content, properties, comments, tracked changes, or
  embedded metadata; and
- a successful full-page visual inspection of the rendered Word manuscript.

## Claim audit

The REF-facing package should preserve these boundaries:

- the paper constructs an auditable bridge from BOJ borrower-sector loan
  stocks to a credit-allocation coordinate;
- borrower-sector allocation is not identified as the purpose of a loan;
- mapping, coverage, positive versus signed net changes, and alternative
  allocations are the main empirical evidence;
- JGB yield, nominal GDP, and BOJ assets are proxy outcomes, not direct welfare
  or financial-stability outcomes;
- the Japan estimates do not establish external validity; and
- the pseudo-out-of-sample table is a use example whose current results do not
  establish incremental predictability.

## Residual risks and controls

The principal preventable desk risks are format noncompliance, non-APA
references, broken Word equations, and incomplete anonymization. These should
be resolved by the final preflight rather than by strengthening the empirical
claim.

The replication archive reproduces the bridge audits and pseudo-out-of-sample
tables from supplied processed public-data inputs. Do not describe it as
refetching raw BOJ data, recreating every figure, or reproducing the entire
working repository unless those capabilities are independently verified.

If the OJS workflow or an editor's current instruction conflicts with this
audit, record the conflict and follow the explicit current journal instruction.

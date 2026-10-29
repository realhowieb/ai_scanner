# Policy texts

Each file is one version of one customer-facing policy. The API (domain/legal.py) reads
them, fills in the `{placeholders}` from settings, hashes the result and records it in
`legal_policy_versions`. Checkout records which versions each buyer accepted.

* Front matter (`key: value` lines up to `---`): title, version, effective_date, status.
  `status: proposed` shows a "pending legal review" banner; set `approved` only after
  legal review.
* Body: `## ` starts a section, `- ` a list item, blank lines separate paragraphs.
* To change the wording of a version someone has already accepted, add a new file with
  the next version number. The API refuses an edited text for an accepted version.

Placeholders: `{legal_entity}` (HSF_LEGAL_ENTITY, default HSFinest LLC),
`{support_email}` (HSF_SUPPORT_EMAIL), `{governing_law}` (HSF_GOVERNING_LAW), plus the
page links `{terms_url}` (founding terms), `{refunds_url}`, `{privacy_url}`, `{site_terms_url}`.

`terms` (Terms of Service, /terms) and `privacy` are the general policies: they go public,
and are recorded at sign-up and at monthly/yearly checkout, once `HSF_LEGAL_PAGES_ENABLED=1`.
The founding pages keep `FOUNDING_LEGAL_PAGES_ENABLED`.

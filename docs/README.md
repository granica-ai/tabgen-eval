# Documentation sources

`source/` is the public Sphinx site. Its navigation and search contain only
curated user guides, metric pages, statistical methodology and API reference.
Build it with the [local documentation workflow](building.md).

Repository guides (`architecture.md`, `usage.md`, `methodology.md`,
`failure-modes.md`) retain engineering and maintenance context. They are outside
the public source tree and are not copied into the website. The failure-mode
catalogue remains at its stable path for test traceability.

For new metric pages, use the Shape and Linear C2ST guides as templates: purpose,
inputs, result key and interpretation, runnable usage, mathematical definition,
curated parameters, detailed output where supported, and limitations. Describe
user-visible behavior; keep implementation and optimizer mechanics in repository
engineering documentation. Statistical sampling and band interpretation belong in
`source/methodology.md`, with links from metric pages.

`source/sharp-bits.md` connects interpretation issues across metrics and links
to their authoritative definitions. Keep `source/troubleshooting.md` focused on
symptoms and remedies, and avoid duplicating mathematical definitions there.

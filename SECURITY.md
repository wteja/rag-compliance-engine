# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public GitHub issue.

Email **teja.weerayut@gmail.com** with:

- a description of the issue and its impact,
- steps to reproduce (or a proof of concept), and
- affected version / commit.

You can expect an acknowledgement within a few days. Please allow a reasonable
window to release a fix before any public disclosure.

## Scope

This is a portfolio/demonstration project. The security model it aims to uphold:

- **Access control is enforced at the data layer**, not in the prompt — a caller
  cannot retrieve chunks outside their groups or tenant.
- **Tenants are physically isolated** (separate vector collection/index per tenant).
- **PII is redacted on ingest and on output**, and the pipeline **fails closed** when
  a query cannot be audited or output-redacted.

Reports that demonstrate a way to bypass any of these are especially welcome.

## Not in scope

The demo defaults (`dev-secret-change-me...` JWT secret, unauthenticated local
OpenSearch/Ollama, recreate-the-DB migrations) are for local evaluation only and are
documented as such — they are not treated as vulnerabilities. Harden them before any
real deployment.

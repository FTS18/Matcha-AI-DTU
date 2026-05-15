# Maintenance & Security Policy

This document outlines the current architectural decisions regarding framework upgrades and security maintenance for the Matcha-AI-DTU project.

## Current Architectural Status

As of May 2026, the project has undergone a comprehensive stabilization and hardening phase. To ensure maximum reliability and performance of the core AI pipeline and visual reporting features, we have made the strategic decision to pin the following major versions:

| Component | Current Version | Maintenance Strategy |
| :--- | :--- | :--- |
| **Frontend Framework** | Next.js 14.2.x | Pinned (Stability over feature-parity) |
| **UI Runtime** | React 18.2.0 | Pinned (Peer dependency compatibility) |
| **Backend Framework** | NestJS 11.x | Pinned (Stability) |
| **Database ORM** | Prisma 5.22.x | Pinned (Environment compatibility) |

---

## Deferred Upgrades

The following upgrades are **NOT** in the immediate plan and are intentionally deferred until further notice:

### 1. Next.js 15/16 & React 19
While Next.js 16 is available, upgrading to it would force a migration to React 19.
*   **Rationale**: React 19 introduces breaking changes to the internal component life-cycle and type-definitions that are currently incompatible with critical third-party libraries used in this project, specifically `@react-pdf/renderer` (for match reports) and `react-dropzone` (for video uploads).
*   **Constraint**: Until these libraries provide stable, production-ready support for React 19, we will remain on the Next.js 14.2.x branch.

### 2. NestJS 11
Upgrading to NestJS 11 is deferred to maintain backend stability and avoid a full-scale refactor of decorators and module architectures.

---

## Security Posture

We maintain a "Green" status on our CI/CD pipeline and have addressed all critical local build and linting failures.

### Known Vulnerabilities
You may notice **5** High/Moderate vulnerabilities reported by `npm audit` (specifically regarding `next` and `postcss`). 
*   **Status**: These are known and documented. 
*   **Reason**: These vulnerabilities exist within the core source code of the Next.js 14 line and its required sub-dependencies. Since we are already on the latest available patch for the 14.x branch (`14.2.35`), these cannot be resolved without a major upgrade to Next.js 15/16.
*   **Mitigation**: We have prioritized **Functional Stability** and **Build Correctness**. The security risk is managed by ensuring the application environment is hardened and only necessary ports are exposed in production.

## Contribution Note
Contributors are requested **not** to attempt major version upgrades for the frameworks listed above without prior discussion and a proven strategy for resolving the dependency cascades (specifically the React 19 / PDF-renderer conflict).

---

*Last Updated: May 2026*
